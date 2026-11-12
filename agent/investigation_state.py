"""HOUND Investigation State & Session Persistence.

Maintains mutable state across iterations, tracks budget counters,
records evidence provenance, and supports session resumption.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field

from agent.agent_models import (
    FinalStatus,
    HypothesisRecord,
    InvestigationEvent,
    StoppingReason,
)
from core.configuration import AgentConfig, get_config
from core.logger import get_logger
from storage.database import DatabaseManager
from storage.evidence_store import EvidenceItem

logger = get_logger("hound.agent.state")


class InvestigationState:
    """Encapsulates the live state of an autonomous investigation session."""

    def __init__(
        self,
        investigation_id: str,
        config: Optional[AgentConfig] = None,
        db_manager: Optional[DatabaseManager] = None,
    ):
        self.investigation_id = investigation_id
        cfg = config or get_config().agent
        self.config = cfg
        self.db = db_manager

        # Counters & Budgets
        self.iteration: int = 0
        self.tool_calls_count: int = 0
        self.external_requests_count: int = 0
        self.start_time: float = time.perf_counter()
        self.max_iterations: int = cfg.max_iterations
        self.max_tool_calls: int = cfg.max_tool_calls
        self.max_external_requests: int = cfg.max_external_requests
        self.max_duration_seconds: int = cfg.max_duration_seconds

        # Action tracking & Deduplication
        self.executed_action_keys: Set[str] = set()

        # Evidence & Hypotheses
        self.evidence_items: Dict[str, EvidenceItem] = {}
        self.hypotheses: Dict[str, HypothesisRecord] = {}
        self.events: List[InvestigationEvent] = []

        # Clue and State Indicators
        self.clues: Dict[str, bool] = {
            "metadata_analyzed": False,
            "has_gps": False,
            "has_exif_timestamp": False,
            "ocr_executed": False,
            "has_ocr_road": False,
            "has_ocr_text": False,
            "infra_executed": False,
            "has_infrastructure": False,
            "solar_executed": False,
            "has_solar_shadow": False,
            "candidates_generated": False,
            "osm_searched": False,
            "terrain_checked": False,
        }

        # Terminal state
        self.stopping_reason: Optional[StoppingReason] = None
        self.final_status: Optional[FinalStatus] = None
        self.is_completed: bool = False

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.start_time

    def is_budget_exhausted(self) -> Tuple[bool, Optional[StoppingReason]]:
        """Check if any execution limit has been exceeded."""
        if self.iteration >= self.max_iterations:
            return True, "budget_reached"
        if self.tool_calls_count >= self.max_tool_calls:
            return True, "budget_reached"
        if self.external_requests_count >= self.max_external_requests:
            return True, "budget_reached"
        if self.elapsed_seconds >= self.max_duration_seconds:
            return True, "budget_reached"
        return False, None

    def record_action(self, action_key: str, is_external: bool = False) -> None:
        """Mark an action as executed and increment counters."""
        self.executed_action_keys.add(action_key)
        self.tool_calls_count += 1
        if is_external:
            self.external_requests_count += 1

    def is_action_already_executed(self, action_key: str) -> bool:
        """Check if identical action has already been performed."""
        return action_key in self.executed_action_keys

    def add_event(
        self,
        event_type: str,
        action_summary: str,
        result_summary: str,
        tool_id: Optional[str] = None,
        evidence_ids: Optional[List[str]] = None,
        duration_s: float = 0.0,
        error: Optional[str] = None,
    ) -> InvestigationEvent:
        """Log a structured investigation event."""
        event = InvestigationEvent(
            investigation_id=self.investigation_id,
            iteration=self.iteration,
            event_type=event_type,
            tool_id=tool_id,
            action_summary=action_summary,
            result_summary=result_summary,
            evidence_ids=evidence_ids or [],
            duration_s=duration_s,
            error=error,
        )
        self.events.append(event)

        # Persist event immediately if database is configured
        if self.db:
            try:
                with self.db.get_connection() as conn:
                    conn.execute(
                        """
                        INSERT INTO investigation_events (
                            investigation_id, iteration, timestamp, event_type,
                            tool_id, action_summary, result_summary, evidence_ids,
                            duration_s, error
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            event.investigation_id,
                            event.iteration,
                            event.timestamp,
                            event.event_type,
                            event.tool_id,
                            event.action_summary,
                            event.result_summary,
                            json.dumps(event.evidence_ids),
                            event.duration_s,
                            event.error,
                        ),
                    )
                    conn.commit()
            except Exception as e:
                logger.debug(f"Failed to persist event: {e}")

        return event

    def save_session(self) -> None:
        """Persist current session state and hypotheses to database."""
        if not self.db:
            return

        now_iso = datetime.now(timezone.utc).isoformat()
        state_dict = {
            "iteration": self.iteration,
            "tool_calls_count": self.tool_calls_count,
            "external_requests_count": self.external_requests_count,
            "clues": self.clues,
            "executed_action_keys": list(self.executed_action_keys),
            "stopping_reason": self.stopping_reason,
            "final_status": self.final_status,
        }

        try:
            with self.db.get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO agent_sessions (
                        investigation_id, status, iteration, tool_calls_count,
                        external_requests_count, started_at, updated_at,
                        completed_at, stopping_reason, final_status, state_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(investigation_id) DO UPDATE SET
                        status = excluded.status,
                        iteration = excluded.iteration,
                        tool_calls_count = excluded.tool_calls_count,
                        external_requests_count = excluded.external_requests_count,
                        updated_at = excluded.updated_at,
                        completed_at = excluded.completed_at,
                        stopping_reason = excluded.stopping_reason,
                        final_status = excluded.final_status,
                        state_json = excluded.state_json
                    """,
                    (
                        self.investigation_id,
                        "completed" if self.is_completed else "running",
                        self.iteration,
                        self.tool_calls_count,
                        self.external_requests_count,
                        now_iso,
                        now_iso,
                        now_iso if self.is_completed else None,
                        self.stopping_reason,
                        self.final_status,
                        json.dumps(state_dict),
                    ),
                )

                # Persist hypotheses
                for hyp in self.hypotheses.values():
                    conn.execute(
                        """
                        INSERT INTO agent_hypotheses (
                            id, investigation_id, name, level, lat, lon, radius_m,
                            status, heuristic_score, seed_evidence_ids,
                            supporting_evidence_ids, contradictory_evidence_ids,
                            untested_constraints, score_breakdown_json,
                            rejection_reason, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            status = excluded.status,
                            heuristic_score = excluded.heuristic_score,
                            supporting_evidence_ids = excluded.supporting_evidence_ids,
                            contradictory_evidence_ids = excluded.contradictory_evidence_ids,
                            untested_constraints = excluded.untested_constraints,
                            score_breakdown_json = excluded.score_breakdown_json,
                            rejection_reason = excluded.rejection_reason,
                            updated_at = excluded.updated_at
                        """,
                        (
                            hyp.id,
                            self.investigation_id,
                            hyp.name,
                            hyp.level,
                            hyp.lat,
                            hyp.lon,
                            hyp.radius_m,
                            hyp.status,
                            hyp.heuristic_score,
                            json.dumps(hyp.seed_evidence_ids),
                            json.dumps(hyp.supporting_evidence_ids),
                            json.dumps(hyp.contradictory_evidence_ids),
                            json.dumps(hyp.untested_constraints),
                            hyp.score_breakdown.model_dump_json(),
                            hyp.rejection_reason,
                            hyp.created_at,
                            hyp.updated_at,
                        ),
                    )
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to persist agent session to database: {e}")

    @classmethod
    def load_session(
        cls,
        investigation_id: str,
        db_manager: DatabaseManager,
        config: Optional[AgentConfig] = None,
    ) -> Optional[InvestigationState]:
        """Restore an investigation session from database if one exists."""
        state = cls(investigation_id, config=config, db_manager=db_manager)
        try:
            with db_manager.get_connection() as conn:
                cur = conn.execute(
                    "SELECT * FROM agent_sessions WHERE investigation_id = ?",
                    (investigation_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None

                state.iteration = row["iteration"]
                state.tool_calls_count = row["tool_calls_count"]
                state.external_requests_count = row["external_requests_count"]
                state.stopping_reason = row["stopping_reason"]
                state.final_status = row["final_status"]
                state.is_completed = row["status"] == "completed"

                if row["state_json"]:
                    data = json.loads(row["state_json"])
                    state.clues.update(data.get("clues", {}))
                    state.executed_action_keys = set(data.get("executed_action_keys", []))

                # Load events
                evt_cur = conn.execute(
                    "SELECT * FROM investigation_events WHERE investigation_id = ? ORDER BY id ASC",
                    (investigation_id,),
                )
                for erow in evt_cur.fetchall():
                    ev_ids = json.loads(erow["evidence_ids"]) if erow["evidence_ids"] else []
                    state.events.append(
                        InvestigationEvent(
                            event_id=f"EVT-{erow['id']}",
                            investigation_id=investigation_id,
                            iteration=erow["iteration"],
                            timestamp=erow["timestamp"],
                            event_type=erow["event_type"],
                            tool_id=erow["tool_id"],
                            action_summary=erow["action_summary"],
                            result_summary=erow["result_summary"],
                            evidence_ids=ev_ids,
                            duration_s=erow["duration_s"],
                            error=erow["error"],
                        )
                    )

                # Load hypotheses
                hyp_cur = conn.execute(
                    "SELECT * FROM agent_hypotheses WHERE investigation_id = ?",
                    (investigation_id,),
                )
                for hrow in hyp_cur.fetchall():
                    breakdown = None
                    if hrow["score_breakdown_json"]:
                        from agent.candidate_comparator import ScoreBreakdown
                        breakdown = ScoreBreakdown.model_validate_json(hrow["score_breakdown_json"])

                    hyp = HypothesisRecord(
                        id=hrow["id"],
                        investigation_id=investigation_id,
                        name=hrow["name"],
                        level=hrow["level"],
                        lat=hrow["lat"],
                        lon=hrow["lon"],
                        radius_m=hrow["radius_m"],
                        status=hrow["status"],
                        heuristic_score=hrow["heuristic_score"],
                        score_breakdown=breakdown or ScoreBreakdown(),
                        seed_evidence_ids=json.loads(hrow["seed_evidence_ids"] or "[]"),
                        supporting_evidence_ids=json.loads(hrow["supporting_evidence_ids"] or "[]"),
                        contradictory_evidence_ids=json.loads(hrow["contradictory_evidence_ids"] or "[]"),
                        untested_constraints=json.loads(hrow["untested_constraints"] or "[]"),
                        rejection_reason=hrow["rejection_reason"],
                        created_at=hrow["created_at"],
                        updated_at=hrow["updated_at"],
                    )
                    state.hypotheses[hyp.id] = hyp

                return state
        except Exception as e:
            logger.warning(f"Error restoring agent session: {e}")
            return None
