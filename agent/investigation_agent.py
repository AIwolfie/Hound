"""HOUND Autonomous Investigation Agent.

Orchestrates the entire multi-modal investigation lifecycle:
plans actions, executes allowlisted tools, updates hypotheses,
runs adversarial checks, logs telemetry, and compiles defensible dossiers.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional
from pydantic import BaseModel

from agent.agent_models import (
    FinalInvestigationDossier,
    FinalStatus,
    HypothesisRecord,
    InvestigationAction,
    InvestigationEvent,
    StoppingReason,
    ToolResult,
)
from agent.hypothesis_manager import HypothesisManager
from agent.investigation_state import InvestigationState
from agent.llm_provider import BaseLLMProvider, get_llm_provider
from agent.planner import InvestigationPlanner
from agent.stopping_policy import StoppingPolicy
from agent.tool_registry import ToolExecutionContext, ToolRegistry
from agent.verification_engine import VerificationEngine
from core.configuration import AgentConfig, get_config
from core.logger import get_logger
from storage.database import DatabaseManager
from storage.evidence_store import EvidenceItem, EvidenceStore
from storage.investigation_store import InvestigationRecord, InvestigationStore

logger = get_logger("hound.agent.core")


class InvestigationAgent:
    """Autonomous agent conducting disciplined OSINT visual investigations."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        config: Optional[AgentConfig] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
    ):
        cfg = get_config()
        self.db = db_manager or DatabaseManager(cfg.paths.database_path)
        self.config = config or cfg.agent
        self.inv_store = InvestigationStore(self.db, cfg.paths.storage_dir)
        self.evid_store = EvidenceStore(self.db)

        # Core Subsystems
        self.tools = ToolRegistry()
        self.llm = llm_provider or get_llm_provider(self.config)
        self.planner = InvestigationPlanner(self.tools, llm_provider=self.llm)
        self.hyp_mgr = HypothesisManager(db_manager=self.db)
        self.verifier = VerificationEngine(self.hyp_mgr, self.evid_store)

    def investigate(
        self,
        investigation_id: str,
        resume: bool = False,
        max_iterations: Optional[int] = None,
        offline: Optional[bool] = None,
        verbose: bool = False,
    ) -> FinalInvestigationDossier:
        """Execute the autonomous multi-modal investigation loop."""
        inv_record = self.inv_store.get_investigation(investigation_id)
        if not inv_record:
            raise ValueError(f"Investigation session '{investigation_id}' does not exist.")

        logger.info(f"Starting autonomous investigation for {inv_record.filename} ({investigation_id})...")

        # 1. Initialize or Resume Investigation State
        state: Optional[InvestigationState] = None
        if resume:
            state = InvestigationState.load_session(investigation_id, self.db, self.config)
            if state:
                logger.info(f"Resuming existing session at iteration {state.iteration}...")

        if state is None:
            state = InvestigationState(investigation_id, config=self.config, db_manager=self.db)

        # Apply runtime overrides
        if max_iterations is not None:
            state.max_iterations = max_iterations
        if offline is not None:
            state.config.offline_mode = offline

        # Load initial pre-existing evidence from database
        existing_ev = self.evid_store.get_evidence(investigation_id)
        for ev in existing_ev:
            state.evidence_items[ev.id] = ev
            self._update_clues_from_evidence(state, ev)

        # Load initial pre-existing hypotheses from database
        try:
            with self.db.get_connection() as conn:
                cur = conn.execute("SELECT * FROM agent_hypotheses WHERE investigation_id = ?", (investigation_id,))
                for hrow in cur.fetchall():
                    if hrow["id"] not in state.hypotheses:
                        import json
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
            if state.hypotheses:
                state.clues["candidates_generated"] = True
        except Exception as e:
            logger.debug(f"Could not load initial hypotheses: {e}")

        ctx = ToolExecutionContext(
            investigation_id=investigation_id,
            db_manager=self.db,
            offline_mode=state.config.offline_mode,
        )

        state.add_event(
            event_type="investigation_started",
            action_summary="Autonomous agent session initialized.",
            result_summary=f"Loaded {len(existing_ev)} pre-existing evidence item(s).",
        )

        # 2. Main Autonomous Execution Loop
        while True:
            state.iteration += 1

            # A. Check Stopping Conditions
            should_stop, stop_reason, final_status = StoppingPolicy.evaluate(state)
            if should_stop:
                state.stopping_reason = stop_reason
                state.final_status = final_status
                state.is_completed = True
                state.add_event(
                    event_type="investigation_stopped",
                    action_summary="Stopping policy triggered.",
                    result_summary=f"Reason: {stop_reason}, Final status: {final_status}.",
                )
                break

            # B. Plan Next Action
            action = self.planner.select_next_action(state)
            if not action:
                logger.info("No further actions selected by planner; investigation paths exhausted.")
                # Evaluate whether we have viable candidates or must abstain
                active = [h for h in state.hypotheses.values() if h.status in ("active", "supported")]
                if active:
                    state.stopping_reason = "paths_exhausted"
                    state.final_status = "candidates_remaining"
                else:
                    state.stopping_reason = "insufficient_evidence"
                    state.final_status = "insufficient_evidence"
                state.is_completed = True
                break

            # C. Execute Selected Tool
            act_signature = f"{action.tool_id}:{action.params}"
            state.record_action(act_signature, is_external=action.is_external_request)

            t0 = time.perf_counter()
            tool_result = self.tools.execute_tool(action.tool_id, ctx, action.params)
            duration_s = time.perf_counter() - t0

            # D. Ingest Evidence & Update Clues
            for ev in tool_result.evidence_items:
                state.evidence_items[ev.id] = ev
                self._update_clues_from_evidence(state, ev)

            # E. Process Tool Output into Hypotheses
            self._process_tool_results_into_state(state, action, tool_result)

            # F. Log Event
            state.add_event(
                event_type="tool_executed",
                tool_id=action.tool_id,
                action_summary=f"Executed '{action.tool_id}': {action.reason}",
                result_summary=f"Status: {tool_result.status}, Findings: {list(tool_result.output_data.keys())}",
                evidence_ids=[e.id for e in tool_result.evidence_items],
                duration_s=duration_s,
                error=tool_result.error_message,
            )

            # G. Persist State Progress
            state.save_session()

        # 3. Final Re-ranking and Dossier Compilation
        state.save_session()
        dossier = self._compile_dossier(state, inv_record)

        # Update parent investigation record in database
        self.inv_store.update_status(
            investigation_id,
            status=dossier.final_status,
            summary=(
                f"Autonomous investigation completed ({dossier.final_status}). "
                f"Top candidate: {dossier.primary_hypothesis.name if dossier.primary_hypothesis else 'None'} "
                f"(Precision: {dossier.geographic_precision_justified})."
            ),
        )

        return dossier

    # =========================================================================
    # INTERNAL LOGIC HELPERS
    # =========================================================================

    def _update_clues_from_evidence(self, state: InvestigationState, ev: EvidenceItem) -> None:
        """Update clue presence flags based on evidence item."""
        text_lower = ev.text.lower()

        if ev.kind == "gps" or "gps" in text_lower:
            state.clues["has_gps"] = True
        if ev.kind == "exif" or "camera" in text_lower or "timestamp" in text_lower:
            state.clues["metadata_analyzed"] = True
            state.clues["has_exif_timestamp"] = True
        if ev.kind == "ocr":
            state.clues["ocr_executed"] = True
            state.clues["has_ocr_text"] = True
            if any(k in text_lower for k in ("road", "highway", "route", "route-sign", "shield")):
                state.clues["has_ocr_road"] = True
        if ev.kind in ("infrastructure", "infrastructure_detector"):
            state.clues["infra_executed"] = True
            state.clues["has_infrastructure"] = True
        if ev.kind in ("solar", "shadow"):
            state.clues["solar_executed"] = True
            state.clues["has_solar_shadow"] = True
        if ev.kind in ("ridge", "skyline", "terrain"):
            state.clues["has_skyline"] = True
            state.clues["terrain_checked"] = True
        if ev.kind == "osm":
            state.clues["osm_searched"] = True

    def _process_tool_results_into_state(
        self,
        state: InvestigationState,
        action: InvestigationAction,
        result: ToolResult,
    ) -> None:
        """Map tool results to hypothesis updates and state flags."""
        if action.tool_id == "metadata_exif":
            state.clues["metadata_analyzed"] = True
            if result.output_data.get("has_gps"):
                state.clues["has_gps"] = True
            if result.output_data.get("has_timestamp"):
                state.clues["has_exif_timestamp"] = True

        elif action.tool_id == "image_preprocessing":
            state.clues["preprocessed"] = True

        elif action.tool_id == "ocr_extraction":
            state.clues["ocr_executed"] = True
            if result.output_data.get("road_designations"):
                state.clues["has_ocr_road"] = True

        elif action.tool_id == "infrastructure_detection":
            state.clues["infra_executed"] = True
            if result.output_data.get("detections_count", 0) > 0:
                state.clues["has_infrastructure"] = True

        elif action.tool_id == "solar_shadow":
            state.clues["solar_executed"] = True
            if result.output_data.get("shadow_clues_count", 0) > 0:
                state.clues["has_solar_shadow"] = True

        elif action.tool_id == "candidate_generation":
            state.clues["candidates_generated"] = True
            top_cands = result.output_data.get("top_candidates", [])
            for c in top_cands:
                if c["id"] not in state.hypotheses:
                    hyp = self.hyp_mgr.register_candidate(
                        investigation_id=state.investigation_id,
                        name=c["name"],
                        level=c["level"],
                        lat=c["lat"],
                        lon=c["lon"],
                        initial_score=c.get("score", 1.0),
                    )
                    hyp.status = "active"
                    state.hypotheses[hyp.id] = hyp

        elif action.tool_id == "adversarial_solar_check":
            cand_id = action.target_hypothesis_id
            if cand_id and cand_id in state.hypotheses:
                cand = state.hypotheses[cand_id]
                status_str = result.output_data.get("status")
                if status_str == "falsified":
                    obs_elev = result.output_data.get("observed_elevation", 0.0)
                    ceiling = result.output_data.get("max_ceiling", 0.0)
                    rejection_msg = f"Solar elevation {obs_elev:.1f}° exceeds physical ceiling {ceiling:.1f}° at lat {cand.lat:.2f}°."
                    ev_item = result.evidence_items[0] if result.evidence_items else EvidenceItem(
                        investigation_id=state.investigation_id,
                        kind="falsification",
                        status="falsified",
                        text=rejection_msg,
                        confidence_weight=1.0,
                    )
                    self.hyp_mgr.record_contradiction(
                        hypothesis=cand,
                        evidence_item=ev_item,
                        rejection_reason=rejection_msg,
                        is_definitive_falsification=True,
                    )
                    state.add_event(
                        event_type="hypothesis_falsified",
                        tool_id=action.tool_id,
                        action_summary=f"Solar ceiling test falsified '{cand.name}'.",
                        result_summary=f"Observed sun elevation exceeds ceiling at lat {cand.lat:.2f}°.",
                    )

        elif action.tool_id in ("osm_road_search", "dem_horizon_profile", "ridge_matching", "adversarial_osm_check"):
            cand_id = action.target_hypothesis_id
            if cand_id and cand_id in state.hypotheses:
                cand = state.hypotheses[cand_id]
                if not result.success or result.status in ("service_unavailable", "skipped", "error"):
                    note = f"{action.tool_id}: {result.error_message or result.status}"
                    if note not in cand.untested_constraints:
                        cand.untested_constraints.append(note)
                else:
                    for ev in result.evidence_items:
                        self.hyp_mgr.add_corroboration(cand, ev)

    def _compile_dossier(
        self,
        state: InvestigationState,
        inv_record: InvestigationRecord,
    ) -> FinalInvestigationDossier:
        """Compile the final structured investigation dossier."""
        all_hypotheses = list(state.hypotheses.values())
        all_hypotheses.sort(key=lambda h: h.heuristic_score, reverse=True)

        primary = all_hypotheses[0] if all_hypotheses and all_hypotheses[0].status != "rejected" else None
        competing = [h for h in all_hypotheses[1:] if h.status != "rejected"]
        rejected = [h for h in all_hypotheses if h.status == "rejected"]

        # Determine justified precision level
        if primary and primary.status in ("active", "supported"):
            level_map = {
                "point": "Point (<=100m)",
                "building": "Building",
                "road": "Road Corridor",
                "area": "District / Local Area",
                "city": "City / Municipality",
                "admin": "Administrative Division",
                "region": "Broad Geographic Region",
            }
            precision = level_map.get(primary.level, "Unlocalized")
        else:
            precision = "Unlocalized"

        # Separate facts from inferences
        facts: List[str] = [
            f"Image SHA-256 fingerprint: {inv_record.sha256}",
            f"Original filename: {inv_record.filename}",
        ]
        inferences: List[str] = []

        for ev in state.evidence_items.values():
            if ev.kind in ("exif", "gps"):
                facts.append(f"[Fact] {ev.text}")
            elif ev.kind == "ocr":
                facts.append(f"[Observed Text] {ev.text}")
            elif ev.kind in ("terrain", "ridge_match", "falsification", "contradiction"):
                inferences.append(f"[Inference/Computation] {ev.text}")

        abstention_reasons: List[str] = []
        if state.final_status == "insufficient_evidence":
            abstention_reasons.append("Image visual features contain no discernible geographic landmarks, text signage, or unique topographies.")
            abstention_reasons.append("EXIF metadata contains no embedded WGS-84 coordinates.")
        elif state.final_status == "inconclusive":
            abstention_reasons.append("Competing hypotheses could not be separated with high confidence.")

        all_untested: List[str] = []
        if primary and primary.untested_constraints:
            all_untested.extend(primary.untested_constraints)
        for h in all_hypotheses:
            for uc in h.untested_constraints:
                if uc not in all_untested:
                    all_untested.append(uc)

        return FinalInvestigationDossier(
            investigation_id=state.investigation_id,
            target_filename=inv_record.filename,
            sha256_hash=inv_record.sha256,
            image_resolution="1920x1080",
            final_status=state.final_status or "candidates_remaining",
            stopping_reason=state.stopping_reason or "paths_exhausted",
            geographic_precision_justified=precision,
            total_iterations=state.iteration,
            total_tool_calls=state.tool_calls_count,
            total_duration_s=state.elapsed_seconds,
            events_count=len(state.events),
            primary_hypothesis=primary,
            competing_hypotheses=competing,
            rejected_hypotheses=rejected,
            facts=facts,
            inferences=inferences,
            untested_constraints=all_untested,
            abstention_reasons=abstention_reasons,
            data_sources_used=["OpenStreetMap", "AWS Terrarium DEM", "NOAA Solar Ephemeris", "Nominatim Gazetteer"],
            known_limitations=[
                "Overpass public mirrors are subject to occasional rate-limits and timeouts.",
                "Sun ephemeris provides latitude bounds but requires camera heading to restrict longitude.",
            ],
            timeline=state.events,
        )
