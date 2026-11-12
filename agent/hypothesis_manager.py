"""HOUND Hypothesis Manager.

Maintains competing geographic hypotheses, tracks the hypothesis lifecycle,
enforces non-circular scoring, and preserves all contradictory evidence.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from agent.agent_models import HypothesisRecord, HypothesisStatus, VerificationQuestion
from agent.candidate_comparator import LEVEL_WEIGHTS, ScoreBreakdown
from core.logger import get_logger
from storage.database import DatabaseManager
from storage.evidence_store import CandidateItem, EvidenceItem, EvidenceStore

logger = get_logger("hound.agent.hypotheses")


class HypothesisManager:
    """Oversees competing geographic candidates throughout the investigation."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager

    def register_candidate(
        self,
        investigation_id: str,
        name: str,
        level: str,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        radius_m: Optional[float] = None,
        seed_evidence_id: Optional[str] = None,
        seed_evidence_kind: Optional[str] = None,
        initial_score: float = 1.0,
    ) -> HypothesisRecord:
        """Create and register a new candidate hypothesis."""
        import uuid

        hyp_id = f"H-{uuid.uuid4().hex[:6].upper()}"
        breakdown = ScoreBreakdown(
            base_seed_score=initial_score,
            seed_clue_description=f"Seed: {seed_evidence_kind or 'Initial observation'}",
            level_specificity_multiplier=LEVEL_WEIGHTS.get(level, 1.0),
            score_type="heuristic_score",
        )

        hyp = HypothesisRecord(
            id=hyp_id,
            investigation_id=investigation_id,
            name=name,
            level=level,
            lat=lat,
            lon=lon,
            radius_m=radius_m,
            status="proposed",
            heuristic_score=initial_score * LEVEL_WEIGHTS.get(level, 1.0),
            score_breakdown=breakdown,
            seed_evidence_ids=[seed_evidence_id] if seed_evidence_id else [],
            uncertainty_description=f"Initial hypothesis at {level} scale.",
        )

        if self.db:
            try:
                with self.db.get_connection() as conn:
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
                            heuristic_score = excluded.heuristic_score
                        """,
                        (
                            hyp.id,
                            hyp.investigation_id,
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
                logger.debug(f"Failed to persist hypothesis: {e}")

        return hyp

    def add_corroboration(
        self,
        hypothesis: HypothesisRecord,
        evidence_item: EvidenceItem,
        corroboration_bonus: float = 3.0,
    ) -> None:
        """Add an independent corroborating evidence item (strictly non-circular)."""
        # Guard against circular reinforcement
        if evidence_item.id in hypothesis.seed_evidence_ids:
            logger.debug(f"Ignoring seed evidence {evidence_item.id} as corroboration for {hypothesis.name} (circular).")
            return

        if evidence_item.id not in hypothesis.supporting_evidence_ids:
            hypothesis.supporting_evidence_ids.append(evidence_item.id)

            # Recompute heuristic score
            b = hypothesis.score_breakdown
            b.independent_corroborations_count = len(hypothesis.supporting_evidence_ids)
            b.corroboration_bonus += corroboration_bonus * evidence_item.confidence_weight

            hypothesis.heuristic_score = (
                (b.base_seed_score + b.corroboration_bonus + b.verified_terrain_bonus)
                * b.level_specificity_multiplier
                * b.contradiction_penalty_factor
            )

            if hypothesis.status == "proposed":
                hypothesis.status = "active"
            if len(hypothesis.supporting_evidence_ids) >= 2 and not hypothesis.contradictory_evidence_ids:
                hypothesis.status = "supported"

            hypothesis.updated_at = datetime.now(timezone.utc).isoformat()

    def record_contradiction(
        self,
        hypothesis: HypothesisRecord,
        evidence_item: EvidenceItem,
        rejection_reason: str,
        is_definitive_falsification: bool = True,
    ) -> None:
        """Record an active contradiction against a candidate."""
        if evidence_item.id not in hypothesis.contradictory_evidence_ids:
            hypothesis.contradictory_evidence_ids.append(evidence_item.id)

        hypothesis.rejection_reason = rejection_reason
        b = hypothesis.score_breakdown

        if is_definitive_falsification:
            # Active physical falsification (e.g. solar ceiling violation)
            hypothesis.status = "rejected"
            b.contradiction_penalty_factor = 0.0
            hypothesis.heuristic_score = 0.0
            logger.info(f"Hypothesis '{hypothesis.name}' REJECTED: {rejection_reason}")
        else:
            # Downweighting contradiction (e.g. minor discrepancy)
            hypothesis.status = "challenged"
            b.contradiction_penalty_factor *= 0.20
            hypothesis.heuristic_score *= 0.20
            logger.info(f"Hypothesis '{hypothesis.name}' CHALLENGED: {rejection_reason}")

        hypothesis.updated_at = datetime.now(timezone.utc).isoformat()

    def attach_verification_question(
        self,
        hypothesis: HypothesisRecord,
        question: VerificationQuestion,
    ) -> None:
        """Attach a verification question to the hypothesis record."""
        hypothesis.verification_questions.append(question)
        hypothesis.remaining_checks.append(question.question_text)
        hypothesis.updated_at = datetime.now(timezone.utc).isoformat()

    def resolve_verification_question(
        self,
        hypothesis: HypothesisRecord,
        question_id: str,
        status: str,
        justification: str,
        evidence_id: Optional[str] = None,
    ) -> None:
        """Mark a verification check as confirmed, contradicted, or data_unavailable."""
        vq = next((q for q in hypothesis.verification_questions if q.question_id == question_id), None)
        if not vq:
            return

        vq.status = status
        vq.justification = justification
        vq.evidence_id = evidence_id

        if vq.question_text in hypothesis.remaining_checks:
            hypothesis.remaining_checks.remove(vq.question_text)
        hypothesis.completed_checks.append(f"{vq.question_text} -> {status.upper()}")

        hypothesis.updated_at = datetime.now(timezone.utc).isoformat()
