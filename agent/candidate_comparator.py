"""HOUND Geographic Candidate Comparator.

Compares, scores, and re-ranks candidate hypotheses based on evidence support,
geometric consistency, terrain correlation, and active falsification criteria.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from agent.candidate_generator import GeographicCandidate
from core.logger import get_logger

logger = get_logger("hound.agent.comparator")

# Specificity level weight multiplier
LEVEL_WEIGHTS = {
    "point": 5.0,
    "building": 4.0,
    "road": 3.0,
    "area": 2.5,
    "city": 2.0,
    "admin": 1.5,
    "region": 1.0,
}


class CandidateEvaluation(BaseModel):
    candidate_id: str
    name: str
    level: str
    base_score: float
    composite_score: float
    rank: int = 1
    supporting_count: int = 0
    contradictory_count: int = 0
    status: str = "active"
    supporting_reasons: List[str] = Field(default_factory=list)
    contradictory_reasons: List[str] = Field(default_factory=list)
    ranking_justification: str = ""


class CandidateComparator:
    """Ranks and contrasts candidate hypotheses with transparent mathematical scoring."""

    @classmethod
    def compare_and_rank(
        cls,
        candidates: List[GeographicCandidate],
        terrain_scores: Optional[Dict[str, float]] = None,
    ) -> List[CandidateEvaluation]:
        """Rank candidates, penalize contradictions, and generate comparative explanations."""
        terrain_scores = terrain_scores or {}
        evaluations: List[CandidateEvaluation] = []

        for c in candidates:
            sup_count = len(c.supporting_evidence_ids)
            contra_count = len(c.contradictory_evidence_ids)

            # Specificity multiplier
            level_mult = LEVEL_WEIGHTS.get(c.level, 1.0)

            # Terrain correlation bonus (if computed)
            t_score = terrain_scores.get(c.id, 0.0)
            terrain_mult = 1.0 + (t_score * 2.0) if t_score > 0 else 1.0

            # Base score calculation
            score = c.score * level_mult * terrain_mult

            # Contradiction penalty: severe down-weighting
            if contra_count > 0 or c.status == "excluded":
                score = score * (0.10**contra_count)
                status = "excluded"
            else:
                status = "active"

            sup_reasons = [f"Corroborated by {sup_count} independent observation(s)."]
            if c.ranking_explanation:
                sup_reasons.append(c.ranking_explanation)
            if t_score > 0:
                sup_reasons.append(f"Synthetic DEM horizon correlation score: {t_score:.3f}")

            contra_reasons = []
            if c.falsification_condition:
                contra_reasons.append(c.falsification_condition)
            if contra_count > 0:
                contra_reasons.append(f"Penalized by {contra_count} contradictory constraint(s).")

            evaluations.append(
                CandidateEvaluation(
                    candidate_id=c.id,
                    name=c.name,
                    level=c.level,
                    base_score=round(c.score, 2),
                    composite_score=round(score, 3),
                    supporting_count=sup_count,
                    contradictory_count=contra_count,
                    status=status,
                    supporting_reasons=sup_reasons,
                    contradictory_reasons=contra_reasons,
                )
            )

        # Sort: active candidates first, ordered by descending composite score
        evaluations.sort(
            key=lambda e: (0 if e.status == "active" else 1, -e.composite_score)
        )

        # Assign ranks and generate comparative explanations
        for idx, ev in enumerate(evaluations):
            ev.rank = idx + 1
            if idx == 0:
                ev.ranking_justification = (
                    f"Top-ranked hypothesis with composite score {ev.composite_score:.2f} "
                    f"({ev.level.upper()} specificity, {ev.supporting_count} corroborating observations, 0 contradictions)."
                )
            else:
                prev = evaluations[idx - 1]
                diff = prev.composite_score - ev.composite_score
                if ev.status == "excluded":
                    ev.ranking_justification = (
                        f"Ranked below #{prev.rank} ({prev.name}) due to active exclusion: "
                        f"{'; '.join(ev.contradictory_reasons)}"
                    )
                else:
                    ev.ranking_justification = (
                        f"Ranked below #{prev.rank} ({prev.name}) by score margin -{diff:.2f} "
                        f"(lower specificity level '{ev.level}' or fewer corroborating constraints)."
                    )

        return evaluations
