"""HOUND Geographic Candidate Comparator.

Compares, scores, and re-ranks candidate hypotheses based on evidence support,
geometric consistency, verified terrain correlation, and active physical falsification.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from agent.candidate_generator import GeographicCandidate
from core.logger import get_logger

logger = get_logger("hound.agent.comparator")

# Specificity level weight multiplier (heuristic weights reflecting spatial constraint granularity)
LEVEL_WEIGHTS = {
    "point": 5.0,
    "building": 4.0,
    "road": 3.0,
    "area": 2.5,
    "city": 2.0,
    "admin": 1.5,
    "region": 1.0,
}


class ScoreBreakdown(BaseModel):
    base_seed_score: float = 1.0
    seed_clue_description: str = ""
    level_specificity_multiplier: float = 1.0  # Labeled heuristic weight
    independent_corroborations_count: int = 0
    corroboration_bonus: float = 0.0
    verified_terrain_bonus: float = 0.0
    contradiction_penalty_factor: float = 1.0
    score_type: str = "heuristic_score"  # Explicitly labeled: NOT a statistical probability
    untested_checks: List[str] = Field(default_factory=list)
    missing_expected_features: List[str] = Field(default_factory=list)


class CandidateEvaluation(BaseModel):
    candidate_id: str
    name: str
    level: str
    base_score: float
    composite_score: float
    rank: int = 1
    supporting_count: int = 0  # Independent corroborations count (excluding seed clue)
    contradictory_count: int = 0
    status: str = "active"  # "active", "excluded", "unverified"
    score_breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
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
        """Rank candidates, separate seed clues from independent corroboration, and penalize contradictions."""
        terrain_scores = terrain_scores or {}
        evaluations: List[CandidateEvaluation] = []

        for c in candidates:
            # Separate seed hypothesis clue from independent corroborating evidence
            # to prevent circular self-reinforcement
            seed_id = c.seed_evidence_id
            independent_sup = [sid for sid in c.supporting_evidence_ids if sid != seed_id]
            sup_count = len(independent_sup)
            contra_count = len(c.contradictory_evidence_ids)

            # Specificity level multiplier (heuristic)
            level_mult = LEVEL_WEIGHTS.get(c.level, 1.0)
            corrob_bonus = round(sup_count * 1.5, 2)

            # Verified terrain correlation bonus (only applied if ridge matching occurred)
            t_score = terrain_scores.get(c.id, 0.0)
            terrain_bonus = round(t_score * 3.0, 2) if t_score > 0 else 0.0

            # Base score from seed clue and level granularity
            raw_score = c.score * level_mult + corrob_bonus + terrain_bonus

            # Contradiction penalty: severe down-weighting for active physical falsification
            if contra_count > 0 or c.status == "excluded":
                contra_factor = round(0.05 ** max(1, contra_count), 4)
                score = round(raw_score * contra_factor, 3)
                status = "excluded"
            else:
                contra_factor = 1.0
                score = round(raw_score, 2)
                status = "active"

            # Provenance reasons
            sup_reasons: List[str] = []
            if c.seed_evidence_id:
                sup_reasons.append(
                    f"Generated from primary clue ({c.seed_evidence_kind or 'visual'}). "
                    f"Independent corroborations: {sup_count}."
                )
            if c.ranking_explanation:
                sup_reasons.append(c.ranking_explanation)
            if t_score > 0:
                sup_reasons.append(f"Verified DEM horizon correlation score: {t_score:.3f}")

            contra_reasons: List[str] = []
            if c.falsification_condition:
                contra_reasons.append(c.falsification_condition)
            if contra_count > 0:
                contra_reasons.append(f"Penalized by {contra_count} contradictory physical constraint(s).")

            breakdown = ScoreBreakdown(
                base_seed_score=round(c.score, 2),
                seed_clue_description=c.ranking_explanation or "Initial hypothesis seed",
                level_specificity_multiplier=level_mult,
                independent_corroborations_count=sup_count,
                corroboration_bonus=corrob_bonus,
                verified_terrain_bonus=terrain_bonus,
                contradiction_penalty_factor=contra_factor,
                score_type="heuristic_score",
                untested_checks=list(c.untested_checks),
            )

            evaluations.append(
                CandidateEvaluation(
                    candidate_id=c.id,
                    name=c.name,
                    level=c.level,
                    base_score=round(c.score, 2),
                    composite_score=score,
                    supporting_count=sup_count,
                    contradictory_count=contra_count,
                    status=status,
                    score_breakdown=breakdown,
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
                    f"Top-ranked hypothesis with heuristic score {ev.composite_score:.2f} "
                    f"({ev.level.upper()} granularity, {ev.supporting_count} independent corroboration(s), 0 contradictions)."
                )
            else:
                prev = evaluations[idx - 1]
                diff = prev.composite_score - ev.composite_score
                if ev.status == "excluded":
                    ev.ranking_justification = (
                        f"Ranked below #{prev.rank} ({prev.name[:30]}) due to active exclusion: "
                        f"{'; '.join(ev.contradictory_reasons)}"
                    )
                else:
                    ev.ranking_justification = (
                        f"Ranked below #{prev.rank} ({prev.name[:30]}) by heuristic margin -{diff:.2f} "
                        f"(granularity '{ev.level}' or fewer corroborating constraints)."
                    )

        return evaluations
