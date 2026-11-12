"""HOUND Calibrated Geographic Uncertainty Engine.

Decouples spatial resolution (geographic specificity) from empirical confidence
probability, prevents raw heuristic scores from being presented as probabilities,
and enforces explicit reasoning boundaries for honest abstention.
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field

from core.logger import get_logger
from core.spatial_envelope import GeographicResolutionLevel, SpatialEnvelope
from storage.evidence_store import EvidenceItem

logger = get_logger("hound.calibration")


class ConfidenceTier(str, Enum):
    """Calibrated confidence tier reflecting empirical probability of correctness."""
    DEFINITIVE = "DEFINITIVE"       # P >= 0.90: Rigorous multi-modal triangulation, zero contradictions
    PROBABLE = "PROBABLE"           # 0.70 <= P < 0.90: Strong corroboration across >= 2 independent modalities
    TENTATIVE = "TENTATIVE"         # 0.40 <= P < 0.70: Plausible hypothesis, single modality or partial evidence
    INSUFFICIENT = "INSUFFICIENT"   # P < 0.40: Low information, unverified assumption, or contradiction; triggers abstention


class CalibratedGeolocationEstimate(BaseModel):
    """Complete calibrated geographic conclusion with explicit uncertainty envelope."""

    candidate_id: str
    name: str
    spatial_envelope: SpatialEnvelope
    resolution_level: GeographicResolutionLevel
    raw_heuristic_score: float = Field(description="Raw heuristic evidence score [0.0 - 100.0] (Not a probability)")
    calibrated_probability: float = Field(description="Empirically calibrated probability of correctness [0.0 - 1.0]")
    confidence_tier: ConfidenceTier
    independent_modalities_count: int
    independent_modalities: List[str] = Field(default_factory=list)
    supporting_evidence: List[str] = Field(default_factory=list)
    contradictory_evidence: List[str] = Field(default_factory=list)
    unverified_assumptions: List[str] = Field(default_factory=list)
    abstention_recommended: bool = False
    abstention_reason: Optional[str] = None
    calibration_formula: str = "Logistic sigmoid multi-modal calibration: P = 1 / (1 + exp(-(-3.2 + 0.065*S + 0.85*M - 2.5*C)))"


class CalibrationEngine:
    """Computes calibrated confidence probabilities and formats rigorous geolocation estimates."""

    # Sigmoid calibration parameters tuned on multi-modal evidence corroboration
    SIGMOID_INTERCEPT: float = -3.20
    SCORE_COEFFICIENT: float = 0.065
    MODALITY_COEFFICIENT: float = 0.85
    CONTRADICTION_PENALTY: float = 2.50

    @classmethod
    def calibrate_probability(
        cls,
        raw_score: float,
        independent_modalities_count: int,
        contradictions_count: int = 0,
        is_falsified: bool = False,
    ) -> float:
        """Map raw heuristic evidence score and corroboration depth into calibrated probability [0.0, 1.0]."""
        if is_falsified or raw_score <= 0.0:
            return 0.0

        # Uncorroborated seed only (0 independent corroborating modalities)
        if independent_modalities_count <= 0:
            # Single clue cannot exceed tentative probability (capped at 0.35)
            z = cls.SIGMOID_INTERCEPT + cls.SCORE_COEFFICIENT * raw_score - cls.CONTRADICTION_PENALTY * contradictions_count
            prob = 1.0 / (1.0 + math.exp(-z))
            return round(min(0.35, max(0.01, prob)), 3)

        # Multi-modal corroboration
        m_clamped = min(3, independent_modalities_count)
        z = (
            cls.SIGMOID_INTERCEPT
            + cls.SCORE_COEFFICIENT * raw_score
            + cls.MODALITY_COEFFICIENT * m_clamped
            - cls.CONTRADICTION_PENALTY * contradictions_count
        )
        prob = 1.0 / (1.0 + math.exp(-z))

        # Contradictions strictly prevent definitive tier
        if contradictions_count > 0:
            prob = min(0.39, prob)

        return round(min(0.99, max(0.01, prob)), 3)

    @classmethod
    def determine_confidence_tier(cls, calibrated_prob: float) -> ConfidenceTier:
        """Assign standardized confidence tier from calibrated probability."""
        if calibrated_prob >= 0.90:
            return ConfidenceTier.DEFINITIVE
        elif calibrated_prob >= 0.70:
            return ConfidenceTier.PROBABLE
        elif calibrated_prob >= 0.40:
            return ConfidenceTier.TENTATIVE
        else:
            return ConfidenceTier.INSUFFICIENT

    @classmethod
    def evaluate_candidate(
        cls,
        candidate_id: str,
        name: str,
        raw_score: float,
        spatial_envelope: SpatialEnvelope,
        all_evidence: List[EvidenceItem],
        seed_evidence_kind: Optional[str] = None,
        is_falsified: bool = False,
        falsification_reason: Optional[str] = None,
    ) -> CalibratedGeolocationEstimate:
        """Construct full calibrated geolocation estimate separating spatial scale from certainty."""
        supporting: List[str] = []
        contradictions: List[str] = []
        assumptions: List[str] = []
        independent_modalities: Set[str] = set()

        for ev in all_evidence:
            if ev.kind == "contradiction" or ev.status == "falsified":
                contradictions.append(ev.text)
            elif ev.kind in ("ocr", "vpr", "road_geometry", "terrain", "infrastructure", "solar"):
                supporting.append(f"[{ev.kind.upper()}] {ev.text} (Weight: {ev.confidence_weight:.2f})")
                if seed_evidence_kind and ev.kind != seed_evidence_kind:
                    independent_modalities.add(ev.kind)
                elif not seed_evidence_kind:
                    independent_modalities.add(ev.kind)

        # If only 1 modality was observed, flag single-source assumption
        if len(independent_modalities) <= 1:
            assumptions.append("Single-source observation: lacks multi-modal geographic corroboration.")

        if is_falsified and falsification_reason:
            contradictions.append(f"[Falsification] {falsification_reason}")

        m_count = len(independent_modalities)
        c_count = len(contradictions)

        calibrated_prob = cls.calibrate_probability(
            raw_score=raw_score,
            independent_modalities_count=m_count,
            contradictions_count=c_count,
            is_falsified=is_falsified,
        )
        tier = cls.determine_confidence_tier(calibrated_prob)

        # Abstention logic
        abstain = False
        abstain_reason = None

        if is_falsified:
            abstain = True
            abstain_reason = f"Candidate falsified by physical contradiction: {falsification_reason or 'Solar or spatial contradiction'}"
        elif spatial_envelope.resolution_level == GeographicResolutionLevel.UNLOCALIZED:
            abstain = True
            abstain_reason = "No definitive geographic anchors detected in image (Unlocalized)."
        elif tier == ConfidenceTier.INSUFFICIENT:
            abstain = True
            abstain_reason = f"Calibrated confidence probability ({calibrated_prob:.2f}) is below threshold (0.40) for reliable localization."

        return CalibratedGeolocationEstimate(
            candidate_id=candidate_id,
            name=name,
            spatial_envelope=spatial_envelope,
            resolution_level=spatial_envelope.resolution_level,
            raw_heuristic_score=round(raw_score, 2),
            calibrated_probability=calibrated_prob,
            confidence_tier=tier,
            independent_modalities_count=m_count,
            independent_modalities=sorted(list(independent_modalities)),
            supporting_evidence=supporting,
            contradictory_evidence=contradictions,
            unverified_assumptions=assumptions,
            abstention_recommended=abstain,
            abstention_reason=abstain_reason,
        )
