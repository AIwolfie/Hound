"""HOUND Multi-Modal Candidate Ranker & Independent Evidence Fusion.

Combines independent evidence modalities (Visual Place Recognition, OCR, Road Geometry,
Terrain Skylines, Solar Ephemeris, and OpenStreetMap Infrastructure) while strictly
preventing correlated evidence duplication, enforcing provenance, and never presenting
heuristic scores as probabilities.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field

from core.logger import get_logger
from core.spatial_envelope import GeographicResolutionLevel, SpatialEnvelope
from storage.evidence_store import EvidenceItem

logger = get_logger("hound.agent.ranker")


# Level-specific multiplier rewarding higher geographic specificity
RESOLUTION_LEVEL_WEIGHTS: Dict[GeographicResolutionLevel, float] = {
    GeographicResolutionLevel.SPECIFIC_COORDINATE: 5.0,
    GeographicResolutionLevel.LANDMARK: 4.5,
    GeographicResolutionLevel.NEIGHBORHOOD: 4.0,
    GeographicResolutionLevel.ROAD_CORRIDOR: 3.5,
    GeographicResolutionLevel.DISTRICT: 3.0,
    GeographicResolutionLevel.CITY: 2.5,
    GeographicResolutionLevel.REGION: 1.5,
    GeographicResolutionLevel.STATE_OR_PROVINCE: 1.2,
    GeographicResolutionLevel.COUNTRY: 1.0,
    GeographicResolutionLevel.UNLOCALIZED: 0.2,
}


class MultiModalEvidenceBreakdown(BaseModel):
    """Transparent mathematical score decomposition across independent modalities."""

    seed_modality: str = "unspecified"
    seed_score: float = 1.0
    vpr_bonus: float = 0.0                 # Visual Place Recognition
    ocr_bonus: float = 0.0                 # OCR signage / road numbers
    road_geometry_bonus: float = 0.0       # Road curvature / heading alignment
    terrain_skyline_bonus: float = 0.0     # Mountain skyline cross-correlation
    infrastructure_bonus: float = 0.0      # Railway catenary, utility masts
    solar_multiplier: float = 1.0          # 1.0 = consistent, 0.0 = falsified
    correlation_discount_factor: float = 1.0 # Penalizes clustered/correlated clues
    level_specificity_multiplier: float = 1.0
    composite_heuristic_score: float = 1.0
    score_type_label: str = "Heuristic evidence score [0.0 - 100.0] (Not a probability)"


class MultiModalCandidateEvaluation(BaseModel):
    """Complete multi-modal evaluation and spatial envelope for a geographic hypothesis."""

    candidate_id: str
    name: str
    spatial_envelope: SpatialEnvelope
    breakdown: MultiModalEvidenceBreakdown
    observed_facts: List[str] = Field(default_factory=list)
    spatial_inferences: List[str] = Field(default_factory=list)
    untested_constraints: List[str] = Field(default_factory=list)
    is_falsified: bool = False
    rejection_reason: Optional[str] = None


class MultiModalRanker:
    """Fuses multi-modal observations into ranked hypotheses with non-circular scoring."""

    @classmethod
    def evaluate_hypothesis(
        cls,
        candidate_id: str,
        name: str,
        seed_evidence_id: Optional[str],
        seed_evidence_kind: Optional[str],
        all_evidence: List[EvidenceItem],
        spatial_envelope: Optional[SpatialEnvelope] = None,
        road_curvature_score: Optional[float] = None,
        vpr_similarity_score: Optional[float] = None,
        terrain_correlation_score: Optional[float] = None,
        solar_falsified: bool = False,
        solar_rejection_reason: Optional[str] = None,
    ) -> MultiModalCandidateEvaluation:
        """Compute composite evidence score and compile facts vs inferences."""
        breakdown = MultiModalEvidenceBreakdown(
            seed_modality=seed_evidence_kind or "initial_intake",
            seed_score=2.0,
        )

        facts: List[str] = []
        inferences: List[str] = []
        untested: List[str] = []

        seen_modalities: Set[str] = set()
        if seed_evidence_kind:
            seen_modalities.add(seed_evidence_kind)

        # 1. Visual Place Recognition (VPR)
        if vpr_similarity_score is not None and vpr_similarity_score >= 0.50:
            if "vpr" not in seen_modalities:
                breakdown.vpr_bonus = vpr_similarity_score * 8.0
                seen_modalities.add("vpr")
                inferences.append(f"[VPR Inference] Visual descriptor matched reference database with similarity {vpr_similarity_score:.3f}.")
            else:
                breakdown.correlation_discount_factor *= 0.80

        # 2. Road Curvature & Scene Geometry
        if road_curvature_score is not None and road_curvature_score >= 0.60:
            if "road_geometry" not in seen_modalities:
                breakdown.road_geometry_bonus = road_curvature_score * 6.0
                seen_modalities.add("road_geometry")
                inferences.append(f"[Road Geometry Inference] Visual road curvature & heading aligned with corridor geometry (Score: {road_curvature_score:.2f}).")
            else:
                breakdown.correlation_discount_factor *= 0.85

        # 3. Mountain Ridge & Skyline Profiling
        if terrain_correlation_score is not None and terrain_correlation_score >= 0.65:
            if "terrain" not in seen_modalities:
                breakdown.terrain_skyline_bonus = terrain_correlation_score * 7.0
                seen_modalities.add("terrain")
                inferences.append(f"[Terrain Inference] Observed skyline matched DEM horizon profile (Correlation: {terrain_correlation_score:.3f}).")
            else:
                breakdown.correlation_discount_factor *= 0.85

        # 4. OCR Signage Text & Infrastructure Clues
        for ev in all_evidence:
            if ev.id == seed_evidence_id:
                facts.append(f"[Seed Fact] {ev.text}")
                continue

            if ev.kind == "ocr":
                facts.append(f"[Observed Text] {ev.text}")
                if "ocr" not in seen_modalities:
                    breakdown.ocr_bonus += 3.5 * ev.confidence_weight
                    seen_modalities.add("ocr")
                else:
                    breakdown.correlation_discount_factor *= 0.90
            elif ev.kind in ("infrastructure", "osm_infra"):
                facts.append(f"[Infrastructure Observation] {ev.text}")
                if "infrastructure" not in seen_modalities:
                    breakdown.infrastructure_bonus += 3.0 * ev.confidence_weight
                    seen_modalities.add("infrastructure")
            elif ev.kind == "falsification" or ev.status == "falsified":
                inferences.append(f"[Falsification Finding] {ev.text}")

        # 5. Solar Ephemeris Physical Feasibility
        if solar_falsified:
            breakdown.solar_multiplier = 0.0
            is_falsified = True
            rejection_reason = solar_rejection_reason or "Violates astronomical solar elevation ceiling."
        else:
            is_falsified = False
            rejection_reason = None

        # 6. Geographic Specificity Scale Multiplier
        envelope = spatial_envelope or SpatialEnvelope.unlocalized("No spatial envelope specified.")
        level_weight = RESOLUTION_LEVEL_WEIGHTS.get(envelope.resolution_level, 1.0)
        breakdown.level_specificity_multiplier = level_weight

        # 7. Composite Heuristic Score Calculation
        raw_sum = (
            breakdown.seed_score
            + breakdown.vpr_bonus
            + breakdown.ocr_bonus
            + breakdown.road_geometry_bonus
            + breakdown.terrain_skyline_bonus
            + breakdown.infrastructure_bonus
        )
        composite = (
            raw_sum
            * breakdown.correlation_discount_factor
            * breakdown.level_specificity_multiplier
            * breakdown.solar_multiplier
        )
        breakdown.composite_heuristic_score = round(max(0.0, min(100.0, composite)), 2)

        return MultiModalCandidateEvaluation(
            candidate_id=candidate_id,
            name=name,
            spatial_envelope=envelope,
            breakdown=breakdown,
            observed_facts=facts,
            spatial_inferences=inferences,
            untested_constraints=untested,
            is_falsified=is_falsified,
            rejection_reason=rejection_reason,
        )
