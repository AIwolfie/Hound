"""HOUND Adversarial Verification Engine.

Formulates challenging falsification questions against leading candidate
hypotheses across multiple independent modalities (Solar, OSM, DEM, Geometry).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel

from agent.agent_models import HypothesisRecord, VerificationQuestion
from agent.hypothesis_manager import HypothesisManager
from core.logger import get_logger
from investigators.shadows.solar_analyzer import is_solar_elevation_feasible
from storage.evidence_store import EvidenceItem, EvidenceStore

logger = get_logger("hound.agent.verification")


class VerificationResult(BaseModel):
    hypothesis_id: str
    modality: str
    outcome: str  # "confirmed", "contradicted", "untested", "data_unavailable"
    description: str
    is_falsified: bool = False
    evidence_item: Optional[EvidenceItem] = None


class VerificationEngine:
    """Orchestrates adversarial checks and falsification challenges."""

    def __init__(self, hypothesis_manager: HypothesisManager, evidence_store: EvidenceStore):
        self.hyp_mgr = hypothesis_manager
        self.evid_store = evidence_store

    def generate_verification_questions(
        self,
        hypothesis: HypothesisRecord,
        observed_clues: List[EvidenceItem],
    ) -> List[VerificationQuestion]:
        """Formulate targeted questions that could disprove the candidate."""
        questions: List[VerificationQuestion] = []

        # 1. Solar Ephemeris & Latitude Ceilings
        has_solar = any("solar elevation" in e.text.lower() or "shadow" in e.text.lower() for e in observed_clues)
        if has_solar and hypothesis.lat is not None:
            questions.append(
                VerificationQuestion(
                    hypothesis_id=hypothesis.id,
                    question_text=f"Does candidate latitude {hypothesis.lat:.2f}° satisfy astronomical solar elevation ceilings?",
                    modality="solar",
                    target_check="solar_elevation_ceiling",
                )
            )

        # 2. Highway / Road Corridor Geometry
        has_road = any("road sign" in e.text.lower() or "highway" in e.text.lower() for e in observed_clues)
        if has_road and hypothesis.lat is not None and hypothesis.lon is not None:
            questions.append(
                VerificationQuestion(
                    hypothesis_id=hypothesis.id,
                    question_text=f"Does candidate envelope '{hypothesis.name}' contain matching physical road infrastructure?",
                    modality="geometry",
                    target_check="osm_highway_corridor",
                )
            )

        # 3. Mountain Ridge & Terrain Skyline
        has_skyline = any("skyline" in e.text.lower() or "mountain" in e.text.lower() for e in observed_clues)
        if has_skyline and hypothesis.lat is not None and hypothesis.lon is not None:
            questions.append(
                VerificationQuestion(
                    hypothesis_id=hypothesis.id,
                    question_text=f"Is the visible mountain ridge profile compatible with DEM horizons at ({hypothesis.lat:.2f}, {hypothesis.lon:.2f})?",
                    modality="terrain",
                    target_check="dem_ridge_cross_correlation",
                )
            )

        # 4. Railway & Overhead Catenary Infrastructure
        has_rail = any("railway" in e.text.lower() or "catenary" in e.text.lower() for e in observed_clues)
        if has_rail and hypothesis.lat is not None and hypothesis.lon is not None:
            questions.append(
                VerificationQuestion(
                    hypothesis_id=hypothesis.id,
                    question_text=f"Are overhead catenary masts and rail tracks mapped near '{hypothesis.name}'?",
                    modality="infrastructure",
                    target_check="osm_railway_tracks",
                )
            )

        return questions

    def evaluate_solar_falsification(
        self,
        hypothesis: HypothesisRecord,
        observed_solar_elevation: float,
        capture_date=None,
    ) -> VerificationResult:
        """Adversarially test if candidate latitude violates astronomical bounds."""
        if hypothesis.lat is None:
            return VerificationResult(
                hypothesis_id=hypothesis.id,
                modality="solar",
                outcome="untested",
                description="Hypothesis has no definitive latitude.",
            )

        feasible, max_ceiling, _ = is_solar_elevation_feasible(
            lat_deg=hypothesis.lat,
            observed_elev_deg=observed_solar_elevation,
            day_of_year=capture_date,
        )

        if not feasible:
            # Active physical contradiction found!
            ev = self.evid_store.add_evidence(
                EvidenceItem(
                    investigation_id=hypothesis.investigation_id,
                    kind="falsification",
                    status="falsified",
                    text=(
                        f"[Adversarial Falsification] Observed solar elevation {observed_solar_elevation:.1f}° "
                        f"exceeds maximum theoretical ceiling {max_ceiling:.1f}° at latitude {hypothesis.lat:.2f}°. "
                        f"Candidate '{hypothesis.name}' is physically impossible."
                    ),
                    confidence_weight=1.0,
                )
            )
            self.hyp_mgr.record_contradiction(
                hypothesis=hypothesis,
                evidence_item=ev,
                rejection_reason=f"Violates solar ceiling ({observed_solar_elevation:.1f}° > {max_ceiling:.1f}° at lat {hypothesis.lat:.2f}°)",
                is_definitive_falsification=True,
            )
            return VerificationResult(
                hypothesis_id=hypothesis.id,
                modality="solar",
                outcome="contradicted",
                description=f"Physically impossible: solar elevation exceeds ceiling by {observed_solar_elevation - max_ceiling:.1f}°.",
                is_falsified=True,
                evidence_item=ev,
            )

        # Consistent
        return VerificationResult(
            hypothesis_id=hypothesis.id,
            modality="solar",
            outcome="confirmed",
            description=f"Solar elevation {observed_solar_elevation:.1f}° is within ceiling {max_ceiling:.1f}° at lat {hypothesis.lat:.2f}°.",
            is_falsified=False,
        )

    def evaluate_corridor_contradiction(
        self,
        hypothesis: HypothesisRecord,
        expected_corridor_name: str,
        elements_found: int,
        service_status: str,
    ) -> VerificationResult:
        """Evaluate whether missing infrastructure contradicts the candidate."""
        if service_status == "service_unavailable":
            # Outage is NOT negative evidence
            return VerificationResult(
                hypothesis_id=hypothesis.id,
                modality="geometry",
                outcome="data_unavailable",
                description="External geospatial service unavailable; absence of corridor cannot be asserted.",
            )

        if elements_found == 0:
            # Check if road was a mandatory requirement
            ev = self.evid_store.add_evidence(
                EvidenceItem(
                    investigation_id=hypothesis.investigation_id,
                    kind="contradiction",
                    status="challenged",
                    text=(
                        f"[Corridor Discrepancy] Mapped infrastructure check found 0 segments "
                        f"matching '{expected_corridor_name}' in candidate area."
                    ),
                    confidence_weight=0.75,
                )
            )
            self.hyp_mgr.record_contradiction(
                hypothesis=hypothesis,
                evidence_item=ev,
                rejection_reason=f"Expected corridor '{expected_corridor_name}' absent from mapped OpenStreetMap data.",
                is_definitive_falsification=False,
            )
            return VerificationResult(
                hypothesis_id=hypothesis.id,
                modality="geometry",
                outcome="contradicted",
                description="Expected corridor features absent from area.",
                is_falsified=False,
                evidence_item=ev,
            )

        return VerificationResult(
            hypothesis_id=hypothesis.id,
            modality="geometry",
            outcome="confirmed",
            description=f"Verified {elements_found} matching corridor elements.",
            is_falsified=False,
        )
