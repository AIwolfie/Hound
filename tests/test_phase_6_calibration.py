"""Unit tests for HOUND Phase 6 Calibrated Geographic Uncertainty Engine."""
from __future__ import annotations

import pytest
from core.calibration_engine import CalibrationEngine, ConfidenceTier, CalibratedGeolocationEstimate
from core.spatial_envelope import GeographicResolutionLevel, SpatialEnvelope
from storage.evidence_store import EvidenceItem


def test_spatial_resolution_decoupled_from_confidence():
    """Verify that high spatial resolution does not imply high confidence probability."""
    # Scenario A: Landmark resolution (50m radius) with low confidence (single weak clue)
    env_landmark = SpatialEnvelope.point(
        lat=48.8584,
        lon=2.2945,
        radius_m=50.0,
        resolution=GeographicResolutionLevel.LANDMARK,
        reasoning="Weak unverified single observation",
    )
    ev_weak = EvidenceItem(
        id="ev_w",
        investigation_id="inv_test",
        kind="ocr",
        text="Single blurred sign snippet",
        confidence_weight=0.35,
        status="observed",
    )
    est_weak_landmark = CalibrationEngine.evaluate_candidate(
        candidate_id="cand_w",
        name="Speculative Eiffel Tower Pinpoint",
        raw_score=15.0,
        spatial_envelope=env_landmark,
        all_evidence=[ev_weak],
        seed_evidence_kind="ocr",
    )

    # Scenario B: Country resolution (1,000,000m radius) with high confidence (multiple independent clues)
    env_country = SpatialEnvelope.point(
        lat=46.2276,
        lon=2.2137,
        radius_m=1000000.0,
        resolution=GeographicResolutionLevel.COUNTRY,
        reasoning="Firm national borders from language, sun, and road network",
    )
    ev_ocr = EvidenceItem(id="ev_1", investigation_id="inv_test", kind="ocr", text="French Highway Sign", confidence_weight=0.9, status="verified")
    ev_vpr = EvidenceItem(id="ev_2", investigation_id="inv_test", kind="vpr", text="French Architectural Style", confidence_weight=0.9, status="verified")
    ev_solar = EvidenceItem(id="ev_3", investigation_id="inv_test", kind="solar", text="Mid-Latitude Solar Position", confidence_weight=0.85, status="computed")

    est_firm_country = CalibrationEngine.evaluate_candidate(
        candidate_id="cand_f",
        name="France National Boundary",
        raw_score=45.0,
        spatial_envelope=env_country,
        all_evidence=[ev_ocr, ev_vpr, ev_solar],
        seed_evidence_kind="ocr",
    )

    # Assertions
    # Landmark has higher resolution (LANDMARK vs COUNTRY)
    assert est_weak_landmark.resolution_level == GeographicResolutionLevel.LANDMARK
    assert est_firm_country.resolution_level == GeographicResolutionLevel.COUNTRY

    # But Landmark has LOW confidence (INSUFFICIENT, abstention recommended)
    assert est_weak_landmark.confidence_tier == ConfidenceTier.INSUFFICIENT
    assert est_weak_landmark.calibrated_probability < 0.40
    assert est_weak_landmark.abstention_recommended is True

    # While Country has HIGH confidence (PROBABLE or DEFINITIVE)
    assert est_firm_country.confidence_tier in (ConfidenceTier.PROBABLE, ConfidenceTier.DEFINITIVE)
    assert est_firm_country.calibrated_probability >= 0.70
    assert est_firm_country.abstention_recommended is False


def test_contradiction_penalizes_calibrated_probability():
    """Verify that unresolved contradictions prevent definitive or probable confidence."""
    env = SpatialEnvelope.point(
        lat=52.5200,
        lon=13.4050,
        radius_m=5000.0,
        resolution=GeographicResolutionLevel.CITY,
    )
    ev_sup = EvidenceItem(id="ev_s", investigation_id="inv_c", kind="ocr", text="Berlin Sign", confidence_weight=0.9, status="verified")
    ev_contra = EvidenceItem(id="ev_c", investigation_id="inv_c", kind="contradiction", text="Driving side is left-hand (UK/JP)", confidence_weight=0.95, status="falsified")

    est = CalibrationEngine.evaluate_candidate(
        candidate_id="cand_contra",
        name="Berlin with Driving Mismatch",
        raw_score=40.0,
        spatial_envelope=env,
        all_evidence=[ev_sup, ev_contra],
        seed_evidence_kind="ocr",
    )

    # Probability must drop to INSUFFICIENT tier due to physical contradiction
    assert est.calibrated_probability < 0.40
    assert est.confidence_tier == ConfidenceTier.INSUFFICIENT
    assert len(est.contradictory_evidence) > 0
    assert est.abstention_recommended is True


def test_falsification_forces_zero_probability():
    """Verify that a candidate flagged as falsified receives exactly 0.0 probability."""
    env = SpatialEnvelope.point(lat=70.0, lon=25.0, radius_m=1000.0, resolution=GeographicResolutionLevel.DISTRICT)
    est = CalibrationEngine.evaluate_candidate(
        candidate_id="cand_falsified",
        name="Impossible Solar Lat",
        raw_score=60.0,
        spatial_envelope=env,
        all_evidence=[],
        is_falsified=True,
        falsification_reason="Solar elevation 65 deg exceeds Arctic ceiling 43 deg",
    )

    assert est.calibrated_probability == 0.0
    assert est.confidence_tier == ConfidenceTier.INSUFFICIENT
    assert est.abstention_recommended is True
    assert "Solar elevation" in est.abstention_reason


def test_multi_modal_corroboration_gain():
    """Verify that additional independent modalities significantly increase calibrated probability."""
    raw_score = 35.0

    # 1 modality
    p1 = CalibrationEngine.calibrate_probability(raw_score=raw_score, independent_modalities_count=1)
    # 2 modalities
    p2 = CalibrationEngine.calibrate_probability(raw_score=raw_score, independent_modalities_count=2)
    # 3 modalities
    p3 = CalibrationEngine.calibrate_probability(raw_score=raw_score, independent_modalities_count=3)

    assert p1 < p2 < p3
    assert p1 < 0.50
    assert p3 >= 0.70
