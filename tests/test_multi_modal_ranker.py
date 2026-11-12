"""Unit tests for HOUND Multi-Modal Candidate Ranker & Evidence Fusion."""
from __future__ import annotations

import pytest
from core.spatial_envelope import GeographicResolutionLevel, SpatialEnvelope
from agent.multi_modal_ranker import MultiModalRanker, RESOLUTION_LEVEL_WEIGHTS
from storage.evidence_store import EvidenceItem


def test_basic_multi_modal_evaluation():
    envelope = SpatialEnvelope.point(
        lat=48.8584,
        lon=2.2945,
        radius_m=350.0,
        resolution=GeographicResolutionLevel.LANDMARK,
        reasoning="Eiffel Tower, Paris",
    )

    ev_seed = EvidenceItem(
        id="ev_001",
        investigation_id="inv_test",
        kind="vpr",
        text="Eiffel Tower visual match",
        confidence_weight=0.9,
        status="verified",
    )
    ev_infra = EvidenceItem(
        id="ev_002",
        investigation_id="inv_test",
        kind="infrastructure",
        text="Lattice steel observation tower",
        confidence_weight=0.85,
        status="verified",
    )

    eval_result = MultiModalRanker.evaluate_hypothesis(
        candidate_id="cand_eiffel",
        name="Eiffel Tower, Paris",
        seed_evidence_id="ev_001",
        seed_evidence_kind="vpr",
        all_evidence=[ev_seed, ev_infra],
        spatial_envelope=envelope,
        vpr_similarity_score=0.92,
        road_curvature_score=None,
        terrain_correlation_score=None,
    )

    assert eval_result.candidate_id == "cand_eiffel"
    assert not eval_result.is_falsified
    assert eval_result.breakdown.level_specificity_multiplier == RESOLUTION_LEVEL_WEIGHTS[GeographicResolutionLevel.LANDMARK]
    assert eval_result.breakdown.composite_heuristic_score > 10.0
    assert eval_result.breakdown.score_type_label == "Heuristic evidence score [0.0 - 100.0] (Not a probability)"
    assert any("Eiffel Tower" in f for f in eval_result.observed_facts)


def test_solar_falsification_zeroes_score():
    envelope = SpatialEnvelope.point(
        lat=65.0,
        lon=25.0,
        radius_m=1000.0,
        resolution=GeographicResolutionLevel.NEIGHBORHOOD,
    )

    eval_result = MultiModalRanker.evaluate_hypothesis(
        candidate_id="cand_falsified_arctic",
        name="Northern Finland",
        seed_evidence_id=None,
        seed_evidence_kind="ocr",
        all_evidence=[],
        spatial_envelope=envelope,
        vpr_similarity_score=0.88,
        solar_falsified=True,
        solar_rejection_reason="Observed solar elevation 48 deg exceeds astronomical ceiling 22 deg.",
    )

    assert eval_result.is_falsified is True
    assert eval_result.breakdown.solar_multiplier == 0.0
    assert eval_result.breakdown.composite_heuristic_score == 0.0
    assert "astronomical ceiling" in eval_result.rejection_reason


def test_correlation_discount_for_redundant_modalities():
    envelope = SpatialEnvelope.point(
        lat=37.7749,
        lon=-122.4194,
        radius_m=2000.0,
        resolution=GeographicResolutionLevel.CITY,
    )

    # 3 separate OCR text snippets from the same sign
    ev1 = EvidenceItem(id="ev_1", investigation_id="inv_test", kind="ocr", text="Exit 42", confidence_weight=0.8, status="verified")
    ev2 = EvidenceItem(id="ev_2", investigation_id="inv_test", kind="ocr", text="Market St", confidence_weight=0.8, status="verified")
    ev3 = EvidenceItem(id="ev_3", investigation_id="inv_test", kind="ocr", text="Downtown", confidence_weight=0.8, status="verified")

    eval_result = MultiModalRanker.evaluate_hypothesis(
        candidate_id="cand_sf",
        name="San Francisco Exit 42",
        seed_evidence_id="ev_1",
        seed_evidence_kind="ocr",
        all_evidence=[ev1, ev2, ev3],
        spatial_envelope=envelope,
    )

    # Correlation discount factor should be strictly less than 1.0 due to redundant OCR
    assert eval_result.breakdown.correlation_discount_factor < 1.0
    assert eval_result.breakdown.correlation_discount_factor == pytest.approx(0.9 * 0.9, 0.01)


def test_resolution_specificity_weighting():
    # Evaluate identical evidence on a specific coordinate vs country-level
    env_coord = SpatialEnvelope.point(
        lat=35.6895,
        lon=139.6917,
        radius_m=50.0,
        resolution=GeographicResolutionLevel.SPECIFIC_COORDINATE,
    )
    env_country = SpatialEnvelope.point(
        lat=35.6895,
        lon=139.6917,
        radius_m=1000000.0,
        resolution=GeographicResolutionLevel.COUNTRY,
    )

    eval_coord = MultiModalRanker.evaluate_hypothesis(
        candidate_id="c_coord",
        name="Tokyo Tower Exact",
        seed_evidence_id=None,
        seed_evidence_kind=None,
        all_evidence=[],
        spatial_envelope=env_coord,
        vpr_similarity_score=0.80,
    )

    eval_country = MultiModalRanker.evaluate_hypothesis(
        candidate_id="c_country",
        name="Japan Country Envelope",
        seed_evidence_id=None,
        seed_evidence_kind=None,
        all_evidence=[],
        spatial_envelope=env_country,
        vpr_similarity_score=0.80,
    )

    # Specific coordinate has 5.0x multiplier vs 1.0x for country level
    assert eval_coord.breakdown.composite_heuristic_score > eval_country.breakdown.composite_heuristic_score
    assert eval_coord.breakdown.level_specificity_multiplier == 5.0
    assert eval_country.breakdown.level_specificity_multiplier == 1.0
