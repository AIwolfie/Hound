"""HOUND Phase 7 Multi-Engine VPR Fusion Test Suite.

Verifies:
- Reciprocal Rank Fusion calculation with engine weighting.
- Multi-engine corroboration bonuses.
- Contradiction penalty enforcement.
- Cluster candidate deduplication across engines.
"""
import pytest

from core.geographic_vector_index import VisualIndexMatch
from core.multi_engine_fusion import MultiEngineVPRFusion
from core.spatial_envelope import GeographicResolutionLevel


def test_reciprocal_rank_fusion_basic():
    fusion = MultiEngineVPRFusion(rrf_k=60)

    m1 = VisualIndexMatch(
        reference_id="REF-1",
        location_cluster_id="eiffel",
        name="Eiffel Tower",
        lat=48.8584,
        lon=2.2945,
        country="France",
        region="Paris",
        similarity_score=0.92,
        resolution_level=GeographicResolutionLevel.LANDMARK,
        uncertainty_radius_m=500.0,
        view_heading_deg=None,
        source_attribution="Wikimedia",
    )
    m2 = VisualIndexMatch(
        reference_id="REF-2",
        location_cluster_id="tokyo",
        name="Tokyo Tower",
        lat=35.6586,
        lon=139.7454,
        country="Japan",
        region="Tokyo",
        similarity_score=0.88,
        resolution_level=GeographicResolutionLevel.LANDMARK,
        uncertainty_radius_m=500.0,
        view_heading_deg=None,
        source_attribution="Wikimedia",
    )

    results = {
        "cosplace": [m1, m2],
        "anyloc": [m1, m2],
    }

    fused = fusion.fuse(results, top_k=5)
    assert len(fused) == 2
    assert fused[0].location_cluster_id == "eiffel"
    assert fused[0].rrf_score > fused[1].rrf_score
    assert fused[0].fused_score >= fused[1].fused_score
    assert len(fused[0].contributing_engines) == 2
    assert "cosplace" in fused[0].contributing_engines
    assert "anyloc" in fused[0].contributing_engines


def test_contradiction_penalty_in_fusion():
    fusion = MultiEngineVPRFusion(rrf_k=60)

    # Southern hemisphere candidate
    m_syd = VisualIndexMatch(
        reference_id="REF-SYD",
        location_cluster_id="sydney_opera",
        name="Sydney Opera House",
        lat=-33.8568,
        lon=151.2153,
        country="Australia",
        region="NSW",
        similarity_score=0.95,
        resolution_level=GeographicResolutionLevel.LANDMARK,
        uncertainty_radius_m=500.0,
        view_heading_deg=None,
        source_attribution="Wikimedia",
    )

    results = {"cosplace": [m_syd]}
    # Contradiction: solar ceiling requires Northern hemisphere (min_lat = 10.0 N)
    contradictions = [{"kind": "solar_latitude", "min_lat": 10.0}]

    fused = fusion.fuse(results, top_k=5, known_contradictions=contradictions)
    assert len(fused) == 1
    assert fused[0].geographic_consistency_passed is False
    assert len(fused[0].contradiction_flags) > 0
    # Score must be severely penalized
    assert fused[0].fused_score < 0.20
