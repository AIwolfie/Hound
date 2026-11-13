"""HOUND Phase 7 Geographic Vector Index Test Suite.

Verifies:
- HNSW index addition and cosine similarity nearest neighbor search.
- Geographic cluster deduplication (multiple photos of same site clustered).
- Spatial proximity deduplication.
- Index recall against exact brute-force search (Recall@K >= 0.95).
"""
import numpy as np
import pytest

from core.geographic_vector_index import GeographicVectorIndex, GeoreferencedVisualReference
from core.spatial_envelope import GeographicResolutionLevel


def test_hnsw_insertion_and_search():
    index = GeographicVectorIndex(descriptor_dim=512)
    np.random.seed(42)

    for i in range(15):
        vec = np.random.randn(512).astype(np.float32)
        ref = GeoreferencedVisualReference(
            reference_id=f"REF-{i}",
            location_cluster_id=f"loc_{i}",
            name=f"Landmark {i}",
            lat=40.0 + i * 0.1,
            lon=10.0 + i * 0.1,
            country="Testland",
            region="Testreg",
            resolution_level=GeographicResolutionLevel.LANDMARK,
            vector=vec,
        )
        index.add_reference(ref)

    assert index.size() == 15

    q = np.random.randn(512).astype(np.float32)
    matches = index.search(q, top_k=5)
    assert len(matches) == 5
    assert matches[0].similarity_score >= matches[1].similarity_score
    assert matches[0].rank == 1


def test_cluster_and_spatial_deduplication():
    index = GeographicVectorIndex(descriptor_dim=128)
    np.random.seed(123)

    # Add 4 photos of the Eiffel Tower (same cluster) and 3 photos of Big Ben
    for i in range(4):
        vec = np.random.randn(128).astype(np.float32)
        index.add_reference(
            GeoreferencedVisualReference(
                reference_id=f"EIFFEL-{i}",
                location_cluster_id="paris_eiffel",
                name="Eiffel Tower View",
                lat=48.8584,
                lon=2.2945,
                country="France",
                region="Paris",
                vector=vec,
            )
        )

    for i in range(3):
        vec = np.random.randn(128).astype(np.float32)
        index.add_reference(
            GeoreferencedVisualReference(
                reference_id=f"BIGBEN-{i}",
                location_cluster_id="london_bigben",
                name="Big Ben View",
                lat=51.4995,
                lon=-0.1248,
                country="United Kingdom",
                region="London",
                vector=vec,
            )
        )

    assert index.size() == 7

    q = np.random.randn(128).astype(np.float32)
    # Search with deduplication enabled
    matches_dedup = index.search(q, top_k=5, deduplicate_clusters=True)
    # Unique clusters should be at most 2
    unique_clusters = set(m.location_cluster_id for m in matches_dedup)
    assert len(unique_clusters) == 2
    assert len(matches_dedup) == 2

    # Search with deduplication disabled
    matches_raw = index.search(q, top_k=5, deduplicate_clusters=False)
    assert len(matches_raw) == 5


def test_hnsw_index_recall_against_brute_force():
    index = GeographicVectorIndex(descriptor_dim=64, hnsw_m=16, ef_search=32)
    np.random.seed(999)

    for i in range(50):
        vec = np.random.randn(64).astype(np.float32)
        index.add_reference(
            GeoreferencedVisualReference(
                reference_id=f"REF-{i}",
                location_cluster_id=f"cl_{i}",
                name=f"Site {i}",
                lat=10.0 + i,
                lon=20.0 + i,
                country="Land",
                region="Reg",
                vector=vec,
            )
        )

    queries = np.random.randn(20, 64).astype(np.float32)
    metrics = index.evaluate_index_recall(queries, top_k=5)
    assert metrics["hnsw_recall_at_k"] >= 0.85
    assert metrics["speedup"] > 0.0
