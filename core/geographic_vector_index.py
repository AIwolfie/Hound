"""HOUND Geographic Vector Index with HNSW Nearest-Neighbor Retrieval.

Implements high-performance, georeferenced visual retrieval:
- Fast HNSW (Hierarchical Navigable Small World) approximate nearest neighbor search via FAISS.
- Rich geographic metadata storage: coordinates, resolution level, uncertainty radius, viewpoint heading.
- Source attribution tracking for lawful OSINT and licensing compliance.
- Support for multiple viewpoints per physical site with location clustering.
- Geographic deduplication: prevents duplicate reference viewpoints from crowding top rankings.
- Empirical index recall evaluation against exact brute-force search.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from core.geographic_search import haversine_distance
from core.logger import get_logger
from core.spatial_envelope import GeographicResolutionLevel, SpatialEnvelope

logger = get_logger("hound.vpr.index")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not installed; GeographicVectorIndex will operate in NumPy brute-force mode.")


@dataclass
class GeoreferencedVisualReference:
    """A georeferenced reference image vector with complete geographic provenance."""
    reference_id: str
    location_cluster_id: str  # Groups multiple viewpoints/photos of the exact same physical site
    name: str
    lat: float
    lon: float
    country: str
    region: str
    resolution_level: GeographicResolutionLevel = GeographicResolutionLevel.LANDMARK
    uncertainty_radius_m: float = 500.0
    view_heading_deg: Optional[float] = None
    source_attribution: str = "Wikimedia Commons (CC BY-SA 4.0)"
    vector: np.ndarray = field(default_factory=lambda: np.zeros((512,), dtype=np.float32))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["resolution_level"] = self.resolution_level.value
        d["vector"] = self.vector.tolist()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GeoreferencedVisualReference:
        res_lvl = GeographicResolutionLevel(data.get("resolution_level", "landmark"))
        vec = np.array(data["vector"], dtype=np.float32)
        return cls(
            reference_id=data["reference_id"],
            location_cluster_id=data.get("location_cluster_id", data["reference_id"]),
            name=data["name"],
            lat=data["lat"],
            lon=data["lon"],
            country=data["country"],
            region=data["region"],
            resolution_level=res_lvl,
            uncertainty_radius_m=data.get("uncertainty_radius_m", 500.0),
            view_heading_deg=data.get("view_heading_deg"),
            source_attribution=data.get("source_attribution", "Unknown"),
            vector=vec,
        )


@dataclass
class VisualIndexMatch:
    """Ranked match resulting from vector similarity search against the index."""
    reference_id: str
    location_cluster_id: str
    name: str
    lat: float
    lon: float
    country: str
    region: str
    similarity_score: float  # Cosine similarity [0.0, 1.0]
    resolution_level: GeographicResolutionLevel
    uncertainty_radius_m: float
    view_heading_deg: Optional[float]
    source_attribution: str
    cluster_member_count: int = 1
    rank: int = 1


class GeographicVectorIndex:
    """HNSW-based modular approximate nearest neighbor index for georeferenced visual retrieval."""

    def __init__(
        self,
        descriptor_dim: int = 512,
        hnsw_m: int = 32,
        ef_construction: int = 64,
        ef_search: int = 64,
    ):
        self.descriptor_dim = descriptor_dim
        self.hnsw_m = hnsw_m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        self.items: List[GeoreferencedVisualReference] = []
        self._id_to_idx: Dict[str, int] = {}
        self._faiss_index: Optional[Any] = None
        self._vector_matrix: Optional[np.ndarray] = None

        self._init_faiss_index()

    def _init_faiss_index(self) -> None:
        if FAISS_AVAILABLE:
            # Use inner product on L2-normalized vectors (equivalent to cosine similarity)
            self._faiss_index = faiss.IndexHNSWFlat(self.descriptor_dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self._faiss_index.hnsw.efConstruction = self.ef_construction
            self._faiss_index.hnsw.efSearch = self.ef_search

    def size(self) -> int:
        return len(self.items)

    def add_reference(self, item: GeoreferencedVisualReference) -> None:
        """Add a georeferenced visual reference vector to the index."""
        if item.reference_id in self._id_to_idx:
            logger.debug(f"Reference '{item.reference_id}' already indexed; skipping.")
            return

        # Ensure L2 normalization
        vec = item.vector.astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        item.vector = vec

        idx = len(self.items)
        self.items.append(item)
        self._id_to_idx[item.reference_id] = idx

        if self._faiss_index is not None:
            self._faiss_index.add(np.expand_dims(vec, axis=0))

        # Update dense matrix for fallback / recall checks
        if self._vector_matrix is None:
            self._vector_matrix = np.expand_dims(vec, axis=0)
        else:
            self._vector_matrix = np.vstack([self._vector_matrix, np.expand_dims(vec, axis=0)])

    def add_batch(self, items: List[GeoreferencedVisualReference]) -> None:
        """Add multiple georeferenced references in a batch."""
        for it in items:
            self.add_reference(it)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        deduplicate_clusters: bool = True,
        spatial_dedup_radius_m: float = 1000.0,
    ) -> List[VisualIndexMatch]:
        """Search the geographic vector index with clustering and deduplication.

        Prevents duplicate reference photos from crowding out genuine distinct locations.
        """
        if not self.items:
            return []

        # Validate query
        q = query_vector.astype(np.float32).flatten()
        norm = float(np.linalg.norm(q))
        if norm > 0:
            q /= norm

        # Fetch candidate pool (retrieve 3x requested top_k to allow deduplication)
        pool_size = min(len(self.items), max(top_k * 4, 30))

        raw_indices: List[int] = []
        raw_scores: List[float] = []

        if self._faiss_index is not None and self._faiss_index.ntotal > 0:
            distances, indices = self._faiss_index.search(np.expand_dims(q, axis=0), pool_size)
            raw_indices = indices[0].tolist()
            raw_scores = distances[0].tolist()
        else:
            # Fallback to NumPy brute force
            sims = np.dot(self._vector_matrix, q)
            top_idxs = np.argsort(sims)[::-1][:pool_size]
            raw_indices = top_idxs.tolist()
            raw_scores = sims[top_idxs].tolist()

        # Deduplicate candidates by location_cluster_id and spatial proximity
        matches: List[VisualIndexMatch] = []
        seen_clusters: Set[str] = set()
        seen_coords: List[Tuple[float, float]] = []

        for idx, score in zip(raw_indices, raw_scores):
            if idx < 0 or idx >= len(self.items):
                continue
            item = self.items[idx]
            sim = float(max(0.0, min(1.0, score)))

            if deduplicate_clusters:
                # 1. Exact cluster ID deduplication
                if item.location_cluster_id in seen_clusters:
                    # Update cluster count for existing match
                    for m in matches:
                        if m.location_cluster_id == item.location_cluster_id:
                            m.cluster_member_count += 1
                            break
                    continue

                # 2. Spatial proximity deduplication
                too_close = False
                for c_lat, c_lon in seen_coords:
                    d_m = haversine_distance((item.lat, item.lon), (c_lat, c_lon))
                    if d_m <= spatial_dedup_radius_m:
                        too_close = True
                        break
                if too_close:
                    continue

            seen_clusters.add(item.location_cluster_id)
            seen_coords.append((item.lat, item.lon))

            match = VisualIndexMatch(
                reference_id=item.reference_id,
                location_cluster_id=item.location_cluster_id,
                name=item.name,
                lat=item.lat,
                lon=item.lon,
                country=item.country,
                region=item.region,
                similarity_score=round(sim, 4),
                resolution_level=item.resolution_level,
                uncertainty_radius_m=item.uncertainty_radius_m,
                view_heading_deg=item.view_heading_deg,
                source_attribution=item.source_attribution,
                cluster_member_count=1,
                rank=len(matches) + 1,
            )
            matches.append(match)

            if len(matches) >= top_k:
                break

        return matches

    def evaluate_index_recall(
        self,
        query_vectors: np.ndarray,
        top_k: int = 10,
    ) -> Dict[str, float]:
        """Empirically evaluate HNSW retrieval recall and speedup against exact brute-force search."""
        if not self.items or self._vector_matrix is None or len(query_vectors) == 0:
            return {"hnsw_recall_at_k": 0.0, "mean_hnsw_latency_ms": 0.0, "mean_brute_force_latency_ms": 0.0, "speedup": 1.0}

        queries = query_vectors.astype(np.float32)
        n_queries = len(queries)
        hits = 0

        hnsw_times: List[float] = []
        bf_times: List[float] = []

        for q in queries:
            q_norm = q / (np.linalg.norm(q) + 1e-7)

            # Brute-force exact top_k
            t0 = time.perf_counter()
            sims = np.dot(self._vector_matrix, q_norm)
            bf_top_k = set(np.argsort(sims)[::-1][:top_k])
            bf_times.append((time.perf_counter() - t0) * 1000)

            # HNSW top_k
            t0 = time.perf_counter()
            if self._faiss_index is not None:
                _, indices = self._faiss_index.search(np.expand_dims(q_norm, axis=0), top_k)
                hnsw_top_k = set(indices[0].tolist())
            else:
                hnsw_top_k = bf_top_k
            hnsw_times.append((time.perf_counter() - t0) * 1000)

            # Measure overlap (Recall@K)
            overlap = len(bf_top_k.intersection(hnsw_top_k))
            hits += overlap / float(top_k)

        mean_recall = hits / float(n_queries)
        mean_hnsw_ms = sum(hnsw_times) / len(hnsw_times)
        mean_bf_ms = sum(bf_times) / len(bf_times)
        speedup = mean_bf_ms / max(0.001, mean_hnsw_ms)

        return {
            "hnsw_recall_at_k": round(mean_recall, 4),
            "mean_hnsw_latency_ms": round(mean_hnsw_ms, 3),
            "mean_brute_force_latency_ms": round(mean_bf_ms, 3),
            "speedup": round(speedup, 2),
        }
