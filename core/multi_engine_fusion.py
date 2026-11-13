"""HOUND Multi-Engine Visual Place Recognition Fusion System.

Implements evidence-aware candidate fusion across diverse VPR backbones:
- Never simply averages raw cosine similarities across different models.
- Normalizes and ranks each retrieval engine independently.
- Implements Reciprocal Rank Fusion (RRF) with configurable engine weighting.
- Tracks independent model provenance and contributing evidence per candidate.
- Applies contradiction penalties and geographic consistency checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from core.geographic_search import haversine_distance
from core.geographic_vector_index import VisualIndexMatch
from core.logger import get_logger
from core.spatial_envelope import GeographicResolutionLevel

logger = get_logger("hound.vpr.fusion")


@dataclass
class FusedCandidate:
    """A geographically verified candidate synthesized from multi-engine visual retrieval."""
    candidate_id: str
    location_cluster_id: str
    name: str
    lat: float
    lon: float
    country: str
    region: str
    resolution_level: GeographicResolutionLevel
    uncertainty_radius_m: float
    fused_score: float  # [0.0, 1.0] calibrated consensus score
    rrf_score: float
    contributing_engines: List[str]
    engine_ranks: Dict[str, int]
    engine_similarities: Dict[str, float]
    source_attribution: str
    geographic_consistency_passed: bool = True
    contradiction_flags: List[str] = field(default_factory=list)
    consensus_distance_km: Optional[float] = None
    provenance_explanation: str = ""


class MultiEngineVPRFusion:
    """Orchestrates independent engine normalization, Reciprocal Rank Fusion, and geographic consistency validation."""

    DEFAULT_ENGINE_WEIGHTS = {
        "cosplace": 1.00,
        "anyloc": 0.90,
        "mixvpr": 0.85,
        "hybrid_color_spatial": 0.35,
        "resnet18_baseline": 0.40,
    }

    def __init__(
        self,
        rrf_k: int = 60,
        engine_weights: Optional[Dict[str, float]] = None,
        consensus_distance_km: float = 25.0,
    ):
        self.rrf_k = rrf_k
        self.engine_weights = engine_weights or self.DEFAULT_ENGINE_WEIGHTS
        self.consensus_distance_km = consensus_distance_km

    def fuse(
        self,
        engine_results: Dict[str, List[VisualIndexMatch]],
        top_k: int = 10,
        known_contradictions: Optional[List[Dict[str, Any]]] = None,
    ) -> List[FusedCandidate]:
        """Fuse retrieval results from multiple independent VPR engines using Reciprocal Rank Fusion (RRF).

        Args:
            engine_results: Dict mapping engine_name to its ranked list of VisualIndexMatch objects.
            top_k: Number of fused candidates to return.
            known_contradictions: Optional list of physical constraints to check (solar, driving side).
        """
        if not engine_results:
            return []

        # 1. Index candidates across all engines by location_cluster_id
        cluster_records: Dict[str, Dict[str, Any]] = {}

        for engine_name, matches in engine_results.items():
            weight = self.engine_weights.get(engine_name, 0.5)

            for rank_idx, m in enumerate(matches):
                rank = rank_idx + 1
                cid = m.location_cluster_id

                if cid not in cluster_records:
                    cluster_records[cid] = {
                        "reference_id": m.reference_id,
                        "location_cluster_id": cid,
                        "name": m.name,
                        "lat": m.lat,
                        "lon": m.lon,
                        "country": m.country,
                        "region": m.region,
                        "resolution_level": m.resolution_level,
                        "uncertainty_radius_m": m.uncertainty_radius_m,
                        "source_attribution": m.source_attribution,
                        "rrf_score": 0.0,
                        "contributing_engines": [],
                        "engine_ranks": {},
                        "engine_similarities": {},
                    }

                # Compute RRF increment: w / (k + rank)
                rrf_increment = weight / (self.rrf_k + rank)
                cluster_records[cid]["rrf_score"] += rrf_increment
                cluster_records[cid]["contributing_engines"].append(engine_name)
                cluster_records[cid]["engine_ranks"][engine_name] = rank
                cluster_records[cid]["engine_similarities"][engine_name] = m.similarity_score

        if not cluster_records:
            return []

        # 2. Check multi-engine consensus and geographic consistency
        candidates_list = list(cluster_records.values())

        # Find maximum possible RRF score to normalize fused_score into [0.0, 1.0]
        max_possible_rrf = sum(self.engine_weights.get(eng, 0.5) / (self.rrf_k + 1) for eng in engine_results.keys())
        if max_possible_rrf <= 0:
            max_possible_rrf = 1.0

        fused_candidates: List[FusedCandidate] = []

        for c in candidates_list:
            cid = c["location_cluster_id"]
            raw_rrf = c["rrf_score"]
            normalized_score = min(1.0, max(0.0, raw_rrf / max_possible_rrf))

            # Corroboration bonus: if >= 2 independent engines retrieved this cluster
            n_engines = len(c["contributing_engines"])
            if n_engines >= 2:
                # Multi-engine consensus bonus preserving relative margin
                bonus = 0.05 * (n_engines - 1)
                normalized_score = min(0.99, normalized_score * (1.0 + bonus))

            # Contradiction checks
            contradictions: List[str] = []
            if known_contradictions:
                for contra in known_contradictions:
                    kind = contra.get("kind")
                    # Solar hemisphere / latitude check
                    if kind == "solar_latitude":
                        max_lat = contra.get("max_lat")
                        min_lat = contra.get("min_lat")
                        if max_lat is not None and c["lat"] > max_lat:
                            contradictions.append(f"Solar contradiction: latitude {c['lat']:.1f}°N exceeds ceiling {max_lat:.1f}°N")
                        if min_lat is not None and c["lat"] < min_lat:
                            contradictions.append(f"Solar contradiction: latitude {c['lat']:.1f}°S below minimum {min_lat:.1f}°S")

            consistency_passed = len(contradictions) == 0
            if not consistency_passed:
                # Severe penalty for contradiction
                normalized_score = round(normalized_score * 0.15, 3)

            # Build provenance explanation
            eng_breakdown = ", ".join(
                f"{eng}: rank #{c['engine_ranks'][eng]} (sim {c['engine_similarities'][eng]:.2f})"
                for eng in c["contributing_engines"]
            )
            explanation = f"Fused from {n_engines} engine(s) [{eng_breakdown}]. RRF Score: {raw_rrf:.4f}."

            fused = FusedCandidate(
                candidate_id=f"FUSED-{cid}",
                location_cluster_id=cid,
                name=c["name"],
                lat=c["lat"],
                lon=c["lon"],
                country=c["country"],
                region=c["region"],
                resolution_level=c["resolution_level"],
                uncertainty_radius_m=c["uncertainty_radius_m"],
                fused_score=round(normalized_score, 4),
                rrf_score=round(raw_rrf, 5),
                contributing_engines=sorted(c["contributing_engines"]),
                engine_ranks=c["engine_ranks"],
                engine_similarities=c["engine_similarities"],
                source_attribution=c["source_attribution"],
                geographic_consistency_passed=consistency_passed,
                contradiction_flags=contradictions,
                provenance_explanation=explanation,
            )
            fused_candidates.append(fused)

        # Sort by fused score descending
        fused_candidates.sort(key=lambda x: x.fused_score, reverse=True)
        return fused_candidates[:top_k]
