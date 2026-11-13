"""HOUND Phase 7.1 Multi-Engine VPR Benchmark Runner.

Executes rigorous independent evaluation across three strictly separated tracks:
- Track A: Closed-Set Retrieval (10 independent images of indexed locations).
- Track B: Unseen-Viewpoint Retrieval (10 extreme viewpoint / night / seasonal shifts).
- Track C: Geographic Open-Set (7 unindexed world landmarks + 8 negative controls).

Evaluates:
- CosPlace (ResNet-18, 512-d)
- AnyLoc (DINOv2 ViT-S/14, 384-d)
- MixVPR (ResNet-50, 512-d)
- ImageNet ResNet-18 (512-d)
- Legacy Descriptors (Hybrid Color-Spatial, 1224-d)
- Multi-Engine Fusion (RRF with consensus bonuses)

Reports:
- Recall@1, Recall@5, Recall@10.
- Median and mean geographic error (km).
- Accuracy within 1 km, 5 km, 25 km, 100 km.
- Open-set False Positive Rate (FPR).
- Prediction coverage and selective accuracy.
- Raw prediction counts and confusion matrices.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_DIR))

from core.empirical_calibration import CalibrationMethod, EmpiricalCalibrator
from core.geographic_search import haversine_distance
from core.geographic_vector_index import GeographicVectorIndex, GeoreferencedVisualReference
from core.logger import get_logger
from core.multi_engine_fusion import MultiEngineVPRFusion
from core.spatial_envelope import GeographicResolutionLevel
from core.vpr_anyloc import AnyLocModel
from core.vpr_base import BaseVPRModel
from core.vpr_cosplace import CosPlaceModel
from core.vpr_legacy import BaselineNeuralModel, HybridColorSpatialModel
from core.vpr_mixvpr import MixVPRModel

logger = get_logger("hound.benchmark.phase7_1")

BASE_DIR = WORKSPACE_DIR / "data" / "benchmarks" / "phase_7_1"
REFS_DIR = BASE_DIR / "references"
META_DIR = BASE_DIR / "metadata"
GT_DIR = BASE_DIR / "ground_truth"
REPORTS_DIR = BASE_DIR / "reports"


@dataclass
class TrackMetrics:
    track_name: str
    model_name: str
    total_queries: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    acc_1km: float
    acc_5km: float
    acc_25km: float
    acc_100km: float
    median_error_km: Optional[float]
    mean_error_km: Optional[float]
    prediction_coverage_pct: float
    selective_accuracy_pct: float
    open_set_fpr_pct: float
    mean_latency_ms: float
    tp: int
    fp: int
    tn: int
    fn: int


class Phase71BenchmarkRunner:
    """Rigorous evaluation runner for Phase 7.1 independent benchmark validation."""

    def __init__(self):
        self.refs_manifest: List[Dict[str, Any]] = []
        self.queries_manifest: List[Dict[str, Any]] = []
        self.ground_truth: Dict[str, Dict[str, Any]] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        refs_file = META_DIR / "references_manifest.json"
        manifest_file = META_DIR / "manifest.json"
        gt_file = GT_DIR / "ground_truth.json"

        if not refs_file.exists() or not manifest_file.exists() or not gt_file.exists():
            raise FileNotFoundError("Phase 7.1 metadata files missing. Run curate_phase7_1_dataset.py first.")

        with open(refs_file, "r", encoding="utf-8") as f:
            self.refs_manifest = json.load(f)
        with open(manifest_file, "r", encoding="utf-8") as f:
            self.queries_manifest = json.load(f)
        with open(gt_file, "r", encoding="utf-8") as f:
            self.ground_truth = json.load(f)

    def build_reference_index(self, model: BaseVPRModel) -> GeographicVectorIndex:
        """Construct the reference database from the 10 real reference prototypes."""
        index = GeographicVectorIndex(descriptor_dim=model.descriptor_dim)
        for ref_spec in self.refs_manifest:
            img_path = BASE_DIR / "references" / ref_spec["filename"]
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                raise FileNotFoundError(f"Reference image not found: {img_path}")

            vec = model.extract_descriptor(img_bgr)
            ref_entry = GeoreferencedVisualReference(
                reference_id=ref_spec["ref_id"],
                location_cluster_id=f"cluster_{ref_spec['ref_id']}",
                name=ref_spec["name"],
                lat=ref_spec["target_lat"],
                lon=ref_spec["target_lon"],
                country=ref_spec["country"],
                region=ref_spec["region"],
                resolution_level=GeographicResolutionLevel.LANDMARK,
                uncertainty_radius_m=ref_spec.get("tolerance_km", 25.0) * 1000.0,
                source_attribution="Wikimedia Commons (Creative Commons)",
                vector=vec,
            )
            index.add_reference(ref_entry)
        return index

    def evaluate_track(
        self,
        track_name: str,
        model: BaseVPRModel,
        index: GeographicVectorIndex,
        confidence_threshold: float = 0.55,
    ) -> Tuple[TrackMetrics, List[Dict[str, Any]]]:
        """Evaluate a model on a single isolated track."""
        track_queries = [q for q in self.queries_manifest if q["track"] == track_name]
        is_open_set_track = (track_name == "TRACK_C")

        rec1_hits = 0
        rec5_hits = 0
        rec10_hits = 0
        acc_1km = 0
        acc_5km = 0
        acc_25km = 0
        acc_100km = 0
        errors_km: List[float] = []
        latencies_ms: List[float] = []

        emitted_count = 0
        emitted_correct = 0

        # Confusion matrix counters
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        query_traces: List[Dict[str, Any]] = []

        for q in track_queries:
            qid = q["query_id"]
            img_path = BASE_DIR / q["filename"]
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            t0 = time.perf_counter()
            query_vec = model.extract_descriptor(img_bgr)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            matches = index.search(query_vec, top_k=10, deduplicate_clusters=False)

            gt = self.ground_truth.get(qid, {})
            gt_lat = gt.get("target_lat")
            gt_lon = gt.get("target_lon")
            tol_km = gt.get("tolerance_km", 25.0)
            is_neg = gt.get("is_negative_control", False)

            top_match = matches[0] if matches else None
            top_score = top_match.similarity_score if top_match else 0.0

            # Decision: emit localization or abstain
            # For CosPlace, threshold 0.55 separates places from non-places
            abstain = (top_match is None) or (top_score < confidence_threshold)

            dist_km: Optional[float] = None
            is_correct_hit = False

            if not is_open_set_track:
                # Track A or Track B (Closed-set)
                target_ref_id = gt.get("target_ref_id")
                if top_match:
                    if top_match.reference_id == target_ref_id:
                        is_correct_hit = True

                    if gt_lat is not None and gt_lon is not None:
                        dist_km = haversine_distance((top_match.lat, top_match.lon), (gt_lat, gt_lon)) / 1000.0
                        errors_km.append(dist_km)
                        if dist_km <= 1.0: acc_1km += 1
                        if dist_km <= 5.0: acc_5km += 1
                        if dist_km <= 25.0: acc_25km += 1
                        if dist_km <= 100.0: acc_100km += 1

                # Check recall@k
                for r_idx, m in enumerate(matches[:10]):
                    if m.reference_id == target_ref_id:
                        if r_idx == 0: rec1_hits += 1
                        if r_idx < 5: rec5_hits += 1
                        if r_idx < 10: rec10_hits += 1
                        break

                if not abstain:
                    emitted_count += 1
                    if is_correct_hit:
                        emitted_correct += 1
                        tp += 1
                    else:
                        fp += 1
                else:
                    fn += 1

            else:
                # Track C (Open-set: true location is NOT in reference index)
                # Correct behavior is abstention (TN). Any emitted localization is false positive (FP).
                if abstain:
                    tn += 1
                else:
                    fp += 1
                    emitted_count += 1

            query_traces.append({
                "query_id": qid,
                "name": q["name"],
                "track": track_name,
                "top_match_id": top_match.reference_id if top_match else None,
                "top_match_name": top_match.name if top_match else None,
                "top_score": round(top_score, 4),
                "abstain": abstain,
                "correct_hit": is_correct_hit,
                "dist_km": round(dist_km, 2) if dist_km is not None else None,
            })

        n_q = len(track_queries)
        sorted_errs = sorted(errors_km) if errors_km else []
        med_err = round(sorted_errs[len(sorted_errs) // 2], 2) if sorted_errs else None
        mean_err = round(sum(sorted_errs) / len(sorted_errs), 2) if sorted_errs else None

        coverage_pct = round(emitted_count / max(1, n_q) * 100.0, 1)
        sel_acc = round(emitted_correct / max(1, emitted_count) * 100.0, 1) if emitted_count else 0.0

        if is_open_set_track:
            open_set_fpr = round(fp / max(1, n_q) * 100.0, 1)
        else:
            open_set_fpr = 0.0

        metrics = TrackMetrics(
            track_name=track_name,
            model_name=model.model_name,
            total_queries=n_q,
            recall_at_1=round(rec1_hits / max(1, n_q) * 100.0, 1),
            recall_at_5=round(rec5_hits / max(1, n_q) * 100.0, 1),
            recall_at_10=round(rec10_hits / max(1, n_q) * 100.0, 1),
            acc_1km=round(acc_1km / max(1, n_q) * 100.0, 1),
            acc_5km=round(acc_5km / max(1, n_q) * 100.0, 1),
            acc_25km=round(acc_25km / max(1, n_q) * 100.0, 1),
            acc_100km=round(acc_100km / max(1, n_q) * 100.0, 1),
            median_error_km=med_err,
            mean_error_km=mean_err,
            prediction_coverage_pct=coverage_pct,
            selective_accuracy_pct=sel_acc,
            open_set_fpr_pct=open_set_fpr,
            mean_latency_ms=round(sum(latencies_ms) / max(1, len(latencies_ms)), 2),
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
        )

        return metrics, query_traces

    def evaluate_fusion_track(
        self,
        track_name: str,
        cos_m: BaseVPRModel,
        cos_idx: GeographicVectorIndex,
        any_m: BaseVPRModel,
        any_idx: GeographicVectorIndex,
        mix_m: BaseVPRModel,
        mix_idx: GeographicVectorIndex,
        fusion_engine: MultiEngineVPRFusion,
        confidence_threshold: float = 0.55,
    ) -> Tuple[TrackMetrics, List[Dict[str, Any]]]:
        """Evaluate multi-engine fused retrieval on a single isolated track."""
        track_queries = [q for q in self.queries_manifest if q["track"] == track_name]
        is_open_set_track = (track_name == "TRACK_C")

        rec1_hits = 0; rec5_hits = 0; rec10_hits = 0
        acc_1km = 0; acc_5km = 0; acc_25km = 0; acc_100km = 0
        errors_km: List[float] = []
        latencies_ms: List[float] = []
        emitted_count = 0; emitted_correct = 0
        tp = 0; fp = 0; tn = 0; fn = 0
        query_traces: List[Dict[str, Any]] = []

        for q in track_queries:
            qid = q["query_id"]
            img_path = BASE_DIR / q["filename"]
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            t0 = time.perf_counter()
            q_cos = cos_m.extract_descriptor(img_bgr)
            q_any = any_m.extract_descriptor(img_bgr)
            q_mix = mix_m.extract_descriptor(img_bgr)

            m_cos = cos_idx.search(q_cos, top_k=10)
            m_any = any_idx.search(q_any, top_k=10)
            m_mix = mix_idx.search(q_mix, top_k=10)

            fused_matches = fusion_engine.fuse({
                "cosplace": m_cos,
                "anyloc": m_any,
                "mixvpr": m_mix,
            }, top_k=10)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            gt = self.ground_truth.get(qid, {})
            gt_lat = gt.get("target_lat")
            gt_lon = gt.get("target_lon")

            top_fused = fused_matches[0] if fused_matches else None
            top_cos_sim = m_cos[0].similarity_score if m_cos else 0.0
            top_score = top_fused.fused_score if top_fused else 0.0

            # Gated abstention: CosPlace confidence threshold
            abstain = (top_fused is None) or (top_cos_sim < confidence_threshold)

            dist_km: Optional[float] = None
            is_correct_hit = False

            if not is_open_set_track:
                target_ref_id = gt.get("target_ref_id")
                if top_fused:
                    if target_ref_id and (target_ref_id in top_fused.location_cluster_id or target_ref_id in top_fused.name):
                        is_correct_hit = True
                    if gt_lat is not None and gt_lon is not None:
                        dist_km = haversine_distance((top_fused.lat, top_fused.lon), (gt_lat, gt_lon)) / 1000.0
                        errors_km.append(dist_km)
                        if dist_km <= 1.0: acc_1km += 1
                        if dist_km <= 5.0: acc_5km += 1
                        if dist_km <= 25.0: acc_25km += 1
                        if dist_km <= 100.0: acc_100km += 1

                for r_idx, fm in enumerate(fused_matches[:10]):
                    if target_ref_id and (target_ref_id in fm.location_cluster_id or target_ref_id in fm.name):
                        if r_idx == 0: rec1_hits += 1
                        if r_idx < 5: rec5_hits += 1
                        if r_idx < 10: rec10_hits += 1
                        break

                if not abstain:
                    emitted_count += 1
                    if is_correct_hit:
                        emitted_correct += 1
                        tp += 1
                    else:
                        fp += 1
                else:
                    fn += 1
            else:
                if abstain:
                    tn += 1
                else:
                    fp += 1
                    emitted_count += 1

            query_traces.append({
                "query_id": qid,
                "name": q["name"],
                "track": track_name,
                "top_match_id": top_fused.candidate_id if top_fused else None,
                "top_match_name": top_fused.name if top_fused else None,
                "fused_score": round(top_score, 4),
                "cosplace_sim": round(top_cos_sim, 4),
                "abstain": abstain,
                "correct_hit": is_correct_hit,
                "dist_km": round(dist_km, 2) if dist_km is not None else None,
            })

        n_q = len(track_queries)
        sorted_errs = sorted(errors_km) if errors_km else []
        med_err = round(sorted_errs[len(sorted_errs) // 2], 2) if sorted_errs else None
        mean_err = round(sum(sorted_errs) / len(sorted_errs), 2) if sorted_errs else None

        coverage_pct = round(emitted_count / max(1, n_q) * 100.0, 1)
        sel_acc = round(emitted_correct / max(1, emitted_count) * 100.0, 1) if emitted_count else 0.0
        open_set_fpr = round(fp / max(1, n_q) * 100.0, 1) if is_open_set_track else 0.0

        metrics = TrackMetrics(
            track_name=track_name,
            model_name="MultiEngine_Fusion_RRF",
            total_queries=n_q,
            recall_at_1=round(rec1_hits / max(1, n_q) * 100.0, 1),
            recall_at_5=round(rec5_hits / max(1, n_q) * 100.0, 1),
            recall_at_10=round(rec10_hits / max(1, n_q) * 100.0, 1),
            acc_1km=round(acc_1km / max(1, n_q) * 100.0, 1),
            acc_5km=round(acc_5km / max(1, n_q) * 100.0, 1),
            acc_25km=round(acc_25km / max(1, n_q) * 100.0, 1),
            acc_100km=round(acc_100km / max(1, n_q) * 100.0, 1),
            median_error_km=med_err,
            mean_error_km=mean_err,
            prediction_coverage_pct=coverage_pct,
            selective_accuracy_pct=sel_acc,
            open_set_fpr_pct=open_set_fpr,
            mean_latency_ms=round(sum(latencies_ms) / max(1, len(latencies_ms)), 2),
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
        )
        return metrics, query_traces

    def run_all(self) -> Dict[str, Any]:
        """Execute complete evaluation across Track A, B, and C for all models."""
        logger.info("Initializing Phase 7.1 Multi-Engine Independent Benchmark Suite...")

        # 1. Models to evaluate
        cosplace = CosPlaceModel(backbone="ResNet18", fc_output_dim=512, use_cuda=True)
        anyloc = AnyLocModel(backbone="dinov2_vits14", desc_dim=384, use_cuda=True)
        mixvpr = MixVPRModel(backbone="resnet50", fc_output_dim=512, use_cuda=True)
        baseline = BaselineNeuralModel(use_cuda=True)
        legacy = HybridColorSpatialModel()

        eval_models: List[BaseVPRModel] = [cosplace, anyloc, mixvpr, baseline, legacy]
        tracks = ["TRACK_A", "TRACK_B", "TRACK_C"]

        results_by_track: Dict[str, Dict[str, Any]] = {t: {} for t in tracks}
        all_traces: Dict[str, Any] = {}
        model_indices: Dict[str, Tuple[BaseVPRModel, GeographicVectorIndex]] = {}

        # Default confidence thresholds per model family
        thresholds = {
            cosplace.model_name: 0.55,
            anyloc.model_name: 0.65,
            mixvpr.model_name: 0.60,
            baseline.model_name: 0.60,
            legacy.model_name: 0.50,
        }

        for m in eval_models:
            logger.info(f"Loading and building index for: {m.model_name}...")
            m.load()
            idx = self.build_reference_index(m)
            model_indices[m.model_name] = (m, idx)
            thresh = thresholds.get(m.model_name, 0.55)

            for t in tracks:
                metrics, traces = self.evaluate_track(t, m, idx, confidence_threshold=thresh)
                results_by_track[t][m.model_name] = asdict(metrics)
                all_traces[f"{m.model_name}_{t}"] = traces

        # 2. Multi-Engine Fusion
        logger.info("Evaluating Multi-Engine Fusion across Tracks A, B, C...")
        fusion_engine = MultiEngineVPRFusion()
        cos_m, cos_idx = model_indices[cosplace.model_name]
        any_m, any_idx = model_indices[anyloc.model_name]
        mix_m, mix_idx = model_indices[mixvpr.model_name]

        for t in tracks:
            f_metrics, f_traces = self.evaluate_fusion_track(
                t, cos_m, cos_idx, any_m, any_idx, mix_m, mix_idx, fusion_engine, confidence_threshold=0.55
            )
            results_by_track[t]["MultiEngine_Fusion_RRF"] = asdict(f_metrics)
            all_traces[f"MultiEngine_Fusion_{t}"] = f_traces

        # Unload models to free GPU
        for m in eval_models:
            m.unload()

        report_payload = {
            "metadata": {
                "evaluation_date": "2026-11-13",
                "phase": "Phase 7.1 Independent Benchmark Validation",
                "total_references": len(self.refs_manifest),
                "track_a_queries": sum(1 for q in self.queries_manifest if q["track"] == "TRACK_A"),
                "track_b_queries": sum(1 for q in self.queries_manifest if q["track"] == "TRACK_B"),
                "track_c_queries": sum(1 for q in self.queries_manifest if q["track"] == "TRACK_C"),
            },
            "results_by_track": results_by_track,
            "per_case_traces": all_traces,
        }

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out_file = REPORTS_DIR / "phase_7_1_benchmark_report.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=2)

        logger.info(f"Phase 7.1 Benchmark Report saved to: {out_file}")
        return report_payload


if __name__ == "__main__":
    runner = Phase71BenchmarkRunner()
    report = runner.run_all()

    print("\n" + "=" * 105)
    print("HOUND PHASE 7.1 — INDEPENDENT VALIDATION BENCHMARK RESULTS")
    print("=" * 105)

    for track in ["TRACK_A", "TRACK_B", "TRACK_C"]:
        print(f"\n--- {track} ---")
        print(f"{'Model Name':<26} | {'R@1':<6} | {'R@5':<6} | {'<=25km':<7} | {'Med Err':<10} | {'Cover':<6} | {'SelAcc':<6} | {'FPR':<6} | {'TP':<3} | {'FP':<3} | {'TN':<3} | {'FN':<3}")
        print("-" * 105)
        for model_name, m in report["results_by_track"][track].items():
            med_str = f"{m['median_error_km']}km" if m['median_error_km'] is not None else "N/A"
            print(f"{model_name:<26} | {m['recall_at_1']:<5.1f}% | {m['recall_at_5']:<5.1f}% | {m['acc_25km']:<6.1f}% | {med_str:<10} | {m['prediction_coverage_pct']:<5.1f}% | {m['selective_accuracy_pct']:<5.1f}% | {m['open_set_fpr_pct']:<5.1f}% | {m['tp']:<3} | {m['fp']:<3} | {m['tn']:<3} | {m['fn']:<3}")

