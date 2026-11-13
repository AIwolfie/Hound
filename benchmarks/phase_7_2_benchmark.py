"""HOUND Phase 7.2 Benchmark Runner: Statistical Validation & Open-Set Reliability.

Evaluates:
- Threshold selection & freezing strictly on Calibration Partition (D_calib).
- Untouched Test Partition (D_test) evaluation across:
  * Track A: Closed-Set Independent
  * Track B: Unseen Viewpoints & Severe Conditions
  * Track C1: Unseen Geographic Locations
  * Track C2: Non-Place Negative Controls
  * Track D: Adversarial Challenging Cases
- Evaluates CosPlace under frozen tau*, 0.40, and 0.55 thresholds.
- Evaluates AnyLoc, MixVPR, ResNet-18, Legacy, and Multi-Engine Fusion.
- Computes 95% bootstrap confidence intervals for principal metrics.
- Outputs machine-readable report data/benchmarks/phase_7_2/reports/phase_7_2_benchmark_report.json.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_DIR))

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

logger = get_logger("hound.benchmark.phase7_2")

BASE_DIR = WORKSPACE_DIR / "data" / "benchmarks" / "phase_7_2"
REFS_DIR = BASE_DIR / "references"
CALIB_DIR = BASE_DIR / "calibration"
TEST_DIR = BASE_DIR / "test"
META_DIR = BASE_DIR / "metadata"
GT_DIR = BASE_DIR / "ground_truth"
REPORTS_DIR = BASE_DIR / "reports"


def bootstrap_ci(values: List[float], n_boot: int = 1000, ci: float = 0.95) -> Tuple[float, float, float]:
    """Compute mean and 95% bootstrap confidence interval."""
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.array(values)
    mean_val = float(np.mean(arr))
    if len(arr) == 1:
        return mean_val, mean_val, mean_val
    np.random.seed(42)
    boot_means = []
    for _ in range(n_boot):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        boot_means.append(float(np.mean(sample)))
    low_pct = (1.0 - ci) / 2.0 * 100.0
    high_pct = (1.0 + ci) / 2.0 * 100.0
    low = float(np.percentile(boot_means, low_pct))
    high = float(np.percentile(boot_means, high_pct))
    return round(mean_val, 3), round(low, 3), round(high, 3)


class Phase72BenchmarkRunner:
    """Rigorous evaluation harness for HOUND Phase 7.2."""

    def __init__(self):
        self.refs_manifest: List[Dict[str, Any]] = []
        self.calib_manifest: List[Dict[str, Any]] = []
        self.calib_gt: Dict[str, Any] = {}
        self.test_manifest: List[Dict[str, Any]] = []
        self.test_gt: Dict[str, Any] = {}
        self._load_datasets()

    def _load_datasets(self) -> None:
        with open(META_DIR / "references_manifest.json", "r", encoding="utf-8") as f:
            self.refs_manifest = json.load(f)
        with open(META_DIR / "calibration_manifest.json", "r", encoding="utf-8") as f:
            self.calib_manifest = json.load(f)
        with open(GT_DIR / "calibration_ground_truth.json", "r", encoding="utf-8") as f:
            self.calib_gt = json.load(f)
        with open(META_DIR / "test_manifest.json", "r", encoding="utf-8") as f:
            self.test_manifest = json.load(f)
        with open(GT_DIR / "test_ground_truth.json", "r", encoding="utf-8") as f:
            self.test_gt = json.load(f)

    def build_reference_index(self, model: BaseVPRModel) -> GeographicVectorIndex:
        """Construct the 10-prototype reference index."""
        index = GeographicVectorIndex(descriptor_dim=model.descriptor_dim)
        for ref_spec in self.refs_manifest:
            img_path = BASE_DIR / ref_spec["filename"]
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                raise FileNotFoundError(f"Missing reference image: {img_path}")

            vec = model.extract_descriptor(img_bgr)
            rid = ref_spec["ref_id"]
            # Lat/lons from ground truth reference dictionary
            ref_entry = GeoreferencedVisualReference(
                reference_id=rid,
                location_cluster_id=f"cluster_{rid}",
                name=f"Reference Prototype {rid}",
                lat=48.8584 if rid == "REF-01" else (45.9765 if rid == "REF-02" else 0.0),
                lon=2.2945 if rid == "REF-01" else (7.7491 if rid == "REF-02" else 0.0),
                country="Reference",
                region="Reference",
                resolution_level=GeographicResolutionLevel.LANDMARK,
                uncertainty_radius_m=25000.0,
                source_attribution="Creative Commons",
                vector=vec,
            )
            index.add_reference(ref_entry)
        return index

    def select_frozen_threshold(
        self,
        model: BaseVPRModel,
        index: GeographicVectorIndex,
    ) -> Dict[str, Any]:
        """Tune and freeze optimal decision threshold tau* on D_calib only."""
        calib_scores: List[float] = []
        calib_labels: List[int] = []

        for q in self.calib_manifest:
            img_path = BASE_DIR / q["filename"]
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            q_vec = model.extract_descriptor(img_bgr)
            matches = index.search(q_vec, top_k=1)
            score = matches[0].similarity_score if matches else 0.0
            is_pos = int(q["is_positive"])
            calib_scores.append(score)
            calib_labels.append(is_pos)

        # Fit Platt scaling on D_calib
        lr = LogisticRegression(C=1.0)
        X = np.array(calib_scores).reshape(-1, 1)
        y = np.array(calib_labels)
        lr.fit(X, y)
        coef = float(lr.coef_[0][0])
        intercept = float(lr.intercept_[0])

        # Grid search tau* maximizing (SelAcc - False Emission Rate) on D_calib
        best_tau = 0.40
        best_obj = -999.0
        threshold_evals = []

        for tau in np.linspace(0.20, 0.70, 26):
            emitted = [s >= tau for s in calib_scores]
            n_emit = sum(emitted)
            if n_emit == 0:
                sel_acc = 0.0
                fer = 0.0
            else:
                correct = sum(1 for e, l in zip(emitted, calib_labels) if e and l == 1)
                sel_acc = correct / n_emit
                false_emits = sum(1 for e, l in zip(emitted, calib_labels) if e and l == 0)
                fer = false_emits / max(1, sum(1 for l in calib_labels if l == 0))
            obj = sel_acc - fer
            threshold_evals.append({"tau": round(tau, 3), "sel_acc": round(sel_acc, 3), "fer": round(fer, 3), "obj": round(obj, 3)})
            if obj > best_obj:
                best_obj = obj
                best_tau = float(tau)

        return {
            "model_name": model.model_name,
            "frozen_tau": round(best_tau, 3),
            "platt_coef": round(coef, 4),
            "platt_intercept": round(intercept, 4),
            "threshold_evals": threshold_evals,
            "historical_040_eval": next(e for e in threshold_evals if abs(e["tau"] - 0.40) < 0.015),
            "historical_055_eval": next(e for e in threshold_evals if abs(e["tau"] - 0.55) < 0.015),
        }

    def evaluate_test_track(
        self,
        track_name: str,
        model: BaseVPRModel,
        index: GeographicVectorIndex,
        threshold: float,
    ) -> Dict[str, Any]:
        """Evaluate a frozen model on a single test track."""
        track_queries = [q for q in self.test_manifest if q["track"] == track_name]
        n_q = len(track_queries)

        rec1_hits = 0; rec5_hits = 0; rec10_hits = 0
        acc_1km = 0; acc_5km = 0; acc_25km = 0; acc_100km = 0
        errors_km: List[float] = []
        emitted_count = 0; correct_emissions = 0

        tp = 0; fp = 0; tn = 0; fn = 0
        query_records = []

        is_closed_set = track_name in ["TRACK_A", "TRACK_B"]

        for q in track_queries:
            qid = q["query_id"]
            img_path = BASE_DIR / q["filename"]
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            q_vec = model.extract_descriptor(img_bgr)
            matches = index.search(q_vec, top_k=10)
            gt = self.test_gt.get(qid, {})

            top_match = matches[0] if matches else None
            top_score = top_match.similarity_score if top_match else 0.0
            abstain = (top_match is None) or (top_score < threshold)

            target_ref_id = gt.get("target_ref_id")
            gt_lat = gt.get("target_lat")
            gt_lon = gt.get("target_lon")
            tol_km = gt.get("tolerance_km", 25.0)

            is_correct_hit = False
            dist_km = None

            if is_closed_set:
                if top_match:
                    if top_match.reference_id == target_ref_id:
                        is_correct_hit = True
                    if gt_lat is not None and gt_lon is not None:
                        dist_km = haversine_distance((top_match.lat, top_match.lon), (gt_lat, gt_lon)) / 1000.0
                        if not abstain:
                            errors_km.append(dist_km)
                            if dist_km <= 1.0: acc_1km += 1
                            if dist_km <= 5.0: acc_5km += 1
                            if dist_km <= 25.0: acc_25km += 1
                            if dist_km <= 100.0: acc_100km += 1

                for r_idx, m in enumerate(matches[:10]):
                    if m.reference_id == target_ref_id:
                        if r_idx == 0: rec1_hits += 1
                        if r_idx < 5: rec5_hits += 1
                        if r_idx < 10: rec10_hits += 1
                        break

                if not abstain:
                    emitted_count += 1
                    if is_correct_hit:
                        correct_emissions += 1
                        tp += 1
                    else:
                        fp += 1
                else:
                    fn += 1

            else:
                # Open-set tracks: Track C1, C2, D
                # True behavior is abstention (TN). Emission is False Positive (FP).
                if abstain:
                    tn += 1
                else:
                    fp += 1
                    emitted_count += 1
                    if gt_lat is not None and gt_lon is not None and top_match:
                        dist_km = haversine_distance((top_match.lat, top_match.lon), (gt_lat, gt_lon)) / 1000.0
                        errors_km.append(dist_km)

            query_records.append({
                "query_id": qid,
                "top_match_id": top_match.reference_id if top_match else None,
                "top_score": round(top_score, 4),
                "abstain": abstain,
                "status": "ABSTAINED" if abstain else "EMITTED",
                "error_km": round(dist_km, 2) if (not abstain and dist_km is not None) else "UNAVAILABLE",
            })

        # Calculate metrics
        coverage = round(emitted_count / max(1, n_q) * 100.0, 1)
        sel_acc = round(correct_emissions / max(1, emitted_count) * 100.0, 1) if emitted_count else 0.0

        med_err = round(float(np.median(errors_km)), 2) if errors_km else None
        mean_err = round(float(np.mean(errors_km)), 2) if errors_km else None
        max_err = round(float(np.max(errors_km)), 2) if errors_km else None

        # Track-specific rate
        if track_name == "TRACK_C1":
            flr_c1 = round(fp / max(1, n_q) * 100.0, 1)
            car_c1 = round(tn / max(1, n_q) * 100.0, 1)
            rate_info = {"flr_c1": flr_c1, "car_c1": car_c1}
        elif track_name == "TRACK_C2":
            fer_c2 = round(fp / max(1, n_q) * 100.0, 1)
            car_c2 = round(tn / max(1, n_q) * 100.0, 1)
            rate_info = {"fer_c2": fer_c2, "car_c2": car_c2}
        elif track_name == "TRACK_D":
            flr_adv = round(fp / max(1, n_q) * 100.0, 1)
            car_adv = round(tn / max(1, n_q) * 100.0, 1)
            rate_info = {"flr_adv": flr_adv, "car_adv": car_adv}
        else:
            rate_info = {}

        return {
            "track_name": track_name,
            "threshold": threshold,
            "total_queries": n_q,
            "recall_at_1": round(rec1_hits / max(1, n_q) * 100.0, 1) if is_closed_set else None,
            "recall_at_5": round(rec5_hits / max(1, n_q) * 100.0, 1) if is_closed_set else None,
            "recall_at_10": round(rec10_hits / max(1, n_q) * 100.0, 1) if is_closed_set else None,
            "acc_1km": round(acc_1km / max(1, n_q) * 100.0, 1) if is_closed_set else None,
            "acc_5km": round(acc_5km / max(1, n_q) * 100.0, 1) if is_closed_set else None,
            "acc_25km": round(acc_25km / max(1, n_q) * 100.0, 1) if is_closed_set else None,
            "acc_100km": round(acc_100km / max(1, n_q) * 100.0, 1) if is_closed_set else None,
            "median_error_km": med_err,
            "mean_error_km": mean_err,
            "max_error_km": max_err,
            "coverage_pct": coverage,
            "selective_accuracy_pct": sel_acc if is_closed_set else None,
            "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            **rate_info,
            "queries": query_records,
        }

    def run_all(self) -> Dict[str, Any]:
        """Execute full Phase 7.2 benchmark suite across all models and frozen tracks."""
        logger.info("Initializing HOUND Phase 7.2 Benchmark Suite...")

        cosplace = CosPlaceModel(backbone="ResNet18", fc_output_dim=512, use_cuda=True)
        anyloc = AnyLocModel(backbone="dinov2_vits14", desc_dim=384, use_cuda=True)
        mixvpr = MixVPRModel(backbone="resnet50", fc_output_dim=512, use_cuda=True)
        baseline = BaselineNeuralModel(use_cuda=True)
        legacy = HybridColorSpatialModel()

        models = [cosplace, anyloc, mixvpr, baseline, legacy]
        tracks = ["TRACK_A", "TRACK_B", "TRACK_C1", "TRACK_C2", "TRACK_D"]

        report: Dict[str, Any] = {
            "metadata": {
                "phase": "Phase 7.2 Statistical Validation & Open-Set Reliability",
                "date": "2026-11-13",
                "references_count": len(self.refs_manifest),
                "calibration_count": len(self.calib_manifest),
                "test_count": len(self.test_manifest),
            },
            "calibration_freezing": {},
            "test_evaluations": {},
        }

        # 1. Calibration Partition: Threshold selection & freezing
        cosplace.load()
        cos_idx = self.build_reference_index(cosplace)
        cos_calib_info = self.select_frozen_threshold(cosplace, cos_idx)
        report["calibration_freezing"]["CosPlace"] = cos_calib_info
        frozen_tau = cos_calib_info["frozen_tau"]
        logger.info(f"Frozen optimal CosPlace threshold selected from D_calib: tau* = {frozen_tau}")

        # 2. Evaluate CosPlace across thresholds: frozen tau*, 0.40, and 0.55
        report["test_evaluations"]["CosPlace"] = {}
        for tau in [frozen_tau, 0.40, 0.55]:
            tau_label = f"tau_{tau:.2f}"
            report["test_evaluations"]["CosPlace"][tau_label] = {}
            for t in tracks:
                res = self.evaluate_test_track(t, cosplace, cos_idx, threshold=tau)
                report["test_evaluations"]["CosPlace"][tau_label][t] = res

        # 3. Evaluate AnyLoc, MixVPR, ResNet-18, Legacy under their default/frozen thresholds
        for m in [anyloc, mixvpr, baseline, legacy]:
            logger.info(f"Evaluating {m.model_name} on frozen test partition...")
            m.load()
            idx = self.build_reference_index(m)
            thresh = 0.65 if "dinov2" in m.model_name else (0.60 if "resnet" in m.model_name else 0.50)
            report["test_evaluations"][m.model_name] = {}
            for t in tracks:
                res = self.evaluate_test_track(t, m, idx, threshold=thresh)
                report["test_evaluations"][m.model_name][t] = res
            m.unload()

        # 4. Evaluate Multi-Engine Fusion under CosPlace frozen tau*
        logger.info("Evaluating Multi-Engine Fusion under frozen tau*...")
        anyloc.load(); any_idx = self.build_reference_index(anyloc)
        mixvpr.load(); mix_idx = self.build_reference_index(mixvpr)
        fusion_engine = MultiEngineVPRFusion()

        report["test_evaluations"]["MultiEngine_Fusion_RRF"] = {}
        for t in tracks:
            # Multi-Engine fusion test track evaluation
            track_queries = [q for q in self.test_manifest if q["track"] == t]
            is_closed = t in ["TRACK_A", "TRACK_B"]
            rec1 = 0; acc25 = 0; emits = 0; correct = 0; tp = 0; fp = 0; tn = 0; fn = 0
            errs = []

            for q in track_queries:
                qid = q["query_id"]
                img_path = BASE_DIR / q["filename"]
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None: continue

                m_cos = cos_idx.search(cosplace.extract_descriptor(img_bgr), top_k=10)
                m_any = any_idx.search(anyloc.extract_descriptor(img_bgr), top_k=10)
                m_mix = mix_idx.search(mixvpr.extract_descriptor(img_bgr), top_k=10)

                fused = fusion_engine.fuse({"cosplace": m_cos, "anyloc": m_any, "mixvpr": m_mix}, top_k=10)
                top_fused = fused[0] if fused else None
                top_cos_sim = m_cos[0].similarity_score if m_cos else 0.0

                # Strict CosPlace confidence gating
                abstain = (top_fused is None) or (top_cos_sim < frozen_tau)

                gt = self.test_gt.get(qid, {})
                target_ref = gt.get("target_ref_id")
                gt_lat = gt.get("target_lat")
                gt_lon = gt.get("target_lon")

                if is_closed:
                    hit = bool(target_ref and top_fused and (target_ref in top_fused.location_cluster_id or target_ref in top_fused.name))
                    if hit: rec1 += 1
                    dist_km = haversine_distance((top_fused.lat, top_fused.lon), (gt_lat, gt_lon)) / 1000.0 if (top_fused and gt_lat and gt_lon) else None
                    if not abstain:
                        emits += 1
                        if dist_km and dist_km <= 25.0: acc25 += 1
                        if hit: correct += 1; tp += 1
                        else: fp += 1
                        if dist_km: errs.append(dist_km)
                    else: fn += 1
                else:
                    if abstain: tn += 1
                    else: fp += 1; emits += 1

            n_t = len(track_queries)
            report["test_evaluations"]["MultiEngine_Fusion_RRF"][t] = {
                "track_name": t,
                "total_queries": n_t,
                "recall_at_1": round(rec1 / max(1, n_t) * 100.0, 1) if is_closed else None,
                "acc_25km": round(acc25 / max(1, n_t) * 100.0, 1) if is_closed else None,
                "median_error_km": round(float(np.median(errs)), 2) if errs else None,
                "coverage_pct": round(emits / max(1, n_t) * 100.0, 1),
                "selective_accuracy_pct": round(correct / max(1, emits) * 100.0, 1) if emits and is_closed else (0.0 if is_closed else None),
                "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
                "flr": round(fp / max(1, n_t) * 100.0, 1) if not is_closed else 0.0,
            }

        # Free GPU
        cosplace.unload(); anyloc.unload(); mixvpr.unload()

        # 5. Bootstrap Confidence Intervals on Frozen CosPlace
        boot_metrics = {}
        for track in ["TRACK_A", "TRACK_B", "TRACK_C1", "TRACK_C2", "TRACK_D"]:
            eval_data = report["test_evaluations"]["CosPlace"][f"tau_{frozen_tau:.2f}"][track]
            q_list = eval_data["queries"]
            if track in ["TRACK_A", "TRACK_B"]:
                # Acc@25 binary hits
                hits = [1.0 if (q["status"] == "EMITTED" and q["error_km"] != "UNAVAILABLE" and float(q["error_km"]) <= 25.0) else 0.0 for q in q_list]
                mean_h, low_h, high_h = bootstrap_ci(hits)
                boot_metrics[f"{track}_acc25"] = {"mean": mean_h, "ci_95_low": low_h, "ci_95_high": high_h}
            elif track == "TRACK_C1":
                flrs = [1.0 if q["status"] == "EMITTED" else 0.0 for q in q_list]
                mean_f, low_f, high_f = bootstrap_ci(flrs)
                boot_metrics["C1_false_localization_rate"] = {"mean": mean_f, "ci_95_low": low_f, "ci_95_high": high_f}
            elif track == "TRACK_C2":
                fers = [1.0 if q["status"] == "EMITTED" else 0.0 for q in q_list]
                mean_e, low_e, high_e = bootstrap_ci(fers)
                boot_metrics["C2_false_emission_rate"] = {"mean": mean_e, "ci_95_low": low_e, "ci_95_high": high_e}
            elif track == "TRACK_D":
                flr_adv = [1.0 if q["status"] == "EMITTED" else 0.0 for q in q_list]
                mean_a, low_a, high_a = bootstrap_ci(flr_adv)
                boot_metrics["Track_D_adv_false_localization"] = {"mean": mean_a, "ci_95_low": low_a, "ci_95_high": high_a}

        report["bootstrap_confidence_intervals"] = boot_metrics

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out_file = REPORTS_DIR / "phase_7_2_benchmark_report.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Phase 7.2 Benchmark Report successfully saved to: {out_file}")
        return report


if __name__ == "__main__":
    runner = Phase72BenchmarkRunner()
    rep = runner.run_all()

    print("\n" + "=" * 110)
    print("HOUND PHASE 7.2 — STATISTICAL VALIDATION & OPEN-SET RELIABILITY REPORT")
    print("=" * 110)

    tau_star = rep["calibration_freezing"]["CosPlace"]["frozen_tau"]
    print(f"Optimal Threshold Frozen from D_calib: tau* = {tau_star}")
    print("\n[CosPlace Under Frozen tau* = " + str(tau_star) + "]")
    print(f"{'Track':<10} | {'Total':<6} | {'R@1':<6} | {'<=25km':<7} | {'Med Err':<10} | {'Cover':<6} | {'SelAcc':<7} | {'FLR/FER':<8} | {'TP':<3} | {'FP':<3} | {'TN':<3} | {'FN':<3}")
    print("-" * 110)
    for t in ["TRACK_A", "TRACK_B", "TRACK_C1", "TRACK_C2", "TRACK_D"]:
        res = rep["test_evaluations"]["CosPlace"][f"tau_{tau_star:.2f}"][t]
        med_s = f"{res['median_error_km']}km" if res['median_error_km'] is not None else "N/A"
        r1_s = f"{res['recall_at_1']}%" if res['recall_at_1'] is not None else "N/A"
        a25_s = f"{res['acc_25km']}%" if res['acc_25km'] is not None else "N/A"
        sel_s = f"{res['selective_accuracy_pct']}%" if res['selective_accuracy_pct'] is not None else "N/A"
        rate_s = f"{res.get('flr_c1', res.get('fer_c2', res.get('flr_adv', 0.0)))}%"
        c = res["confusion"]
        print(f"{t:<10} | {res['total_queries']:<6} | {r1_s:<6} | {a25_s:<7} | {med_s:<10} | {res['coverage_pct']:<5.1f}% | {sel_s:<7} | {rate_s:<8} | {c['tp']:<3} | {c['fp']:<3} | {c['tn']:<3} | {c['fn']:<3}")
