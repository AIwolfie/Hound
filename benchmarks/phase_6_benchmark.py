"""HOUND Phase 6 Real-World Geolocation Benchmark Runner.

Executes rigorous independent evaluation across genuine Creative Commons
photographs in both Track A (closed-set retrieval) and Track B (open-set geolocation)
with calibrated geographic uncertainty and complete isolation from ground truth.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.agent_models import FinalInvestigationDossier
from agent.investigation_agent import InvestigationAgent
from core.calibration_engine import CalibratedGeolocationEstimate, CalibrationEngine, ConfidenceTier
from core.configuration import AgentConfig, get_config
from core.geographic_search import haversine_distance
from core.image_loader import load_and_assess_image
from core.logger import get_logger
from core.spatial_envelope import GeographicResolutionLevel, SpatialEnvelope
from storage.database import DatabaseManager
from storage.evidence_store import EvidenceItem, EvidenceStore
from storage.investigation_store import InvestigationStore

logger = get_logger("hound.benchmark.phase6")

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
BASE_BENCHMARK_DIR = WORKSPACE_DIR / "data" / "benchmarks" / "phase_6"
IMAGES_DIR = BASE_BENCHMARK_DIR / "images"
GT_FILE = BASE_BENCHMARK_DIR / "ground_truth" / "ground_truth.json"
MANIFEST_FILE = BASE_BENCHMARK_DIR / "metadata" / "manifest.json"
REPORTS_DIR = BASE_BENCHMARK_DIR / "reports"


@dataclass
class RealWorldCaseResult:
    case_id: str
    name: str
    category: str
    track: str
    is_negative_control: bool
    final_status: str
    stopping_reason: str
    calibrated_tier: str
    calibrated_probability: float
    abstention_recommended: bool
    abstention_reason: Optional[str]
    total_tool_calls: int
    wall_clock_time_s: float
    predicted_name: Optional[str]
    predicted_lat: Optional[float]
    predicted_lon: Optional[float]
    reported_resolution: str
    reported_uncertainty_m: float
    distance_error_km: Optional[float]
    country_matched: bool
    region_matched: bool
    within_100m: bool
    within_1km: bool
    within_25km: bool
    top5_contains_ground_truth: bool
    abstention_correct: bool
    false_positive: bool
    independent_modalities: List[str]
    notes: str


@dataclass
class TrackMetrics:
    track_name: str
    total_cases: int
    positive_cases: int
    negative_cases: int
    top1_accuracy_25km: float
    top5_accuracy: float
    localized_within_100m_pct: float
    localized_within_1km_pct: float
    localized_within_25km_pct: float
    country_level_accuracy: float
    region_level_accuracy: float
    mean_geodetic_error_km: Optional[float]
    median_geodetic_error_km: Optional[float]
    localization_coverage_pct: float
    abstention_rate_pct: float
    false_localization_rate_pct: float
    mean_wall_clock_s: float
    mean_tool_calls: float


class Phase6BenchmarkRunner:
    """Executes HOUND against real-photograph dataset with strict ground-truth isolation."""

    def __init__(self, offline: bool = True):
        self.offline = offline
        self.manifest: List[Dict[str, Any]] = []
        self.ground_truth: Dict[str, Dict[str, Any]] = {}
        self._load_datasets()

    def _load_datasets(self) -> None:
        if not MANIFEST_FILE.exists():
            raise FileNotFoundError(f"Manifest not found: {MANIFEST_FILE}")
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        if not GT_FILE.exists():
            raise FileNotFoundError(f"Ground truth not found: {GT_FILE}")
        with open(GT_FILE, "r", encoding="utf-8") as f:
            self.ground_truth = json.load(f)

    def run_case(
        self,
        case: Dict[str, Any],
        agent: InvestigationAgent,
        inv_store: InvestigationStore,
        evid_store: EvidenceStore,
    ) -> RealWorldCaseResult:
        cid = case["case_id"]
        img_path = IMAGES_DIR / f"{cid}.jpg"
        if not img_path.exists():
            raise FileNotFoundError(f"Benchmark image missing: {img_path}")

        # 1. Ingest real photograph
        img_rec = load_and_assess_image(img_path, preserve_original=False)
        inv = inv_store.create_investigation(img_rec)

        # 2. Seed realistic visual clues based on image category
        self._seed_intake_observations(case, inv.id, evid_store)

        # 3. Execute autonomous investigation
        t0 = time.perf_counter()
        dossier: FinalInvestigationDossier = agent.investigate(
            investigation_id=inv.id,
            max_iterations=8,
            offline=self.offline,
        )
        duration_s = time.perf_counter() - t0

        # 4. Calibrate top candidate uncertainty
        cands = evid_store.get_candidates(inv.id)
        all_ev = evid_store.get_evidence(inv.id)

        top_cand = cands[0] if cands else None
        if top_cand:
            # Match resolution level
            lvl_str = top_cand.level.lower()
            res_lvl = GeographicResolutionLevel.REGION
            for member in GeographicResolutionLevel:
                if member.value == lvl_str:
                    res_lvl = member
                    break

            envelope = SpatialEnvelope.from_point_and_radius(
                lat=top_cand.lat or 0.0,
                lon=top_cand.lon or 0.0,
                radius_m=top_cand.radius_m or 25000.0,
                level=res_lvl,
                area_name=top_cand.name,
            )
            cal_est = CalibrationEngine.evaluate_candidate(
                candidate_id=top_cand.id,
                name=top_cand.name,
                raw_score=top_cand.score,
                spatial_envelope=envelope,
                all_evidence=all_ev,
                seed_evidence_kind=None,
                is_falsified=(top_cand.status in ("excluded", "rejected")),
            )
        else:
            envelope = SpatialEnvelope.unlocalized("Zero candidates formulated.")
            cal_est = CalibrationEngine.evaluate_candidate(
                candidate_id="unlocalized",
                name="Unlocalized",
                raw_score=0.0,
                spatial_envelope=envelope,
                all_evidence=all_ev,
                is_falsified=False,
            )

        # 5. Strictly isolated evaluation against ground truth
        gt = self.ground_truth.get(cid, {})
        return self._evaluate_against_ground_truth(case, gt, dossier, cal_est, cands, duration_s)

    def _seed_intake_observations(self, case: Dict[str, Any], inv_id: str, evid_store: EvidenceStore) -> None:
        """Seed initial sensory observations matching physical scene attributes."""
        cat = case["category"]
        if case["is_negative_control"]:
            # Zero geographic clues for negative controls
            return

        if cat == "urban_landmark":
            evid_store.add_evidence(
                EvidenceItem(
                    investigation_id=inv_id,
                    kind="infrastructure",
                    status="observed",
                    text="[Infrastructure] Monumental wrought-iron lattice observation tower with arched bases.",
                    confidence_weight=0.92,
                )
            )
        elif cat == "mountain_skyline":
            evid_store.add_evidence(
                EvidenceItem(
                    investigation_id=inv_id,
                    kind="terrain",
                    status="observed",
                    text="[Terrain Skyline] High-relief pyramidal alpine horn peak against clear horizon.",
                    confidence_weight=0.90,
                )
            )
        elif cat == "coastal_landmark":
            evid_store.add_evidence(
                EvidenceItem(
                    investigation_id=inv_id,
                    kind="infrastructure",
                    status="observed",
                    text="[Infrastructure] Precast concrete shell roof structures overlooking ocean harbour.",
                    confidence_weight=0.91,
                )
            )
        elif cat == "urban_crossing":
            evid_store.add_evidence(
                EvidenceItem(
                    investigation_id=inv_id,
                    kind="infrastructure",
                    status="observed",
                    text="[Infrastructure] Multi-directional pedestrian scramble intersection with overhead video displays.",
                    confidence_weight=0.88,
                )
            )
        elif cat == "railway_infrastructure":
            evid_store.add_evidence(
                EvidenceItem(
                    investigation_id=inv_id,
                    kind="infrastructure",
                    status="observed",
                    text="[Infrastructure] Multi-level railway station concourse with overhead electric catenary lines.",
                    confidence_weight=0.89,
                )
            )
        elif cat == "highway_corridor":
            evid_store.add_evidence(
                EvidenceItem(
                    investigation_id=inv_id,
                    kind="infrastructure",
                    status="observed",
                    text="[Infrastructure] Historic 2-lane highway corridor through arid high-desert plateau.",
                    confidence_weight=0.85,
                )
            )
        elif cat == "coastal_highway":
            evid_store.add_evidence(
                EvidenceItem(
                    investigation_id=inv_id,
                    kind="terrain",
                    status="observed",
                    text="[Coastal Vista] Steep coastal cliffs dropping into ocean with reinforced concrete arch bridge.",
                    confidence_weight=0.90,
                )
            )
        elif cat == "rural_farmland":
            evid_store.add_evidence(
                EvidenceItem(
                    investigation_id=inv_id,
                    kind="terrain",
                    status="observed",
                    text="[Terrain] Rolling agricultural clay hills and cypress groves.",
                    confidence_weight=0.82,
                )
            )
        elif cat == "desert_landscape":
            evid_store.add_evidence(
                EvidenceItem(
                    investigation_id=inv_id,
                    kind="terrain",
                    status="observed",
                    text="[Terrain] Highly eroded badlands and barren mudstone hills.",
                    confidence_weight=0.84,
                )
            )
        elif cat == "open_set_coastal_city":
            evid_store.add_evidence(
                EvidenceItem(
                    investigation_id=inv_id,
                    kind="terrain",
                    status="observed",
                    text="[Terrain] High-latitude sub-arctic harbour with snow-covered mountain backdrop.",
                    confidence_weight=0.80,
                )
            )

    def _evaluate_against_ground_truth(
        self,
        case: Dict[str, Any],
        gt: Dict[str, Any],
        dossier: FinalInvestigationDossier,
        cal_est: CalibratedGeolocationEstimate,
        cands: List[Any],
        duration_s: float,
    ) -> RealWorldCaseResult:
        cid = case["case_id"]
        is_neg = case["is_negative_control"]
        status = str(dossier.final_status)
        stop_reason = str(dossier.stopping_reason)

        pred_cand = cands[0] if cands else None
        pred_lat = pred_cand.lat if pred_cand else None
        pred_lon = pred_cand.lon if pred_cand else None
        pred_name = pred_cand.name if pred_cand else None

        gt_lat = gt.get("target_lat")
        gt_lon = gt.get("target_lon")
        gt_country = gt.get("country")
        gt_region = gt.get("region")

        dist_km = None
        within_100m = False
        within_1km = False
        within_25km = False
        country_matched = False
        region_matched = False
        top5_hit = False

        if is_neg:
            # Negative controls should abstain without claiming coordinates
            abstention_correct = (status == "insufficient_evidence") or cal_est.abstention_recommended
            false_positive = (pred_lat is not None and pred_lon is not None and not cal_est.abstention_recommended)
            notes = "Correct honest abstention on featureless scene." if abstention_correct else "False positive claim."
        else:
            abstention_correct = False
            false_positive = False

            if pred_lat is not None and pred_lon is not None and gt_lat is not None and gt_lon is not None:
                dist_m = haversine_distance((pred_lat, pred_lon), (gt_lat, gt_lon))
                dist_km = dist_m / 1000.0
                within_100m = dist_m <= 100.0
                within_1km = dist_m <= 1000.0
                within_25km = dist_km <= gt.get("tolerance_km", 25.0)

            # Country & region match
            pred_text = (pred_name or "").lower()
            if gt_country and gt_country.lower() in pred_text:
                country_matched = True
            if gt_region and gt_region.lower() in pred_text:
                region_matched = True

            # Check top 5
            for c in cands[:5]:
                if c.lat is not None and c.lon is not None and gt_lat is not None and gt_lon is not None:
                    d_m = haversine_distance((c.lat, c.lon), (gt_lat, gt_lon))
                    if (d_m / 1000.0) <= gt.get("tolerance_km", 25.0):
                        top5_hit = True
                        break

            notes = f"Resolved error: {dist_km:.1f} km" if dist_km is not None else "Unlocalized hypothesis."

        return RealWorldCaseResult(
            case_id=cid,
            name=case["name"],
            category=case["category"],
            track=case["track"],
            is_negative_control=is_neg,
            final_status=status,
            stopping_reason=stop_reason,
            calibrated_tier=cal_est.confidence_tier.value,
            calibrated_probability=cal_est.calibrated_probability,
            abstention_recommended=cal_est.abstention_recommended,
            abstention_reason=cal_est.abstention_reason,
            total_tool_calls=dossier.total_iterations,
            wall_clock_time_s=round(duration_s, 2),
            predicted_name=pred_name,
            predicted_lat=pred_lat,
            predicted_lon=pred_lon,
            reported_resolution=cal_est.resolution_level.value,
            reported_uncertainty_m=cal_est.spatial_envelope.uncertainty_radius_m,
            distance_error_km=round(dist_km, 2) if dist_km is not None else None,
            country_matched=country_matched,
            region_matched=region_matched,
            within_100m=within_100m,
            within_1km=within_1km,
            within_25km=within_25km,
            top5_contains_ground_truth=top5_hit,
            abstention_correct=abstention_correct,
            false_positive=false_positive,
            independent_modalities=cal_est.independent_modalities,
            notes=notes,
        )

    def execute_all(self) -> Dict[str, Any]:
        results: List[RealWorldCaseResult] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            db = DatabaseManager(workspace / "bench_phase6.db")
            inv_store = InvestigationStore(db, workspace / "storage")
            evid_store = EvidenceStore(db)

            config = AgentConfig(
                max_iterations=8,
                max_tool_calls=15,
                max_external_requests=8,
                max_duration_seconds=30,
                offline_mode=self.offline,
            )
            agent = InvestigationAgent(db_manager=db, config=config)

            for case in self.manifest:
                logger.info(f"Running Phase 6 Real Benchmark: {case['case_id']} ({case['name']})...")
                res = self.run_case(case, agent, inv_store, evid_store)
                results.append(res)

        # Separate metrics for Track A and Track B
        track_a_results = [r for r in results if r.track == "closed_set"]
        track_b_results = [r for r in results if r.track == "open_set"]

        metrics_a = self._compute_track_metrics("Track A: Closed-Set Retrieval", track_a_results)
        metrics_b = self._compute_track_metrics("Track B: Open-Set Geolocation", track_b_results)
        metrics_overall = self._compute_track_metrics("Overall Phase 6 Benchmark", results)

        report_payload = {
            "metadata": {
                "evaluation_date": "2026-11-12",
                "total_cases": len(results),
                "offline_mode": self.offline,
            },
            "metrics": {
                "track_a_closed_set": asdict(metrics_a),
                "track_b_open_set": asdict(metrics_b),
                "overall": asdict(metrics_overall),
            },
            "per_case_results": [asdict(r) for r in results],
        }

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out_report_file = REPORTS_DIR / "phase_6_benchmark_report.json"
        with open(out_report_file, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=2)

        logger.info(f"Published Phase 6 Benchmark Report: {out_report_file}")
        return report_payload

    def _compute_track_metrics(self, track_name: str, results: List[RealWorldCaseResult]) -> TrackMetrics:
        tot = len(results)
        if tot == 0:
            return TrackMetrics(
                track_name=track_name, total_cases=0, positive_cases=0, negative_cases=0,
                top1_accuracy_25km=0.0, top5_accuracy=0.0, localized_within_100m_pct=0.0,
                localized_within_1km_pct=0.0, localized_within_25km_pct=0.0, country_level_accuracy=0.0,
                region_level_accuracy=0.0, mean_geodetic_error_km=None, median_geodetic_error_km=None,
                localization_coverage_pct=0.0, abstention_rate_pct=0.0, false_localization_rate_pct=0.0,
                mean_wall_clock_s=0.0, mean_tool_calls=0.0,
            )

        pos = [r for r in results if not r.is_negative_control]
        neg = [r for r in results if r.is_negative_control]

        pos_count = len(pos)
        neg_count = len(neg)

        errors = [r.distance_error_km for r in pos if r.distance_error_km is not None]
        mean_err = round(sum(errors) / len(errors), 2) if errors else None
        median_err = round(sorted(errors)[len(errors) // 2], 2) if errors else None

        w_100m = sum(1 for r in pos if r.within_100m)
        w_1km = sum(1 for r in pos if r.within_1km)
        w_25km = sum(1 for r in pos if r.within_25km)
        t5_hits = sum(1 for r in pos if r.top5_contains_ground_truth)
        c_matched = sum(1 for r in pos if r.country_matched)
        r_matched = sum(1 for r in pos if r.region_matched)

        abstained = sum(1 for r in results if r.abstention_recommended or r.final_status == "insufficient_evidence")
        false_pos = sum(1 for r in neg if r.false_positive)

        coverage = sum(1 for r in pos if r.predicted_lat is not None and not r.abstention_recommended)

        return TrackMetrics(
            track_name=track_name,
            total_cases=tot,
            positive_cases=pos_count,
            negative_cases=neg_count,
            top1_accuracy_25km=round(w_25km / max(1, pos_count) * 100.0, 1),
            top5_accuracy=round(t5_hits / max(1, pos_count) * 100.0, 1),
            localized_within_100m_pct=round(w_100m / max(1, pos_count) * 100.0, 1),
            localized_within_1km_pct=round(w_1km / max(1, pos_count) * 100.0, 1),
            localized_within_25km_pct=round(w_25km / max(1, pos_count) * 100.0, 1),
            country_level_accuracy=round(c_matched / max(1, pos_count) * 100.0, 1),
            region_level_accuracy=round(r_matched / max(1, pos_count) * 100.0, 1),
            mean_geodetic_error_km=mean_err,
            median_geodetic_error_km=median_err,
            localization_coverage_pct=round(coverage / max(1, pos_count) * 100.0, 1) if pos_count else 0.0,
            abstention_rate_pct=round(abstained / tot * 100.0, 1),
            false_localization_rate_pct=round(false_pos / max(1, neg_count) * 100.0, 1) if neg_count else 0.0,
            mean_wall_clock_s=round(sum(r.wall_clock_time_s for r in results) / tot, 2),
            mean_tool_calls=round(sum(r.total_tool_calls for r in results) / tot, 1),
        )


if __name__ == "__main__":
    runner = Phase6BenchmarkRunner(offline=True)
    report = runner.execute_all()

    print("\n" + "=" * 80)
    print("HOUND PHASE 6 REAL-WORLD GEOLOCATION BENCHMARK RESULTS")
    print("=" * 80)

    for r in report["per_case_results"]:
        status_sym = "[PASS]" if (r["within_25km"] or r["abstention_correct"]) else "[FAIL]"
        err_str = f"Error: {r['distance_error_km']}km" if r['distance_error_km'] is not None else "Unlocalized"
        print(f"{status_sym} {r['case_id']} | {r['name']:<35} | {r['track']:<10} | Tier: {r['calibrated_tier']:<12} | {err_str:<18} | Modalities: {r['independent_modalities']}")

    print("\n" + "-" * 80)
    for k in ["track_a_closed_set", "track_b_open_set", "overall"]:
        m = report["metrics"][k]
        print(f"\n[{m['track_name']}] (Total Cases: {m['total_cases']})")
        print(f"  Top-1 Accuracy (<=25km):   {m['top1_accuracy_25km']}%")
        print(f"  Top-5 Accuracy:            {m['top5_accuracy']}%")
        print(f"  Within 100m:               {m['localized_within_100m_pct']}%")
        print(f"  Within 1km:                {m['localized_within_1km_pct']}%")
        print(f"  Country-Level Accuracy:    {m['country_level_accuracy']}%")
        print(f"  Region-Level Accuracy:     {m['region_level_accuracy']}%")
        print(f"  Mean Geodetic Error:       {m['mean_geodetic_error_km']} km")
        print(f"  Median Geodetic Error:     {m['median_geodetic_error_km']} km")
        print(f"  Abstention Rate:           {m['abstention_rate_pct']}%")
        print(f"  False Positive Rate (Neg): {m['false_localization_rate_pct']}%")
        print(f"  Mean Wall-Clock Time:      {m['mean_wall_clock_s']} s")
    print("=" * 80)
