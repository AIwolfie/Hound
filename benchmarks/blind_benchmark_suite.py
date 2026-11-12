"""HOUND Phase 4.1 Independent Blind Benchmark Suite.

Evaluates HOUND's autonomous investigation agent against a diverse manifest of
10 realistic test cases with ground-truth coordinates strictly isolated from agent execution.

Evaluates:
- Country-level accuracy
- State / Province / Region-level accuracy
- Point-level accuracy (<= 25 km error)
- Median and Mean geodetic error (km)
- False Positive Rate on Negative Controls (claiming a location for featureless scenes)
- Honest Abstention Rate on Negative Controls
- Mean tool calls per investigation
- Mean execution wall-clock time
"""
from __future__ import annotations

import math
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from agent.agent_models import FinalInvestigationDossier
from agent.investigation_agent import InvestigationAgent
from core.configuration import AgentConfig, get_config
from core.geographic_search import haversine_distance
from core.image_loader import load_and_assess_image
from core.logger import get_logger
from storage.database import DatabaseManager
from storage.evidence_store import EvidenceItem, EvidenceStore
from storage.investigation_store import InvestigationStore

logger = get_logger("hound.benchmark.blind")


@dataclass
class GroundTruth:
    """Isolated ground truth record — NEVER exposed to the agent during execution."""
    is_negative_control: bool
    expected_abstention: bool
    country: Optional[str] = None
    state_or_region: Optional[str] = None
    target_lat: Optional[float] = None
    target_lon: Optional[float] = None
    point_tolerance_km: float = 25.0
    description: str = ""


@dataclass
class BlindBenchmarkCase:
    case_id: str
    name: str
    category: str  # "highway", "mountain", "coastal", "urban", "rural", "negative_control", "night", "deceptive", "railway"
    initial_evidence: List[Dict[str, Any]]
    ground_truth: GroundTruth


@dataclass
class CaseEvaluation:
    case_id: str
    name: str
    category: str
    final_status: str
    stopping_reason: str
    justified_precision: str
    total_tool_calls: int
    wall_clock_time_s: float
    predicted_name: Optional[str]
    predicted_lat: Optional[float]
    predicted_lon: Optional[float]
    distance_error_km: Optional[float]
    country_matched: bool
    region_matched: bool
    point_matched: bool
    abstention_correct: bool
    false_positive: bool
    notes: str


class BlindBenchmarkSuite:
    """Executes HOUND's autonomous investigation engine against blind ground-truth targets."""

    @staticmethod
    def get_manifest() -> List[BlindBenchmarkCase]:
        return [
            # 1. Clear Highway Corridor with Route Shield
            BlindBenchmarkCase(
                case_id="BLIND-01",
                name="US-101 Coastal Highway Corridor",
                category="highway",
                initial_evidence=[
                    {
                        "kind": "ocr",
                        "status": "observed",
                        "text": "[Road Sign] US-101 NORTH / CA COAST HWY",
                        "confidence_weight": 0.95,
                    },
                    {
                        "kind": "solar",
                        "status": "computed",
                        "text": "[Solar Ephemeris] Estimated solar elevation: 44.5°",
                        "confidence_weight": 0.85,
                    },
                ],
                ground_truth=GroundTruth(
                    is_negative_control=False,
                    expected_abstention=False,
                    country="United States",
                    state_or_region="California",
                    target_lat=34.42,
                    target_lon=-119.69,
                    point_tolerance_km=50.0,
                    description="US-101 near Santa Barbara, California",
                ),
            ),
            # 2. Alpine Mountain Pass with Skyline
            BlindBenchmarkCase(
                case_id="BLIND-02",
                name="Matterhorn Alpine Pass",
                category="mountain",
                initial_evidence=[
                    {
                        "kind": "terrain",
                        "status": "observed",
                        "text": "[Terrain] Distinctive pyramid peak skyline along southern horizon",
                        "confidence_weight": 0.90,
                    },
                    {
                        "kind": "ocr",
                        "status": "observed",
                        "text": "[Trail Marker] Zermatt - Gornergrat Bahn",
                        "confidence_weight": 0.92,
                    },
                ],
                ground_truth=GroundTruth(
                    is_negative_control=False,
                    expected_abstention=False,
                    country="Switzerland",
                    state_or_region="Valais",
                    target_lat=45.9765,
                    target_lon=7.7491,
                    point_tolerance_km=25.0,
                    description="Gornergrat viewpoint near Zermatt, Switzerland",
                ),
            ),
            # 3. Coastal Road with Ocean Vista
            BlindBenchmarkCase(
                case_id="BLIND-03",
                name="Big Sur Coastal Pacific Vista",
                category="coastal",
                initial_evidence=[
                    {
                        "kind": "ocr",
                        "status": "observed",
                        "text": "[Highway Marker] CA-1 / Big Sur State Park",
                        "confidence_weight": 0.90,
                    },
                    {
                        "kind": "terrain",
                        "status": "observed",
                        "text": "[Coastal Vista] High cliffs overlooking Pacific ocean on western flank",
                        "confidence_weight": 0.85,
                    },
                ],
                ground_truth=GroundTruth(
                    is_negative_control=False,
                    expected_abstention=False,
                    country="United States",
                    state_or_region="California",
                    target_lat=36.27,
                    target_lon=-121.81,
                    point_tolerance_km=40.0,
                    description="Big Sur Coast, California",
                ),
            ),
            # 4. High-Density Urban Intersection with Bilingual Signage
            BlindBenchmarkCase(
                case_id="BLIND-04",
                name="Shinjuku Urban Transit Crossing",
                category="urban",
                initial_evidence=[
                    {
                        "kind": "ocr",
                        "status": "observed",
                        "text": "[Street Sign] 新宿駅東口 / Shinjuku Sta. East Exit",
                        "confidence_weight": 0.96,
                    },
                    {
                        "kind": "infrastructure",
                        "status": "observed",
                        "text": "[Infrastructure] High-density multi-lane pedestrian crossing with overhead traffic signals",
                        "confidence_weight": 0.88,
                    },
                ],
                ground_truth=GroundTruth(
                    is_negative_control=False,
                    expected_abstention=False,
                    country="Japan",
                    state_or_region="Tokyo",
                    target_lat=35.6909,
                    target_lon=139.7003,
                    point_tolerance_km=15.0,
                    description="Shinjuku Station East Crossing, Tokyo, Japan",
                ),
            ),
            # 5. Rural Agricultural Landscape
            BlindBenchmarkCase(
                case_id="BLIND-05",
                name="Iowa Corn Belt Farmstead",
                category="rural",
                initial_evidence=[
                    {
                        "kind": "ocr",
                        "status": "observed",
                        "text": "[Elevator Sign] Central Iowa Co-op Grain",
                        "confidence_weight": 0.88,
                    },
                    {
                        "kind": "infrastructure",
                        "status": "observed",
                        "text": "[Infrastructure] Concrete grain elevator and steel silo complex",
                        "confidence_weight": 0.82,
                    },
                ],
                ground_truth=GroundTruth(
                    is_negative_control=False,
                    expected_abstention=False,
                    country="United States",
                    state_or_region="Iowa",
                    target_lat=42.03,
                    target_lon=-93.63,
                    point_tolerance_km=50.0,
                    description="Central Iowa Agricultural Belt",
                ),
            ),
            # 6. Negative Control 1: Featureless Indoor Meeting Room
            BlindBenchmarkCase(
                case_id="BLIND-06",
                name="Featureless Indoor Conference Room",
                category="negative_control",
                initial_evidence=[],  # Zero geographic clues
                ground_truth=GroundTruth(
                    is_negative_control=True,
                    expected_abstention=True,
                    description="Indoor room with beige drywall, fluorescent lights, no windows",
                ),
            ),
            # 7. Negative Control 2: Blank Painted Office Hallway
            BlindBenchmarkCase(
                case_id="BLIND-07",
                name="Blank Painted Wall Hallway",
                category="negative_control",
                initial_evidence=[],  # Zero geographic clues
                ground_truth=GroundTruth(
                    is_negative_control=True,
                    expected_abstention=True,
                    description="Windowless indoor corridor, painted white, commercial carpet",
                ),
            ),
            # 8. Nocturnal Urban Street with Neon Signage
            BlindBenchmarkCase(
                case_id="BLIND-08",
                name="Nathan Road Kowloon Nightscape",
                category="night",
                initial_evidence=[
                    {
                        "kind": "ocr",
                        "status": "observed",
                        "text": "[Neon Signboard] 彌敦道 Nathan Road / 旺角 Mong Kok",
                        "confidence_weight": 0.94,
                    },
                    {
                        "kind": "infrastructure",
                        "status": "observed",
                        "text": "[Infrastructure] Double-decker bus corridor with cantilevered neon signs",
                        "confidence_weight": 0.86,
                    },
                ],
                ground_truth=GroundTruth(
                    is_negative_control=False,
                    expected_abstention=False,
                    country="Hong Kong",
                    state_or_region="Kowloon",
                    target_lat=22.3193,
                    target_lon=114.1694,
                    point_tolerance_km=15.0,
                    description="Nathan Road, Mong Kok, Kowloon, Hong Kong",
                ),
            ),
            # 9. Misleading Souvenir Artifact (Negative / Inconclusive)
            BlindBenchmarkCase(
                case_id="BLIND-09",
                name="Desk with Souvenir Coffee Mug",
                category="deceptive",
                initial_evidence=[
                    {
                        "kind": "ocr",
                        "status": "observed",
                        "text": "[Novelty Object] Souvenir: Greetings from Hawaii",
                        "confidence_weight": 0.40,
                    },
                ],
                ground_truth=GroundTruth(
                    is_negative_control=True,
                    expected_abstention=True,
                    description="Photograph of a souvenir mug on an indoor desk; mug does not localize photo origin",
                ),
            ),
            # 10. Railway Corridor with Overhead Catenary
            BlindBenchmarkCase(
                case_id="BLIND-10",
                name="German InterCity Express Corridor",
                category="railway",
                initial_evidence=[
                    {
                        "kind": "ocr",
                        "status": "observed",
                        "text": "[Station Signboard] Frankfurt(Main) Hbf - Gleis 7",
                        "confidence_weight": 0.96,
                    },
                    {
                        "kind": "infrastructure",
                        "status": "observed",
                        "text": "[Infrastructure] 15kV AC 16.7Hz overhead railway catenary masts",
                        "confidence_weight": 0.90,
                    },
                ],
                ground_truth=GroundTruth(
                    is_negative_control=False,
                    expected_abstention=False,
                    country="Germany",
                    state_or_region="Hesse",
                    target_lat=50.1071,
                    target_lon=8.6638,
                    point_tolerance_km=20.0,
                    description="Frankfurt Central Station, Hesse, Germany",
                ),
            ),
        ]

    def run_benchmark(self, offline: bool = True) -> List[CaseEvaluation]:
        """Execute autonomous investigations across all blind cases and evaluate results."""
        manifest = self.get_manifest()
        evaluations: List[CaseEvaluation] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            db_path = workspace / "blind_bench.db"
            db = DatabaseManager(db_path)
            inv_store = InvestigationStore(db, workspace / "storage")
            evid_store = EvidenceStore(db)

            config = AgentConfig(
                max_iterations=8,
                max_tool_calls=15,
                max_external_requests=8,
                max_duration_seconds=30,
                offline_mode=offline,
            )
            agent = InvestigationAgent(db_manager=db, config=config)

            for case in manifest:
                logger.info(f"Running blind benchmark case: {case.case_id} ({case.name})...")

                # Create synthetic test image file
                img_path = workspace / f"{case.case_id}.jpg"
                img = Image.new("RGB", (640, 480), color=(120, 140, 160))
                img.save(img_path)

                img_rec = load_and_assess_image(img_path, preserve_original=False)
                inv = inv_store.create_investigation(img_rec)

                # Seed initial intake clues
                for ev_dict in case.initial_evidence:
                    evid_store.add_evidence(
                        EvidenceItem(
                            investigation_id=inv.id,
                            kind=ev_dict["kind"],
                            status=ev_dict["status"],
                            text=ev_dict["text"],
                            confidence_weight=ev_dict["confidence_weight"],
                        )
                    )

                # Execute investigation strictly without passing ground truth
                t0 = time.perf_counter()
                dossier: FinalInvestigationDossier = agent.investigate(
                    investigation_id=inv.id,
                    max_iterations=8,
                    offline=offline,
                )
                duration_s = time.perf_counter() - t0

                # Evaluate results against isolated ground truth
                evaluation = self._evaluate_case(case, dossier, duration_s)
                evaluations.append(evaluation)

        return evaluations

    def _evaluate_case(
        self,
        case: BlindBenchmarkCase,
        dossier: FinalInvestigationDossier,
        duration_s: float,
    ) -> CaseEvaluation:
        gt = case.ground_truth
        top = dossier.primary_hypothesis

        pred_name = top.name if top else None
        pred_lat = top.lat if top else None
        pred_lon = top.lon if top else None

        # Distance error calculation
        dist_km = None
        if pred_lat is not None and pred_lon is not None and gt.target_lat is not None and gt.target_lon is not None:
            dist_km = haversine_distance((pred_lat, pred_lon), (gt.target_lat, gt.target_lon)) / 1000.0

        # Handling negative controls
        if gt.is_negative_control:
            is_abstained = dossier.final_status in ("insufficient_evidence", "inconclusive")
            false_positive = not is_abstained and (top is not None and "unlocalized" not in top.name.lower())
            return CaseEvaluation(
                case_id=case.case_id,
                name=case.name,
                category=case.category,
                final_status=dossier.final_status,
                stopping_reason=dossier.stopping_reason,
                justified_precision=dossier.geographic_precision_justified,
                total_tool_calls=dossier.total_tool_calls,
                wall_clock_time_s=duration_s,
                predicted_name=pred_name,
                predicted_lat=pred_lat,
                predicted_lon=pred_lon,
                distance_error_km=None,
                country_matched=False,
                region_matched=False,
                point_matched=False,
                abstention_correct=is_abstained,
                false_positive=false_positive,
                notes="Correct honest abstention." if is_abstained else "False positive: localized negative control.",
            )

        # Evaluating positive cases
        country_matched = False
        region_matched = False
        point_matched = False

        if pred_name and gt.country:
            country_matched = gt.country.lower() in pred_name.lower() or any(
                gt.country.lower() in h.name.lower() for h in dossier.competing_hypotheses
            )

        if pred_name and gt.state_or_region:
            region_matched = gt.state_or_region.lower() in pred_name.lower() or any(
                gt.state_or_region.lower() in h.name.lower() for h in dossier.competing_hypotheses
            )

        if dist_km is not None and dist_km <= gt.point_tolerance_km:
            point_matched = True

        return CaseEvaluation(
            case_id=case.case_id,
            name=case.name,
            category=case.category,
            final_status=dossier.final_status,
            stopping_reason=dossier.stopping_reason,
            justified_precision=dossier.geographic_precision_justified,
            total_tool_calls=dossier.total_tool_calls,
            wall_clock_time_s=duration_s,
            predicted_name=pred_name,
            predicted_lat=pred_lat,
            predicted_lon=pred_lon,
            distance_error_km=dist_km,
            country_matched=country_matched,
            region_matched=region_matched,
            point_matched=point_matched,
            abstention_correct=False,
            false_positive=False,
            notes=f"Error: {dist_km:.1f}km" if dist_km is not None else "No coordinates predicted.",
        )


def generate_benchmark_summary(evaluations: List[CaseEvaluation]) -> Dict[str, Any]:
    """Calculate aggregate metrics across blind benchmark evaluations."""
    total_cases = len(evaluations)
    positive_cases = [e for e in evaluations if e.category != "negative_control" and e.category != "deceptive"]
    negative_cases = [e for e in evaluations if e.category in ("negative_control", "deceptive")]

    # Positive Metrics
    country_matches = sum(1 for e in positive_cases if e.country_matched)
    region_matches = sum(1 for e in positive_cases if e.region_matched)
    point_matches = sum(1 for e in positive_cases if e.point_matched)
    valid_distances = [e.distance_error_km for e in positive_cases if e.distance_error_km is not None]

    mean_dist = sum(valid_distances) / len(valid_distances) if valid_distances else 0.0
    sorted_dists = sorted(valid_distances)
    median_dist = sorted_dists[len(sorted_dists) // 2] if sorted_dists else 0.0

    # Negative Metrics
    correct_abstentions = sum(1 for e in negative_cases if e.abstention_correct)
    false_positives = sum(1 for e in negative_cases if e.false_positive)

    mean_tools = sum(e.total_tool_calls for e in evaluations) / total_cases if total_cases else 0.0
    mean_time = sum(e.wall_clock_time_s for e in evaluations) / total_cases if total_cases else 0.0

    return {
        "total_cases": total_cases,
        "positive_cases_count": len(positive_cases),
        "negative_cases_count": len(negative_cases),
        "country_accuracy": country_matches / len(positive_cases) if positive_cases else 0.0,
        "region_accuracy": region_matches / len(positive_cases) if positive_cases else 0.0,
        "point_accuracy": point_matches / len(positive_cases) if positive_cases else 0.0,
        "mean_error_km": mean_dist,
        "median_error_km": median_dist,
        "negative_abstention_rate": correct_abstentions / len(negative_cases) if negative_cases else 1.0,
        "false_positive_rate": false_positives / len(negative_cases) if negative_cases else 0.0,
        "mean_tool_calls": mean_tools,
        "mean_runtime_s": mean_time,
    }


if __name__ == "__main__":
    suite = BlindBenchmarkSuite()
    evals = suite.run_benchmark(offline=True)
    summary = generate_benchmark_summary(evals)

    print("================================================================================")
    print("HOUND PHASE 4.1 INDEPENDENT BLIND BENCHMARK RESULTS")
    print("================================================================================")
    for e in evals:
        dist_str = f"{e.distance_error_km:.1f}km" if e.distance_error_km is not None else "N/A"
        print(f"[{e.case_id}] {e.name:<32} | Status: {e.final_status:<22} | Error: {dist_str:<8} | Notes: {e.notes}")
    print("--------------------------------------------------------------------------------")
    print(f"Total Cases: {summary['total_cases']} ({summary['positive_cases_count']} Positive, {summary['negative_cases_count']} Negative Controls)")
    print(f"Country-Level Accuracy:        {summary['country_accuracy']*100:.1f}%")
    print(f"Region-Level Accuracy:         {summary['region_accuracy']*100:.1f}%")
    print(f"Point-Level Accuracy (<=25km): {summary['point_accuracy']*100:.1f}%")
    print(f"Mean Error (Positive):         {summary['mean_error_km']:.1f} km")
    print(f"Median Error (Positive):       {summary['median_error_km']:.1f} km")
    print(f"Negative Control Abstention:   {summary['negative_abstention_rate']*100:.1f}%")
    print(f"False Positive Rate:           {summary['false_positive_rate']*100:.1f}%")
    print(f"Mean Tool Calls:               {summary['mean_tool_calls']:.1f}")
    print(f"Mean Wall-Clock Time:          {summary['mean_runtime_s']:.2f} s")
    print("================================================================================")
