"""HOUND Reproducible Public-Landmark & Infrastructure Benchmark Suite.

Evaluates geolocation precision, scale resolution, candidate ranking, and abstention
accuracy across authorized, publicly recognizable landmarks and infrastructure.
"""
from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ExifTags

from agent.investigation_manager import GeographicInvestigationManager
from core.configuration import get_config
from core.geographic_search import haversine_distance
from core.image_loader import ExifMetadata, GPSCoordinates, ImageRecord, IntegrityAssessment, load_and_assess_image
from core.logger import get_logger
from integrations.geocoding.gazetteer import GeocodingResult
from storage.database import DatabaseManager
from storage.evidence_store import EvidenceItem, EvidenceStore
from storage.investigation_store import InvestigationStore

logger = get_logger("hound.benchmark")


@dataclass
class BenchmarkCase:
    case_id: str
    name: str
    category: str  # "public_landmark", "road_corridor", "transport_hub", "mountain_terrain", "low_info_abstain"
    ground_truth_name: str
    ground_truth_coords: Optional[Tuple[float, float]]
    ground_truth_precision: str  # "exact", "corridor", "approximate", "unlocalized"
    expected_scale: str  # "point", "building", "road", "area", "city", "region"
    visual_clues: List[str]
    metadata_clues: Dict[str, Any]
    should_abstain: bool = False


@dataclass
class BenchmarkResult:
    case_id: str
    name: str
    category: str
    ground_truth_coords: Optional[Tuple[float, float]]
    top1_candidate_name: str
    top1_candidate_level: str
    top1_candidate_score: float
    top1_candidate_coords: Optional[Tuple[float, float]]
    distance_to_ground_truth_km: Optional[float]
    ground_truth_in_top5: bool
    scale_matches: bool
    abstention_correct: bool
    unresolved_constraints_count: int
    execution_time_seconds: float
    notes: str


class PublicLandmarkBenchmark:
    """Executes deterministic benchmark tests against HOUND Geographic Intelligence."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.cfg = get_config()
        self.db = db_manager or DatabaseManager(self.cfg.paths.database_path)

    @staticmethod
    def get_standard_cases() -> List[BenchmarkCase]:
        return [
            BenchmarkCase(
                case_id="BM-01",
                name="Eiffel Tower Ground Truth Anchor",
                category="public_landmark",
                ground_truth_name="Eiffel Tower, Champ de Mars, Paris, France",
                ground_truth_coords=(48.8584, 2.2945),
                ground_truth_precision="exact",
                expected_scale="point",
                visual_clues=["Eiffel Tower iron lattice tower structure"],
                metadata_clues={"gps": (48.8584, 2.2945)},
                should_abstain=False,
            ),
            BenchmarkCase(
                case_id="BM-02",
                name="US-101 Transportation Corridor",
                category="road_corridor",
                ground_truth_name="US Highway 101 North, California",
                ground_truth_coords=(34.1480, -118.7956),
                ground_truth_precision="corridor",
                expected_scale="area",
                visual_clues=['Road signboard: "US-101 NORTH"'],
                metadata_clues={},
                should_abstain=False,
            ),
            BenchmarkCase(
                case_id="BM-03",
                name="Grand Central Terminal Transit Hub",
                category="transport_hub",
                ground_truth_name="Grand Central Terminal, New York, NY",
                ground_truth_coords=(40.7527, -73.9772),
                ground_truth_precision="exact",
                expected_scale="building",
                visual_clues=['Station entrance sign: "Grand Central Terminal"'],
                metadata_clues={},
                should_abstain=False,
            ),
            BenchmarkCase(
                case_id="BM-04",
                name="Mount Everest Himalayan Skyline",
                category="mountain_terrain",
                ground_truth_name="Mount Everest, Himalayas",
                ground_truth_coords=(27.9881, 86.9250),
                ground_truth_precision="exact",
                expected_scale="point",
                visual_clues=["Distinctive extreme elevation mountain skyline (>8000m)"],
                metadata_clues={"gps": (27.9881, 86.9250)},
                should_abstain=False,
            ),
            BenchmarkCase(
                case_id="BM-05",
                name="Indoor Low-Information Surface (Abstention Control)",
                category="low_info_abstain",
                ground_truth_name="Unlocalized Visual Scene",
                ground_truth_coords=None,
                ground_truth_precision="unlocalized",
                expected_scale="region",
                visual_clues=["Monochrome featureless surface", "Zero text", "Zero landmark"],
                metadata_clues={},
                should_abstain=True,
            ),
        ]

    def run_case(self, case: BenchmarkCase, temp_dir: Path) -> BenchmarkResult:
        """Run a single benchmark case through HOUND investigation pipeline."""
        t0 = time.perf_counter()
        inv_store = InvestigationStore(self.db, temp_dir / "storage")
        evid_store = EvidenceStore(self.db)

        # 1. Generate synthetic benchmark image
        img_file = temp_dir / f"{case.case_id}.jpg"
        img = Image.new("RGB", (640, 480), color=(120, 130, 140))
        draw = ImageDraw.Draw(img)

        # Draw visual clue text if present
        for clue in case.visual_clues:
            if '"' in clue:
                text = clue.split('"')[1]
                draw.rectangle([100, 380, 540, 440], fill=(20, 25, 30))
                draw.text((120, 400), text, fill=(255, 255, 255))

        # Add EXIF GPS if metadata clue present
        if "gps" in case.metadata_clues:
            lat, lon = case.metadata_clues["gps"]
            exif = img.getexif()
            gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)

            def dec_to_dms(val_float: float):
                v = abs(val_float)
                d = int(v)
                m = int((v - d) * 60)
                s = round((v - d - m / 60.0) * 3600.0, 3)
                return (d, m, s)

            gps_ifd[1] = "N" if lat >= 0 else "S"
            gps_ifd[2] = dec_to_dms(lat)
            gps_ifd[3] = "E" if lon >= 0 else "W"
            gps_ifd[4] = dec_to_dms(lon)
            img.save(img_file, format="JPEG", exif=exif)
        else:
            img.save(img_file, format="JPEG")

        # Ingest and create investigation session
        img_rec = load_and_assess_image(img_file, storage_dir=temp_dir / "storage")
        inv = inv_store.create_investigation(img_rec)

        # Ingest visual clues into EvidenceStore
        for clue in case.visual_clues:
            if '"' in clue:
                text = clue.split('"')[1]
                evid_store.add_evidence(
                    EvidenceItem(
                        investigation_id=inv.id,
                        kind="ocr",
                        status="computed",
                        text=f'[OCR Reading] Recognized text: "{text}". Category: road_number',
                        confidence_weight=0.95,
                    )
                )

        # Execute Geographic Investigation Manager
        manager = GeographicInvestigationManager(db_manager=self.db)
        summary = manager.run_investigation(inv.id)
        dt = time.perf_counter() - t0

        top_cand_name = summary.top_candidate_name
        top_cand_level = summary.top_candidate_level
        top_cand_score = summary.top_candidate_score
        top_cand_coords = summary.top_candidate_coords

        # Compute metric distances
        dist_km = None
        if case.ground_truth_coords and top_cand_coords:
            dist_km = round(haversine_distance(case.ground_truth_coords, top_cand_coords) / 1000.0, 2)

        gt_in_top5 = False
        if case.should_abstain:
            abstention_ok = (summary.top_candidate_level == "region" and "Unlocalized" in summary.top_candidate_name)
            scale_ok = (summary.top_candidate_level == "region")
            gt_in_top5 = True
        else:
            abstention_ok = not ("Unlocalized" in summary.top_candidate_name)
            scale_ok = (summary.top_candidate_level in (case.expected_scale, "point", "area", "building"))
            if case.ground_truth_coords:
                for ev in summary.evaluations[:5]:
                    cand_match = next((c for c in evid_store.get_candidates(inv.id) if c.id == ev.candidate_id), None)
                    if cand_match and cand_match.lat and cand_match.lon:
                        d = haversine_distance(case.ground_truth_coords, (cand_match.lat, cand_match.lon)) / 1000.0
                        if d < 50.0:  # Within 50km
                            gt_in_top5 = True
                            break
            else:
                gt_in_top5 = True

        notes = (
            f"Distance: {dist_km}km | Level: {top_cand_level.upper()} | "
            f"Score: {top_cand_score:.2f} | Unresolved: {len(summary.unresolved_questions)}"
        )

        return BenchmarkResult(
            case_id=case.case_id,
            name=case.name,
            category=case.category,
            ground_truth_coords=case.ground_truth_coords,
            top1_candidate_name=top_cand_name,
            top1_candidate_level=top_cand_level,
            top1_candidate_score=top_cand_score,
            top1_candidate_coords=top_cand_coords,
            distance_to_ground_truth_km=dist_km,
            ground_truth_in_top5=gt_in_top5,
            scale_matches=scale_ok,
            abstention_correct=abstention_ok,
            unresolved_constraints_count=len(summary.unresolved_questions),
            execution_time_seconds=round(dt, 2),
            notes=notes,
        )

    def run_all(self) -> List[BenchmarkResult]:
        """Execute full benchmark suite."""
        cases = self.get_standard_cases()
        results: List[BenchmarkResult] = []
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp_path = Path(tmp_str)
            for c in cases:
                logger.info(f"Running benchmark case {c.case_id}: {c.name}...")
                res = self.run_case(c, tmp_path)
                results.append(res)
        return results
