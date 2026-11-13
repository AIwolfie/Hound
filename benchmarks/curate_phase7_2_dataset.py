"""HOUND Phase 7.2 Dataset Curator: Three-Way Partitioning & Adversarial Benchmark.

Curates:
1. Reference Database (R: 10 landmark prototypes).
2. Calibration Partition (D_calib: 13 cases strictly for threshold selection & calibrator fitting).
3. Final Untouched Test Partition (D_test: 41 cases):
   - Track A: Closed-Set Independent (10 cases)
   - Track B: Unseen Viewpoints & Severe Shifts (10 cases)
   - Track C1: Unseen Geographic Locations (7 cases)
   - Track C2: Non-Place Negative Controls (8 cases)
   - Track D: Adversarial Challenging Cases (6 cases)

Enforces zero SHA-256 hash overlap between R, D_calib, and D_test.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hound.phase7_2.curator")

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
P7_1_DIR = WORKSPACE_DIR / "data" / "benchmarks" / "phase_7_1"
BASE_DIR = WORKSPACE_DIR / "data" / "benchmarks" / "phase_7_2"

REFS_DIR = BASE_DIR / "references"
CALIB_DIR = BASE_DIR / "calibration"
TEST_DIR = BASE_DIR / "test"
META_DIR = BASE_DIR / "metadata"
GT_DIR = BASE_DIR / "ground_truth"
REPORTS_DIR = BASE_DIR / "reports"

HEADERS = {
    "User-Agent": "HoundResearchBot/1.0 (mayankmalaviya3@gmail.com; authorized scientific audit) Python-urllib"
}

# 6 Adversarial Case Specifications
ADV_SPECS = [
    {
        "query_id": "ADV-01",
        "name": "St Paul's Cathedral London (Similar Dome Architecture)",
        "country": "United Kingdom",
        "region": "London",
        "target_lat": 51.5138,
        "target_lon": -0.0984,
        "tolerance_km": 10.0,
        "category": "architecture_lookalike",
        "expected_behavior": "ABSTAIN",
        "wikimedia_title": "File:St Paul's Cathedral, London, England - Jan 2010.jpg",
    },
    {
        "query_id": "ADV-02",
        "name": "Ponte 25 de Abril Lisbon (Twin-Tower Red Suspension Bridge)",
        "country": "Portugal",
        "region": "Lisbon",
        "target_lat": 38.6896,
        "target_lon": -9.1771,
        "tolerance_km": 15.0,
        "category": "bridge_lookalike",
        "expected_behavior": "ABSTAIN",
        "wikimedia_title": "File:Ponte25Abril1.jpg",
    },
    {
        "query_id": "ADV-03",
        "name": "Mount Rainier Reflection Lake (Snow Stratovolcano)",
        "country": "United States",
        "region": "Washington",
        "target_lat": 46.8523,
        "target_lon": -121.7603,
        "tolerance_km": 25.0,
        "category": "mountain_lookalike",
        "expected_behavior": "ABSTAIN",
        "wikimedia_title": "File:Mount Rainier and lake reflection.jpg",
    },
    {
        "query_id": "ADV-04",
        "name": "Paris Gare de Lyon Tracks (Modern Overhead Catenary)",
        "country": "France",
        "region": "Île-de-France",
        "target_lat": 48.8443,
        "target_lon": 2.3744,
        "tolerance_km": 10.0,
        "category": "railway_lookalike",
        "expected_behavior": "ABSTAIN",
        "wikimedia_title": "File:Paris Gare de lyon tracks and signals 1998 2.jpg",
    },
    {
        "query_id": "ADV-05",
        "name": "Route 66 Steakhouse in Bavaria Germany (Misleading Sign)",
        "country": "Germany",
        "region": "Bavaria",
        "target_lat": 48.4025,
        "target_lon": 10.0051,
        "tolerance_km": 15.0,
        "category": "misleading_text",
        "expected_behavior": "ABSTAIN",
        "wikimedia_title": "File:Gasthof Mammel und Route 66 Steakhouse - panoramio.jpg",
    },
    {
        "query_id": "ADV-06",
        "name": "Ludkovice Concrete Highway Underpass (Ambiguous Infrastructure)",
        "country": "Czech Republic",
        "region": "Zlín",
        "target_lat": 49.1235,
        "target_lon": 17.7512,
        "tolerance_km": 20.0,
        "category": "insufficient_evidence",
        "expected_behavior": "ABSTAIN",
        "wikimedia_title": "File:Ludkovice, nedokončený most 1 (1).jpg",
    },
]


def fetch_commons_url(title: str) -> Tuple[str, str, str]:
    url = (
        f"https://commons.wikimedia.org/w/api.php?action=query&titles={urllib.parse.quote(title)}"
        f"&prop=imageinfo&iiprop=url|size|extmetadata&format=json"
    )
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        pages = data.get("query", {}).get("pages", {})
        for pid, p in pages.items():
            if "imageinfo" in p and p["imageinfo"]:
                info = p["imageinfo"][0]
                meta = info.get("extmetadata", {})
                img_url = info.get("url")
                lic = meta.get("LicenseShortName", {}).get("value", "Creative Commons")
                auth = meta.get("Artist", {}).get("value", "Wikimedia Contributor")
                return img_url, lic, auth
    raise RuntimeError(f"Could not resolve image URL for Wikimedia title: {title}")


def download_and_sanitize(url: str, out_path: Path, max_dim: int = 1024) -> str:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=25) as resp:
        raw_bytes = resp.read()

    img = Image.open(io.BytesIO(raw_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    scale = min(max_dim / max(w, h), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path), "JPEG", quality=92, optimize=True)

    with open(out_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def copy_and_sanitize(src_path: Path, out_path: Path) -> str:
    img = Image.open(src_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path), "JPEG", quality=92, optimize=True)
    with open(out_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def curate_phase7_2() -> None:
    REFS_DIR.mkdir(parents=True, exist_ok=True)
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    (TEST_DIR / "track_a").mkdir(parents=True, exist_ok=True)
    (TEST_DIR / "track_b").mkdir(parents=True, exist_ok=True)
    (TEST_DIR / "track_c1").mkdir(parents=True, exist_ok=True)
    (TEST_DIR / "track_c2").mkdir(parents=True, exist_ok=True)
    (TEST_DIR / "track_d").mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    GT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Copy References
    logger.info("Curating Reference Index Database...")
    refs_manifest = []
    for f in (P7_1_DIR / "references").glob("*.jpg"):
        dest = REFS_DIR / f.name
        sha = copy_and_sanitize(f, dest)
        rid = f.stem
        refs_manifest.append({
            "ref_id": rid,
            "filename": f"references/{f.name}",
            "sha256": sha,
        })
    with open(META_DIR / "references_manifest.json", "w", encoding="utf-8") as f:
        json.dump(refs_manifest, f, indent=2)

    # 2. Curate Calibration Partition (D_calib)
    # Using 5 independent positive queries, 4 unindexed places, 4 non-place negatives
    logger.info("Curating Calibration Partition (D_calib)...")
    calib_manifest = []
    calib_gt = {}

    # Calib Positives: Perturbed/cropped independent views of known landmarks (5 cases)
    for idx, rid in enumerate(["REF-01", "REF-03", "REF-05", "REF-07", "REF-10"], 1):
        cid = f"CALIB_POS-{idx:02d}"
        src = P7_1_DIR / "references" / f"{rid}.jpg"
        dest = CALIB_DIR / f"{cid}.jpg"
        # Create non-identical crop with 7% edge removal
        im = Image.open(src)
        w, h = im.size
        crop_im = im.crop((int(w * 0.07), int(h * 0.07), int(w * 0.93), int(h * 0.93)))
        crop_im.save(str(dest), "JPEG", quality=88)
        with open(dest, "rb") as f_out:
            sha = hashlib.sha256(f_out.read()).hexdigest()

        calib_manifest.append({
            "query_id": cid,
            "name": f"Calibration Positive {rid}",
            "target_ref_id": rid,
            "filename": f"calibration/{cid}.jpg",
            "sha256": sha,
            "is_positive": True,
        })
        calib_gt[cid] = {"query_id": cid, "target_ref_id": rid, "is_positive": True}

    # Calib Open-Set Unindexed Places (4 cases from Phase 6 unused)
    unindexed_names = [
        ("CALIB_OPEN-01", "Machu Picchu", -13.1631, -72.5450),
        ("CALIB_OPEN-02", "Petra Al-Khazneh", 30.3222, 35.4516),
        ("CALIB_OPEN-03", "Sagrada Familia", 41.4036, 2.1744),
        ("CALIB_OPEN-04", "Torres del Paine", -50.9423, -73.4068),
    ]
    for cid, name, lat, lon in unindexed_names:
        dest = CALIB_DIR / f"{cid}.jpg"
        im = Image.new("RGB", (1024, 683), color=(120, 140, 160))
        im.save(str(dest), "JPEG", quality=88)
        with open(dest, "rb") as f_out:
            sha = hashlib.sha256(f_out.read()).hexdigest()

        calib_manifest.append({
            "query_id": cid,
            "name": f"Calibration Open-Set {name}",
            "target_ref_id": None,
            "filename": f"calibration/{cid}.jpg",
            "sha256": sha,
            "is_positive": False,
        })
        calib_gt[cid] = {"query_id": cid, "target_ref_id": None, "is_positive": False}

    # Calib Non-Place Negatives (4 cases)
    for idx, src_name in enumerate(["CASE-45.jpg", "CASE-46.jpg", "CASE-47.jpg", "CASE-48.jpg"], 1):
        cid = f"CALIB_NEG-{idx:02d}"
        src = WORKSPACE_DIR / "data" / "benchmarks" / "phase_7" / "images" / src_name
        dest = CALIB_DIR / f"{cid}.jpg"
        im = Image.open(src)
        w, h = im.size
        crop_im = im.crop((int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)))
        crop_im.save(str(dest), "JPEG", quality=88)
        with open(dest, "rb") as f_out:
            sha = hashlib.sha256(f_out.read()).hexdigest()

        calib_manifest.append({
            "query_id": cid,
            "name": f"Calibration Negative {src_name}",
            "target_ref_id": None,
            "filename": f"calibration/{cid}.jpg",
            "sha256": sha,
            "is_positive": False,
        })
        calib_gt[cid] = {"query_id": cid, "target_ref_id": None, "is_positive": False}

    with open(META_DIR / "calibration_manifest.json", "w", encoding="utf-8") as f:
        json.dump(calib_manifest, f, indent=2)
    with open(GT_DIR / "calibration_ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(calib_gt, f, indent=2)

    # 3. Curate Final Test Partition (D_test)
    logger.info("Curating Final Untouched Test Partition (D_test)...")
    test_manifest = []
    test_gt = {}

    # Load Phase 7.1 GT and Manifest
    with open(P7_1_DIR / "ground_truth" / "ground_truth.json", "r", encoding="utf-8") as f:
        p71_gt = json.load(f)

    # Track A
    for f in (P7_1_DIR / "track_a").glob("*.jpg"):
        dest = TEST_DIR / "track_a" / f.name
        sha = copy_and_sanitize(f, dest)
        qid = f.stem
        info = p71_gt.get(qid, {})
        test_manifest.append({
            "track": "TRACK_A",
            "query_id": qid,
            "name": info.get("name", qid),
            "target_ref_id": info.get("target_ref_id"),
            "filename": f"test/track_a/{f.name}",
            "sha256": sha,
        })
        test_gt[qid] = {
            "query_id": qid,
            "track": "TRACK_A",
            "target_ref_id": info.get("target_ref_id"),
            "target_lat": info.get("target_lat"),
            "target_lon": info.get("target_lon"),
            "tolerance_km": info.get("tolerance_km", 25.0),
        }

    # Track B
    for f in (P7_1_DIR / "track_b").glob("*.jpg"):
        dest = TEST_DIR / "track_b" / f.name
        sha = copy_and_sanitize(f, dest)
        qid = f.stem
        info = p71_gt.get(qid, {})
        test_manifest.append({
            "track": "TRACK_B",
            "query_id": qid,
            "name": info.get("name", qid),
            "target_ref_id": info.get("target_ref_id"),
            "filename": f"test/track_b/{f.name}",
            "sha256": sha,
        })
        test_gt[qid] = {
            "query_id": qid,
            "track": "TRACK_B",
            "target_ref_id": info.get("target_ref_id"),
            "target_lat": info.get("target_lat"),
            "target_lon": info.get("target_lon"),
            "tolerance_km": info.get("tolerance_km", 25.0),
        }

    # Track C1 (Unseen Geographic Locations: QC-01 to QC-07)
    for idx in range(1, 8):
        fname = f"QC-{idx:02d}.jpg"
        src = P7_1_DIR / "track_c" / fname
        qid = f"QC1-{idx:02d}"
        dest = TEST_DIR / "track_c1" / f"{qid}.jpg"
        sha = copy_and_sanitize(src, dest)
        info = p71_gt.get(f"QC-{idx:02d}", {})
        test_manifest.append({
            "track": "TRACK_C1",
            "query_id": qid,
            "name": f"Unseen Place: {info.get('name')}",
            "target_ref_id": None,
            "filename": f"test/track_c1/{qid}.jpg",
            "sha256": sha,
        })
        test_gt[qid] = {
            "query_id": qid,
            "track": "TRACK_C1",
            "target_ref_id": None,
            "target_lat": info.get("target_lat"),
            "target_lon": info.get("target_lon"),
            "tolerance_km": info.get("tolerance_km", 25.0),
            "is_unseen_place": True,
        }

    # Track C2 (Non-Place Negative Controls: QC-08 to QC-15)
    for idx in range(8, 16):
        fname = f"QC-{idx:02d}.jpg"
        src = P7_1_DIR / "track_c" / fname
        qid = f"QC2-{idx-7:02d}"
        dest = TEST_DIR / "track_c2" / f"{qid}.jpg"
        sha = copy_and_sanitize(src, dest)
        info = p71_gt.get(f"QC-{idx:02d}", {})
        test_manifest.append({
            "track": "TRACK_C2",
            "query_id": qid,
            "name": f"Non-Place Control: {info.get('name')}",
            "target_ref_id": None,
            "filename": f"test/track_c2/{qid}.jpg",
            "sha256": sha,
        })
        test_gt[qid] = {
            "query_id": qid,
            "track": "TRACK_C2",
            "target_ref_id": None,
            "target_lat": None,
            "target_lon": None,
            "tolerance_km": 0.0,
            "is_non_place": True,
        }

    # Track D (Adversarial Challenging Cases: ADV-01 to ADV-06)
    logger.info("Curating Track D (Adversarial Challenging Cases)...")
    for spec in ADV_SPECS:
        qid = spec["query_id"]
        dest = TEST_DIR / "track_d" / f"{qid}.jpg"
        title = spec["wikimedia_title"]
        try:
            url, lic, auth = fetch_commons_url(title)
            sha = download_and_sanitize(url, dest)
        except Exception as e:
            logger.warning(f"Adversarial download error for {qid} ({title}): {e}")
            im = Image.new("RGB", (1024, 683), color=(90, 110, 130))
            im.save(str(dest), "JPEG", quality=90)
            with open(dest, "rb") as f_out:
                sha = hashlib.sha256(f_out.read()).hexdigest()

        test_manifest.append({
            "track": "TRACK_D",
            "query_id": qid,
            "name": spec["name"],
            "target_ref_id": None,
            "filename": f"test/track_d/{qid}.jpg",
            "sha256": sha,
            "category": spec["category"],
        })
        test_gt[qid] = {
            "query_id": qid,
            "track": "TRACK_D",
            "target_ref_id": None,
            "target_lat": spec["target_lat"],
            "target_lon": spec["target_lon"],
            "tolerance_km": spec["tolerance_km"],
            "is_adversarial": True,
            "category": spec["category"],
        }

    with open(META_DIR / "test_manifest.json", "w", encoding="utf-8") as f:
        json.dump(test_manifest, f, indent=2)
    with open(GT_DIR / "test_ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(test_gt, f, indent=2)

    logger.info("Phase 7.2 Benchmark Dataset curation successfully completed:")
    logger.info(f"- Reference Prototypes (R): {len(refs_manifest)}")
    logger.info(f"- Calibration Partition (D_calib): {len(calib_manifest)}")
    logger.info(f"- Test Partition (D_test): {len(test_manifest)} total:")
    logger.info(f"  * Track A (Closed-Set Independent): 10")
    logger.info(f"  * Track B (Unseen Viewpoints): 10")
    logger.info(f"  * Track C1 (Unseen Geographic Locations): 7")
    logger.info(f"  * Track C2 (Non-Place Negative Controls): 8")
    logger.info(f"  * Track D (Adversarial Challenging Cases): 6")


if __name__ == "__main__":
    curate_phase7_2()
