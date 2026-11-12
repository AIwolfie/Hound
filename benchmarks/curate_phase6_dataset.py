"""HOUND Phase 6 Real-Photograph Dataset Curator.

Fetches legally usable, documented Creative Commons real-world photographs
from Wikimedia Commons for Phase 6 evaluation, strips metadata to prevent
EXIF leakage, standardizes dimensions, and computes cryptographic hashes.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hound.phase6.curator")

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
BASE_BENCHMARK_DIR = WORKSPACE_DIR / "data" / "benchmarks" / "phase_6"
IMAGES_DIR = BASE_BENCHMARK_DIR / "images"
GT_DIR = BASE_BENCHMARK_DIR / "ground_truth"
REFS_DIR = BASE_BENCHMARK_DIR / "references"
META_DIR = BASE_BENCHMARK_DIR / "metadata"
REPORTS_DIR = BASE_BENCHMARK_DIR / "reports"

HEADERS = {
    "User-Agent": "HoundResearchBot/1.0 (mayankmalaviya3@gmail.com; authorized scientific evaluation) Python-urllib"
}

CASES = [
    {
        "case_id": "REAL-01",
        "file_title": "File:Tour Eiffel Wikimedia Commons (cropped).jpg",
        "name": "Eiffel Tower Champ de Mars",
        "category": "urban_landmark",
        "track": "closed_set",
        "is_negative_control": False,
        "country": "France",
        "region": "Île-de-France",
        "target_lat": 48.8584,
        "target_lon": 2.2945,
        "tolerance_km": 15.0,
        "expected_scale": "landmark",
    },
    {
        "case_id": "REAL-02",
        "file_title": "File:Matterhorn from Domhütte - 2.jpg",
        "name": "Matterhorn Alpine Summit",
        "category": "mountain_skyline",
        "track": "closed_set",
        "is_negative_control": False,
        "country": "Switzerland",
        "region": "Valais",
        "target_lat": 45.9765,
        "target_lon": 7.7491,
        "tolerance_km": 25.0,
        "expected_scale": "landmark",
    },
    {
        "case_id": "REAL-03",
        "file_title": "File:Sydney Opera House - Dec 2008.jpg",
        "name": "Sydney Opera House Harbour",
        "category": "coastal_landmark",
        "track": "closed_set",
        "is_negative_control": False,
        "country": "Australia",
        "region": "New South Wales",
        "target_lat": -33.8568,
        "target_lon": 151.2153,
        "tolerance_km": 15.0,
        "expected_scale": "landmark",
    },
    {
        "case_id": "REAL-04",
        "file_title": "File:Tokyo Shibuya Scramble Crossing 2018-10-09.jpg",
        "name": "Shibuya Scramble Crossing",
        "category": "urban_crossing",
        "track": "closed_set",
        "is_negative_control": False,
        "country": "Japan",
        "region": "Tokyo",
        "target_lat": 35.6595,
        "target_lon": 139.7005,
        "tolerance_km": 15.0,
        "expected_scale": "district",
    },
    {
        "case_id": "REAL-05",
        "file_title": "File:Berlin Hauptbahnhof 2016.jpg",
        "name": "Berlin Central Station Tracks",
        "category": "railway_infrastructure",
        "track": "closed_set",
        "is_negative_control": False,
        "country": "Germany",
        "region": "Berlin",
        "target_lat": 52.5251,
        "target_lon": 13.3694,
        "tolerance_km": 20.0,
        "expected_scale": "landmark",
    },
    {
        "case_id": "REAL-06",
        "file_title": "File:Seligman.- Snow car, sur la Route 66.jpg",
        "name": "Historic Route 66 Seligman Arizona",
        "category": "highway_corridor",
        "track": "closed_set",
        "is_negative_control": False,
        "country": "United States",
        "region": "Arizona",
        "target_lat": 35.3256,
        "target_lon": -112.8744,
        "tolerance_km": 40.0,
        "expected_scale": "road_corridor",
    },
    {
        "case_id": "REAL-07",
        "file_title": "File:Bixby Creek Bridge May 2011 panorama.jpg",
        "name": "Bixby Creek Bridge Big Sur",
        "category": "coastal_highway",
        "track": "closed_set",
        "is_negative_control": False,
        "country": "United States",
        "region": "California",
        "target_lat": 36.3714,
        "target_lon": -121.9018,
        "tolerance_km": 30.0,
        "expected_scale": "road_corridor",
    },
    {
        "case_id": "REAL-08",
        "file_title": "File:Tuscan landscape with lonely tree.jpg",
        "name": "Val d'Orcia Rolling Farmland",
        "category": "rural_farmland",
        "track": "closed_set",
        "is_negative_control": False,
        "country": "Italy",
        "region": "Tuscany",
        "target_lat": 43.0560,
        "target_lon": 11.6020,
        "tolerance_km": 35.0,
        "expected_scale": "region",
    },
    {
        "case_id": "REAL-09",
        "file_title": "File:Zabriskie Point in Death Valley.jpg",
        "name": "Zabriskie Point Death Valley",
        "category": "desert_landscape",
        "track": "closed_set",
        "is_negative_control": False,
        "country": "United States",
        "region": "California",
        "target_lat": 36.4208,
        "target_lon": -116.8122,
        "tolerance_km": 35.0,
        "expected_scale": "region",
    },
    {
        "case_id": "REAL-10",
        "file_title": "File:Puerto, Tromsø, Noruega, 2019-09-04, DD 48.jpg",
        "name": "Tromso Arctic Port & Mountains",
        "category": "open_set_coastal_city",
        "track": "open_set",
        "is_negative_control": False,
        "country": "Norway",
        "region": "Troms",
        "target_lat": 69.6492,
        "target_lon": 18.9553,
        "tolerance_km": 25.0,
        "expected_scale": "city",
    },
    {
        "case_id": "REAL-11",
        "file_title": "File:A meeting room at Government Office of Vietnam.jpg",
        "name": "Windowless Corporate Meeting Room",
        "category": "negative_control",
        "track": "open_set",
        "is_negative_control": True,
        "country": None,
        "region": None,
        "target_lat": None,
        "target_lon": None,
        "tolerance_km": 0.0,
        "expected_scale": "unlocalized",
    },
    {
        "case_id": "REAL-12",
        "file_title": "File:Erg Chebbi sunset.jpg",
        "name": "Featureless Desert Sand Ripples",
        "category": "negative_control",
        "track": "open_set",
        "is_negative_control": True,
        "country": None,
        "region": None,
        "target_lat": None,
        "target_lon": None,
        "tolerance_km": 0.0,
        "expected_scale": "unlocalized",
    },
]


def fetch_file_metadata(title: str) -> Dict[str, Any]:
    url = (
        f"https://commons.wikimedia.org/w/api.php?action=query&titles={urllib.parse.quote(title)}"
        f"&prop=imageinfo&iiprop=url|size|sha1|extmetadata&format=json"
    )
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode())
        pages = data.get("query", {}).get("pages", {})
        for pid, p in pages.items():
            if "imageinfo" in p and p["imageinfo"]:
                info = p["imageinfo"][0]
                meta = info.get("extmetadata", {})
                return {
                    "url": info.get("url"),
                    "orig_width": info.get("width"),
                    "orig_height": info.get("height"),
                    "license": meta.get("LicenseShortName", {}).get("value", "Creative Commons"),
                    "artist": meta.get("Artist", {}).get("value", "Wikimedia Commons Contributor"),
                    "credit": meta.get("Credit", {}).get("value", "Wikimedia Commons"),
                }
    raise RuntimeError(f"Could not fetch metadata for {title}")


def download_and_sanitize_image(url: str, out_path: Path, max_dim: int = 1024) -> Tuple[int, int, str]:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=25) as resp:
        raw_bytes = resp.read()

    import io
    img = Image.open(io.BytesIO(raw_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    scale = min(max_dim / max(w, h), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
    else:
        new_w, new_h = w, h

    # Save clean JPEG with zero EXIF or camera metadata (strictly stripped)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path), "JPEG", quality=90, optimize=True)

    # Compute SHA-256 of the sanitized, stripped image file
    clean_bytes = out_path.read_bytes()
    sha256 = hashlib.sha256(clean_bytes).hexdigest()
    return new_w, new_h, sha256


def curate_dataset() -> None:
    for d in [IMAGES_DIR, GT_DIR, REFS_DIR, META_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    manifest_records = []
    ground_truth_records = {}

    for case in CASES:
        cid = case["case_id"]
        title = case["file_title"]
        logger.info(f"Processing {cid}: {case['name']} ({title})...")

        meta = fetch_file_metadata(title)
        img_url = meta["url"]
        target_path = IMAGES_DIR / f"{cid}.jpg"

        width, height, sha256 = download_and_sanitize_image(img_url, target_path)

        manifest_item = {
            "case_id": cid,
            "name": case["name"],
            "category": case["category"],
            "track": case["track"],
            "is_negative_control": case["is_negative_control"],
            "source_title": title,
            "source_url": img_url,
            "license": meta["license"],
            "author": meta["artist"],
            "sanitized_file": f"images/{cid}.jpg",
            "resolution": f"{width}x{height}",
            "sha256": sha256,
            "metadata_stripped": True,
        }
        manifest_records.append(manifest_item)

        gt_item = {
            "case_id": cid,
            "name": case["name"],
            "is_negative_control": case["is_negative_control"],
            "country": case["country"],
            "region": case["region"],
            "target_lat": case["target_lat"],
            "target_lon": case["target_lon"],
            "tolerance_km": case["tolerance_km"],
            "expected_scale": case["expected_scale"],
        }
        ground_truth_records[cid] = gt_item

        logger.info(f"  [OK] Saved {cid}.jpg ({width}x{height}) SHA256: {sha256[:16]}... License: {meta['license']}")

    # Save manifest.json
    manifest_file = META_DIR / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, indent=2)
    logger.info(f"Wrote manifest: {manifest_file}")

    # Save ground_truth.json (ISOLATED from inference pipeline)
    gt_file = GT_DIR / "ground_truth.json"
    with open(gt_file, "w", encoding="utf-8") as f:
        json.dump(ground_truth_records, f, indent=2)
    logger.info(f"Wrote ground truth: {gt_file}")


if __name__ == "__main__":
    curate_dataset()
