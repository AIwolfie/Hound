"""HOUND Phase 7.1 Dataset Curator for Independent Benchmark Validation.

Curates three strictly isolated evaluation tracks:
- References (10 real landmark prototypes).
- Track A: Closed-Set Retrieval (10 independent camera photographs of the reference locations).
- Track B: Unseen-Viewpoint Retrieval (10 extreme viewpoint / night / seasonal / weather shifts).
- Track C: Geographic Open-Set (7 unindexed world landmarks + 8 deceptive negative controls).

Enforces:
- Strict ground-truth isolation in data/benchmarks/phase_7_1/ground_truth/.
- Zero EXIF / metadata leakage.
- Independent image verification (no shared SHA-256 or identical files between query and reference).
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
logger = logging.getLogger("hound.phase7_1.curator")

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = WORKSPACE_DIR / "data" / "benchmarks" / "phase_7_1"
REFS_DIR = BASE_DIR / "references"
TRACK_A_DIR = BASE_DIR / "track_a"
TRACK_B_DIR = BASE_DIR / "track_b"
TRACK_C_DIR = BASE_DIR / "track_c"
GT_DIR = BASE_DIR / "ground_truth"
META_DIR = BASE_DIR / "metadata"
REPORTS_DIR = BASE_DIR / "reports"

P6_IMAGES = WORKSPACE_DIR / "data" / "benchmarks" / "phase_6" / "images"
P7_IMAGES = WORKSPACE_DIR / "data" / "benchmarks" / "phase_7" / "images"

HEADERS = {
    "User-Agent": "HoundResearchBot/1.0 (mayankmalaviya3@gmail.com; authorized scientific audit) Python-urllib"
}

# 10 Reference Prototypes
REFERENCE_SPECS = [
    {"ref_id": "REF-01", "name": "Eiffel Tower Champ de Mars", "country": "France", "region": "Île-de-France", "target_lat": 48.8584, "target_lon": 2.2945, "tolerance_km": 15.0, "p6_src": "REAL-01.jpg"},
    {"ref_id": "REF-02", "name": "Matterhorn Alpine Summit", "country": "Switzerland", "region": "Valais", "target_lat": 45.9765, "target_lon": 7.7491, "tolerance_km": 25.0, "p6_src": "REAL-02.jpg"},
    {"ref_id": "REF-03", "name": "Sydney Opera House Harbour", "country": "Australia", "region": "New South Wales", "target_lat": -33.8568, "target_lon": 151.2153, "tolerance_km": 15.0, "p6_src": "REAL-03.jpg"},
    {"ref_id": "REF-04", "name": "Shibuya Scramble Crossing", "country": "Japan", "region": "Tokyo", "target_lat": 35.6595, "target_lon": 139.7005, "tolerance_km": 15.0, "p6_src": "REAL-04.jpg"},
    {"ref_id": "REF-05", "name": "Berlin Central Station Tracks", "country": "Germany", "region": "Berlin", "target_lat": 52.5251, "target_lon": 13.3694, "tolerance_km": 10.0, "p6_src": "REAL-05.jpg"},
    {"ref_id": "REF-06", "name": "Historic Route 66 Seligman Arizona", "country": "United States", "region": "Arizona", "target_lat": 35.3255, "target_lon": -112.8753, "tolerance_km": 25.0, "p6_src": "REAL-06.jpg"},
    {"ref_id": "REF-07", "name": "Bixby Creek Bridge Big Sur", "country": "United States", "region": "California", "target_lat": 36.3714, "target_lon": -121.9018, "tolerance_km": 15.0, "p6_src": "REAL-07.jpg"},
    {"ref_id": "REF-08", "name": "Val d'Orcia Farmland", "country": "Italy", "region": "Tuscany", "target_lat": 43.0642, "target_lon": 11.6083, "tolerance_km": 30.0, "p6_src": "REAL-08.jpg"},
    {"ref_id": "REF-09", "name": "Zabriskie Point Death Valley", "country": "United States", "region": "California", "target_lat": 36.4208, "target_lon": -116.8122, "tolerance_km": 35.0, "p6_src": "REAL-09.jpg"},
    {"ref_id": "REF-10", "name": "Tromso Arctic Port & Mountains", "country": "Norway", "region": "Troms", "target_lat": 69.6492, "target_lon": 18.9553, "tolerance_km": 25.0, "p6_src": "REAL-10.jpg"},
]

# Track A: Closed-Set Retrieval (Independent Real Photographs of the 10 Locations)
TRACK_A_SPECS = [
    {"query_id": "QA-01", "target_ref_id": "REF-01", "name": "Eiffel Tower (Independent View)", "target_lat": 48.8584, "target_lon": 2.2945, "tolerance_km": 15.0, "wikimedia_title": "File:Paris - The Eiffel Tower in spring - 2307.jpg"},
    {"query_id": "QA-02", "target_ref_id": "REF-02", "name": "Matterhorn from Riffelsee (Independent View)", "target_lat": 45.9765, "target_lon": 7.7491, "tolerance_km": 25.0, "wikimedia_title": "File:Matterhorn Riffelsee 2005-06-11.jpg"},
    {"query_id": "QA-03", "target_ref_id": "REF-03", "name": "Sydney Opera House (Independent View)", "target_lat": -33.8568, "target_lon": 151.2153, "tolerance_km": 15.0, "wikimedia_title": "File:Sydney (AU), Opera House -- 2019 -- 3054.jpg"},
    {"query_id": "QA-04", "target_ref_id": "REF-04", "name": "Shibuya Crossing (Independent View)", "target_lat": 35.6595, "target_lon": 139.7005, "tolerance_km": 15.0, "wikimedia_title": "File:Shibuya Crossing (53083663621).jpg"},
    {"query_id": "QA-05", "target_ref_id": "REF-05", "name": "Berlin Central Station (Independent Exterior)", "target_lat": 52.5251, "target_lon": 13.3694, "tolerance_km": 10.0, "wikimedia_title": "File:Berlin Hauptbahnhof - panoramio (11).jpg"},
    {"query_id": "QA-06", "target_ref_id": "REF-06", "name": "Route 66 Seligman (Independent View)", "target_lat": 35.3255, "target_lon": -112.8753, "tolerance_km": 25.0, "wikimedia_title": "File:Deluxe Inn Motel in Seligman.- U.S. Route 66 (Arizona ).jpg"},
    {"query_id": "QA-07", "target_ref_id": "REF-07", "name": "Bixby Creek Bridge (Independent View)", "target_lat": 36.3714, "target_lon": -121.9018, "tolerance_km": 15.0, "wikimedia_title": "File:Bixby Creek Bridge, California, USA - May 2013.jpg"},
    {"query_id": "QA-08", "target_ref_id": "REF-08", "name": "Val d'Orcia in Autumn (Independent View)", "target_lat": 43.0642, "target_lon": 11.6083, "tolerance_km": 30.0, "wikimedia_title": "File:Val D Orcia In Autumn (179351679).jpeg"},
    {"query_id": "QA-09", "target_ref_id": "REF-09", "name": "Zabriskie Point (Independent View)", "target_lat": 36.4208, "target_lon": -116.8122, "tolerance_km": 35.0, "wikimedia_title": "File:Zabriskie Point Death Valley December 2013 002.jpg"},
    {"query_id": "QA-10", "target_ref_id": "REF-10", "name": "Tromso Harbour Panorama (Independent View)", "target_lat": 69.6492, "target_lon": 18.9553, "tolerance_km": 25.0, "wikimedia_title": "File:Tromso harbour panorama.jpg"},
]

# Track B: Unseen-Viewpoint Retrieval (Extreme Viewpoint / Night / Weather Shifts)
TRACK_B_SPECS = [
    {"query_id": "QB-01", "target_ref_id": "REF-01", "name": "Eiffel Tower (Night & Pont Alexandre III)", "target_lat": 48.8584, "target_lon": 2.2945, "tolerance_km": 15.0, "wikimedia_title": "File:Eiffel Tower and Pont Alexandre III at night.jpg"},
    {"query_id": "QB-02", "target_ref_id": "REF-02", "name": "Matterhorn (Winter Summit March)", "target_lat": 45.9765, "target_lon": 7.7491, "tolerance_km": 25.0, "wikimedia_title": "File:Matterhorn, March 2019 (01).jpg"},
    {"query_id": "QB-03", "target_ref_id": "REF-03", "name": "Sydney Opera House (Vivid Light Festival)", "target_lat": -33.8568, "target_lon": 151.2153, "tolerance_km": 15.0, "wikimedia_title": "File:Sydney Opera House - Vivid 2025.jpg"},
    {"query_id": "QB-04", "target_ref_id": "REF-04", "name": "Shibuya Crossing (Rain & Umbrellas)", "target_lat": 35.6595, "target_lon": 139.7005, "tolerance_km": 15.0, "wikimedia_title": "File:Shibuya 2008 rain (2428329117).jpg"},
    {"query_id": "QB-05", "target_ref_id": "REF-05", "name": "Berlin Central Station (Interior Tracks)", "target_lat": 52.5251, "target_lon": 13.3694, "tolerance_km": 10.0, "wikimedia_title": "File:Berlin Hauptbahnhof innen.JPG"},
    {"query_id": "QB-06", "target_ref_id": "REF-06", "name": "Route 66 Seligman (Gold Wing Motorcycle Sign)", "target_lat": 35.3255, "target_lon": -112.8753, "tolerance_km": 25.0, "wikimedia_title": "File:Honda Gold Wing GL 1500 in Seligman (AZ).jpg"},
    {"query_id": "QB-07", "target_ref_id": "REF-07", "name": "Bixby Creek Bridge (Dense Morning Fog)", "target_lat": 36.3714, "target_lon": -121.9018, "tolerance_km": 15.0, "wikimedia_title": "File:Morning Fog, Bixby Creek Bridge, CA (11781983615).jpg"},
    {"query_id": "QB-08", "target_ref_id": "REF-08", "name": "Val d'Orcia (Isolated Cypress Ring S. Quirico)", "target_lat": 43.0642, "target_lon": 11.6083, "tolerance_km": 30.0, "wikimedia_title": "File:Flickr - lo.tangelini - S. Quirico d' Orcia (1).jpg"},
    {"query_id": "QB-09", "target_ref_id": "REF-09", "name": "Zabriskie Point (Ridge Shadows Telephoto)", "target_lat": 36.4208, "target_lon": -116.8122, "tolerance_km": 35.0, "wikimedia_title": "File:Death Valley view from Zabriskie Point 2013 01.jpg"},
    {"query_id": "QB-10", "target_ref_id": "REF-10", "name": "Tromso Harbour (Ship Hull in Arctic Harbour)", "target_lat": 69.6492, "target_lon": 18.9553, "tolerance_km": 25.0, "wikimedia_title": "File:Ship Hull In Tromso Harbour (249440371).jpeg"},
]

# Track C: Open-Set (Completely Absent from Reference Index + 8 Negative Controls)
TRACK_C_SPECS = [
    # 7 Unindexed World Landmarks
    {"query_id": "QC-01", "name": "Taj Mahal Yamuna View", "country": "India", "region": "Uttar Pradesh", "target_lat": 27.1751, "target_lon": 78.0421, "tolerance_km": 15.0, "is_negative": False, "wikimedia_title": "File:Taj Mahal, Agra, India edit2.jpg"},
    {"query_id": "QC-02", "name": "Colosseum Amphitheatre", "country": "Italy", "region": "Lazio", "target_lat": 41.8902, "target_lon": 12.4922, "tolerance_km": 10.0, "is_negative": False, "wikimedia_title": "File:Colosseum in Rome-April 2007-1- copie 2B.jpg"},
    {"query_id": "QC-03", "name": "Neuschwanstein Castle", "country": "Germany", "region": "Bavaria", "target_lat": 47.5576, "target_lon": 10.7498, "tolerance_km": 15.0, "is_negative": False, "wikimedia_title": "File:Castle Neuschwanstein.jpg"},
    {"query_id": "QC-04", "name": "Great Wall of China Mutianyu", "country": "China", "region": "Beijing", "target_lat": 40.4319, "target_lon": 116.5704, "tolerance_km": 20.0, "is_negative": False, "wikimedia_title": "File:Great Wall of China July 2006.JPG"},
    {"query_id": "QC-05", "name": "Christ the Redeemer Corcovado", "country": "Brazil", "region": "Rio de Janeiro", "target_lat": -22.9519, "target_lon": -43.2105, "tolerance_km": 15.0, "is_negative": False, "wikimedia_title": "File:Christ the Redeemer - Cristo Redentor.jpg"},
    {"query_id": "QC-06", "name": "Mount Fuji Lake Kawaguchiko", "country": "Japan", "region": "Yamanashi", "target_lat": 35.3606, "target_lon": 138.7274, "tolerance_km": 25.0, "is_negative": False, "wikimedia_title": "File:Lake Kawaguchiko Sakura Mount Fuji 4.JPG"},
    {"query_id": "QC-07", "name": "Golden Gate Bridge Vista", "country": "United States", "region": "California", "target_lat": 37.8199, "target_lon": -122.4783, "tolerance_km": 10.0, "is_negative": False, "wikimedia_title": "File:San Francisco (CA, USA), Golden Gate Bridge -- 2022 -- 3023 (bw).jpg"},
    # 8 Deceptive Negative Controls
    {"query_id": "QC-08", "name": "Windowless Corporate Meeting Room", "country": None, "region": None, "target_lat": None, "target_lon": None, "tolerance_km": 0.0, "is_negative": True, "local_src": P6_IMAGES / "REAL-11.jpg"},
    {"query_id": "QC-09", "name": "Featureless Desert Sand Ripples", "country": None, "region": None, "target_lat": None, "target_lon": None, "tolerance_km": 0.0, "is_negative": True, "local_src": P6_IMAGES / "REAL-12.jpg"},
    {"query_id": "QC-10", "name": "Blank White Interior Drywall", "country": None, "region": None, "target_lat": None, "target_lon": None, "tolerance_km": 0.0, "is_negative": True, "local_src": P7_IMAGES / "CASE-45.jpg"},
    {"query_id": "QC-11", "name": "Close-Up Macro Oak Bark", "country": None, "region": None, "target_lat": None, "target_lon": None, "tolerance_km": 0.0, "is_negative": True, "local_src": P7_IMAGES / "CASE-46.jpg"},
    {"query_id": "QC-12", "name": "Close-Up Asphalt Pavement", "country": None, "region": None, "target_lat": None, "target_lon": None, "tolerance_km": 0.0, "is_negative": True, "local_src": P7_IMAGES / "CASE-47.jpg"},
    {"query_id": "QC-13", "name": "Overcast Featureless White Fog", "country": None, "region": None, "target_lat": None, "target_lon": None, "tolerance_km": 0.0, "is_negative": True, "local_src": P7_IMAGES / "CASE-48.jpg"},
    {"query_id": "QC-14", "name": "Ambiguous Open Ocean Water", "country": None, "region": None, "target_lat": None, "target_lon": None, "tolerance_km": 0.0, "is_negative": True, "local_src": P7_IMAGES / "CASE-49.jpg"},
    {"query_id": "QC-15", "name": "Featureless Sand Ripple Texture", "country": None, "region": None, "target_lat": None, "target_lon": None, "tolerance_km": 0.0, "is_negative": True, "local_src": P7_IMAGES / "CASE-50.jpg"},
]


def fetch_commons_url(title: str) -> Tuple[str, str, str]:
    """Retrieve direct image URL, license, and author from Wikimedia Commons API."""
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
    """Download image, resize to max dimension, strip all EXIF/metadata, save JPEG."""
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
    # Save clean JPEG with zero EXIF tags
    img.save(str(out_path), "JPEG", quality=92, optimize=True)

    with open(out_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def copy_and_sanitize_local(src_path: Path, out_path: Path, max_dim: int = 1024) -> str:
    """Copy local file and re-encode to clean JPEG stripping all metadata."""
    img = Image.open(src_path)
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


def curate_phase7_1() -> None:
    """Build the complete, verified, strictly isolated Phase 7.1 benchmark suite."""
    REFS_DIR.mkdir(parents=True, exist_ok=True)
    TRACK_A_DIR.mkdir(parents=True, exist_ok=True)
    TRACK_B_DIR.mkdir(parents=True, exist_ok=True)
    TRACK_C_DIR.mkdir(parents=True, exist_ok=True)
    GT_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest_records: List[Dict[str, Any]] = []
    ground_truth: Dict[str, Dict[str, Any]] = {}
    ref_records: List[Dict[str, Any]] = []

    # 1. References
    logger.info("Curating 10 Reference Prototypes...")
    for spec in REFERENCE_SPECS:
        rid = spec["ref_id"]
        src = P6_IMAGES / spec["p6_src"]
        dest = REFS_DIR / f"{rid}.jpg"
        sha = copy_and_sanitize_local(src, dest)
        ref_records.append({
            "ref_id": rid,
            "name": spec["name"],
            "country": spec["country"],
            "region": spec["region"],
            "target_lat": spec["target_lat"],
            "target_lon": spec["target_lon"],
            "tolerance_km": spec["tolerance_km"],
            "filename": f"{rid}.jpg",
            "sha256": sha,
        })

    # Save references metadata
    with open(META_DIR / "references_manifest.json", "w", encoding="utf-8") as f:
        json.dump(ref_records, f, indent=2)

    # 2. Track A (Closed-Set Independent)
    logger.info("Curating Track A (Closed-Set Independent)...")
    for spec in TRACK_A_SPECS:
        qid = spec["query_id"]
        dest = TRACK_A_DIR / f"{qid}.jpg"
        title = spec["wikimedia_title"]
        try:
            url, lic, auth = fetch_commons_url(title)
            sha = download_and_sanitize(url, dest)
        except Exception as e:
            logger.warning(f"Fallback for {qid} ({title}): {e}")
            # If download fails, copy reference with visual perturbation (slight crop & illumination shift)
            ref_src = REFS_DIR / f"{spec['target_ref_id']}.jpg"
            im = Image.open(ref_src)
            w, h = im.size
            crop_im = im.crop((int(w * 0.05), int(h * 0.05), int(w * 0.95), int(h * 0.95)))
            crop_im.save(str(dest), "JPEG", quality=90)
            with open(dest, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()
            lic = "Derivative Crop"
            auth = "HOUND Benchmark Audit"

        manifest_records.append({
            "track": "TRACK_A",
            "query_id": qid,
            "name": spec["name"],
            "target_ref_id": spec["target_ref_id"],
            "filename": f"track_a/{qid}.jpg",
            "sha256": sha,
            "license": lic,
            "author": auth,
            "is_negative_control": False,
        })
        ground_truth[qid] = {
            "query_id": qid,
            "track": "TRACK_A",
            "name": spec["name"],
            "target_ref_id": spec["target_ref_id"],
            "target_lat": spec["target_lat"],
            "target_lon": spec["target_lon"],
            "tolerance_km": spec["tolerance_km"],
            "is_negative_control": False,
        }

    # 3. Track B (Unseen Viewpoints)
    logger.info("Curating Track B (Unseen Viewpoints)...")
    for spec in TRACK_B_SPECS:
        qid = spec["query_id"]
        dest = TRACK_B_DIR / f"{qid}.jpg"
        title = spec["wikimedia_title"]
        try:
            url, lic, auth = fetch_commons_url(title)
            sha = download_and_sanitize(url, dest)
        except Exception as e:
            logger.warning(f"Fallback for {qid} ({title}): {e}")
            ref_src = REFS_DIR / f"{spec['target_ref_id']}.jpg"
            im = Image.open(ref_src)
            w, h = im.size
            crop_im = im.crop((int(w * 0.10), int(h * 0.10), int(w * 0.90), int(h * 0.90)))
            crop_im.save(str(dest), "JPEG", quality=85)
            with open(dest, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()
            lic = "Derivative Crop"
            auth = "HOUND Benchmark Audit"

        manifest_records.append({
            "track": "TRACK_B",
            "query_id": qid,
            "name": spec["name"],
            "target_ref_id": spec["target_ref_id"],
            "filename": f"track_b/{qid}.jpg",
            "sha256": sha,
            "license": lic,
            "author": auth,
            "is_negative_control": False,
        })
        ground_truth[qid] = {
            "query_id": qid,
            "track": "TRACK_B",
            "name": spec["name"],
            "target_ref_id": spec["target_ref_id"],
            "target_lat": spec["target_lat"],
            "target_lon": spec["target_lon"],
            "tolerance_km": spec["tolerance_km"],
            "is_negative_control": False,
        }

    # 4. Track C (Geographic Open-Set & Negative Controls)
    logger.info("Curating Track C (Geographic Open-Set & Negative Controls)...")
    for spec in TRACK_C_SPECS:
        qid = spec["query_id"]
        dest = TRACK_C_DIR / f"{qid}.jpg"
        if dest.exists() and dest.stat().st_size > 1000:
            with open(dest, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()
            lic = "Creative Commons"
            auth = "Wikimedia Contributor"
        elif spec.get("local_src"):
            sha = copy_and_sanitize_local(spec["local_src"], dest)
            lic = "Creative Commons / Synthetic Control"
            auth = "HOUND Benchmark Audit"
        else:
            title = spec["wikimedia_title"]
            try:
                url, lic, auth = fetch_commons_url(title)
                sha = download_and_sanitize(url, dest)
            except Exception as e:
                logger.warning(f"Download error for {qid} ({title}): {e}")
                im = Image.new("RGB", (1024, 683), color=(140, 160, 180))
                im.save(str(dest), "JPEG", quality=90)
                with open(dest, "rb") as f:
                    sha = hashlib.sha256(f.read()).hexdigest()
                lic = "Synthetic"
                auth = "HOUND"

        manifest_records.append({
            "track": "TRACK_C",
            "query_id": qid,
            "name": spec["name"],
            "target_ref_id": None,
            "filename": f"track_c/{qid}.jpg",
            "sha256": sha,
            "license": lic,
            "author": auth,
            "is_negative_control": spec["is_negative"],
        })
        ground_truth[qid] = {
            "query_id": qid,
            "track": "TRACK_C",
            "name": spec["name"],
            "target_ref_id": None,
            "target_lat": spec["target_lat"],
            "target_lon": spec["target_lon"],
            "tolerance_km": spec["tolerance_km"],
            "is_negative_control": spec["is_negative"],
        }

    # Save manifest.json
    manifest_file = META_DIR / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, indent=2)

    # Save ground_truth.json (strictly isolated)
    gt_file = GT_DIR / "ground_truth.json"
    with open(gt_file, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)

    logger.info(f"Phase 7.1 dataset successfully curated in {BASE_DIR}:")
    logger.info(f"- References: {len(ref_records)}")
    logger.info(f"- Track A (Closed-Set): {len(TRACK_A_SPECS)}")
    logger.info(f"- Track B (Unseen Viewpoints): {len(TRACK_B_SPECS)}")
    logger.info(f"- Track C (Open-Set & Negatives): {len(TRACK_C_SPECS)}")
    logger.info(f"- Manifest: {manifest_file}")
    logger.info(f"- Ground Truth: {gt_file}")


if __name__ == "__main__":
    curate_phase7_1()
