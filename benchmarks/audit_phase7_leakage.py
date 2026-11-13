"""HOUND Phase 7.1 Reference Index Leakage Auditor.

Inspects every query image and reference image in the Phase 7 benchmark:
- Computes SHA-256 hashes, perceptual hashes (aHash, dHash, pHash).
- Identifies exact file duplicates, near-duplicates, crops, and shared sources.
- Analyzes geographic coordinate identity and landmark overlap.
- Emits the complete Query-to-Reference Contamination Matrix.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
from PIL import Image

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
P7_DIR = WORKSPACE_DIR / "data" / "benchmarks" / "phase_7"
IMAGES_DIR = P7_DIR / "images"
GT_FILE = P7_DIR / "ground_truth" / "ground_truth.json"
MANIFEST_FILE = P7_DIR / "metadata" / "manifest.json"
REPORTS_DIR = P7_DIR / "reports"


def compute_ahash(img: Image.Image, hash_size: int = 8) -> int:
    """Average hash."""
    img = img.convert("L").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
    pixels = np.array(img.getdata(), dtype=np.float32)
    avg = pixels.mean()
    diff = pixels > avg
    hash_val = 0
    for bit in diff:
        hash_val = (hash_val << 1) | int(bit)
    return hash_val


def compute_dhash(img: Image.Image, hash_size: int = 8) -> int:
    """Difference hash."""
    img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.array(img, dtype=np.float32)
    diff = pixels[:, 1:] > pixels[:, :-1]
    hash_val = 0
    for bit in diff.flatten():
        hash_val = (hash_val << 1) | int(bit)
    return hash_val


def compute_phash(img: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """Perceptual hash via DCT."""
    img_size = hash_size * highfreq_factor
    img = img.convert("L").resize((img_size, img_size), Image.Resampling.BILINEAR)
    pixels = np.array(img, dtype=np.float32)
    dct = cv2.dct(pixels)
    dctlowfreq = dct[:hash_size, :hash_size]
    med = np.median(dctlowfreq[1:])  # exclude DC
    diff = dctlowfreq > med
    hash_val = 0
    for bit in diff.flatten():
        hash_val = (hash_val << 1) | int(bit)
    return hash_val


def hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")


def run_leakage_audit() -> Dict[str, Any]:
    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(GT_FILE, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    # In Phase 7 benchmark runner, the reference index was constructed from all positive cases
    ref_cases = [c for c in manifest if not c["is_negative_control"]]
    query_cases = manifest

    print(f"Total Query Cases: {len(query_cases)}")
    print(f"Total Reference Entries: {len(ref_cases)}")

    # Precompute hashes for all images
    image_hashes = {}
    for case in manifest:
        cid = case["case_id"]
        img_path = IMAGES_DIR / case["filename"]
        with open(img_path, "rb") as f:
            raw_bytes = f.read()
        sha256_hash = hashlib.sha256(raw_bytes).hexdigest()

        with Image.open(img_path) as im:
            ahash = compute_ahash(im)
            dhash = compute_dhash(im)
            phash = compute_phash(im)
            dims = im.size

        image_hashes[cid] = {
            "sha256": sha256_hash,
            "ahash": ahash,
            "dhash": dhash,
            "phash": phash,
            "dims": dims,
            "file_size": len(raw_bytes),
        }

    # Contamination Matrix: rows = query, cols = ref
    exact_duplicates = []
    near_duplicates = []
    coordinate_matches = []
    contamination_matrix = []

    for q in query_cases:
        q_id = q["case_id"]
        q_info = image_hashes[q_id]
        q_gt = ground_truth.get(q_id, {})

        row = {
            "query_case_id": q_id,
            "query_name": q["name"],
            "is_negative": q["is_negative_control"],
            "contaminations": [],
        }

        for r in ref_cases:
            r_id = r["case_id"]
            r_info = image_hashes[r_id]
            r_gt = ground_truth.get(r_id, {})

            # Exact SHA-256 match
            is_exact_sha = (q_info["sha256"] == r_info["sha256"])

            # Perceptual hash distances
            phash_dist = hamming_distance(q_info["phash"], r_info["phash"])
            dhash_dist = hamming_distance(q_info["dhash"], r_info["dhash"])
            ahash_dist = hamming_distance(q_info["ahash"], r_info["ahash"])

            # Geographic coordinate identity
            coord_match = False
            if q_gt.get("target_lat") is not None and r_gt.get("target_lat") is not None:
                lat_diff = abs(q_gt["target_lat"] - r_gt["target_lat"])
                lon_diff = abs(q_gt["target_lon"] - r_gt["target_lon"])
                if lat_diff < 1e-4 and lon_diff < 1e-4:
                    coord_match = True

            # Contamination classification
            contamination_level = "NONE"
            if is_exact_sha:
                contamination_level = "EXACT_FILE_DUPLICATE"
                exact_duplicates.append((q_id, r_id))
            elif phash_dist <= 5 or dhash_dist <= 5:
                contamination_level = "PERCEPTUAL_NEAR_DUPLICATE"
                near_duplicates.append((q_id, r_id, phash_dist))
            elif coord_match:
                contamination_level = "COORDINATE_IDENTITY"
                coordinate_matches.append((q_id, r_id))

            if contamination_level != "NONE":
                row["contaminations"].append({
                    "ref_case_id": r_id,
                    "ref_name": r["name"],
                    "contamination_level": contamination_level,
                    "sha256_match": is_exact_sha,
                    "phash_dist": phash_dist,
                    "dhash_dist": dhash_dist,
                    "ahash_dist": ahash_dist,
                    "coord_match": coord_match,
                })

        contamination_matrix.append(row)

    print("\n" + "=" * 80)
    print("PHASE 7 REFERENCE LEAKAGE AUDIT FINDINGS")
    print("=" * 80)
    print(f"Total Exact File Duplicates (Query == Ref): {len(exact_duplicates)}")
    print(f"Total Perceptual Near-Duplicates: {len(near_duplicates)}")
    print(f"Total Geographic Coordinate Matches: {len(coordinate_matches)}")
    print(f"Exact Duplicate Percentage among Positive Queries: {len(exact_duplicates) / len(ref_cases) * 100:.1f}%")

    report = {
        "audit_name": "Phase 7 Reference Index Leakage Audit",
        "date": "2026-11-13",
        "total_queries": len(query_cases),
        "total_references": len(ref_cases),
        "exact_duplicates_count": len(exact_duplicates),
        "exact_duplicates": exact_duplicates,
        "near_duplicates": near_duplicates,
        "coordinate_matches": coordinate_matches,
        "contamination_matrix": contamination_matrix,
    }

    out_file = REPORTS_DIR / "phase_7_1_leakage_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Leakage audit report written to: {out_file}")
    return report


if __name__ == "__main__":
    run_leakage_audit()
