"""HOUND Phase 7 Benchmark Suite Integrity & Isolation Tests.

Verifies:
- 50-case benchmark dataset completeness and manifest integrity.
- Cryptographic SHA-256 match for all 50 cases.
- Physical isolation of ground truth from inference code.
"""
import hashlib
import json
import re
from pathlib import Path
import pytest

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = WORKSPACE_DIR / "data" / "benchmarks" / "phase_7"
IMAGES_DIR = BASE_DIR / "images"
GT_FILE = BASE_DIR / "ground_truth" / "ground_truth.json"
MANIFEST_FILE = BASE_DIR / "metadata" / "manifest.json"


def test_phase_7_manifest_and_ground_truth_completeness():
    assert MANIFEST_FILE.exists(), f"Missing manifest: {MANIFEST_FILE}"
    assert GT_FILE.exists(), f"Missing ground truth: {GT_FILE}"

    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(GT_FILE, "r", encoding="utf-8") as f:
        gt = json.load(f)

    assert len(manifest) == 50, f"Expected 50 manifest records, found {len(manifest)}"
    assert len(gt) == 50, f"Expected 50 ground truth records, found {len(gt)}"

    # 42 positive, 8 negative controls
    pos = [c for c in manifest if not c["is_negative_control"]]
    neg = [c for c in manifest if c["is_negative_control"]]
    assert len(pos) == 42
    assert len(neg) == 8


def test_phase_7_image_cryptographic_hashes():
    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for entry in manifest:
        cid = entry["case_id"]
        img_path = IMAGES_DIR / f"{cid}.jpg"
        assert img_path.exists(), f"Missing image file for {cid}: {img_path}"

        with open(img_path, "rb") as f:
            actual_sha256 = hashlib.sha256(f.read()).hexdigest()

        assert actual_sha256 == entry["sha256"], f"Hash mismatch for {cid}: expected {entry['sha256']}, got {actual_sha256}"


def test_phase_7_ground_truth_isolation():
    """Verify that no production modules import or access phase_7 ground truth."""
    prod_dirs = [WORKSPACE_DIR / "agent", WORKSPACE_DIR / "core", WORKSPACE_DIR / "investigators", WORKSPACE_DIR / "storage"]
    pattern = re.compile(r'phase_7/ground_truth|ground_truth\.json', re.IGNORECASE)

    for pdir in prod_dirs:
        for py_file in pdir.rglob("*.py"):
            with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            matches = pattern.findall(content)
            assert len(matches) == 0, f"Found ground-truth reference in production code: {py_file}"
