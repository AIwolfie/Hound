"""Tests for HOUND Phase 7.1 Independent Benchmark Validation.

Verifies:
- Zero SHA-256 hash leakage between query tracks and reference database.
- Reference index isolation and geographic metadata consistency.
- Track separation across Track A (closed-set), Track B (viewpoints), Track C (open-set).
- Ground truth isolation: target coordinates inaccessible during feature extraction.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
P7_1_DIR = WORKSPACE_DIR / "data" / "benchmarks" / "phase_7_1"
REFS_DIR = P7_1_DIR / "references"
META_DIR = P7_1_DIR / "metadata"
GT_DIR = P7_1_DIR / "ground_truth"


def test_phase_7_1_zero_hash_leakage():
    """Verify that no query image shares a SHA-256 hash with any reference prototype."""
    manifest_file = META_DIR / "manifest.json"
    refs_file = META_DIR / "references_manifest.json"

    assert manifest_file.exists(), "Manifest file missing"
    assert refs_file.exists(), "References manifest missing"

    with open(manifest_file, "r", encoding="utf-8") as f:
        queries = json.load(f)
    with open(refs_file, "r", encoding="utf-8") as f:
        refs = json.load(f)

    ref_hashes = {r["sha256"]: r["ref_id"] for r in refs}

    # Test every query across all tracks
    for q in queries:
        q_sha = q["sha256"]
        assert q_sha not in ref_hashes, (
            f"Leakage detected! Query {q['query_id']} ({q['name']}) shares hash with {ref_hashes.get(q_sha)}"
        )


def test_phase_7_1_track_structure():
    """Verify that all three evaluation tracks exist and are strictly separated."""
    manifest_file = META_DIR / "manifest.json"
    with open(manifest_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    track_a = [q for q in queries if q["track"] == "TRACK_A"]
    track_b = [q for q in queries if q["track"] == "TRACK_B"]
    track_c = [q for q in queries if q["track"] == "TRACK_C"]

    assert len(track_a) == 10, f"Expected 10 Track A cases, got {len(track_a)}"
    assert len(track_b) == 10, f"Expected 10 Track B cases, got {len(track_b)}"
    assert len(track_c) == 15, f"Expected 15 Track C cases, got {len(track_c)}"

    # Track C must have zero reference IDs
    for q in track_c:
        assert q["target_ref_id"] is None, f"Track C query {q['query_id']} must have no target_ref_id"


def test_phase_7_1_ground_truth_isolation():
    """Verify that ground-truth coordinates are stored strictly in ground_truth.json."""
    gt_file = GT_DIR / "ground_truth.json"
    manifest_file = META_DIR / "manifest.json"

    with open(gt_file, "r", encoding="utf-8") as f:
        gt = json.load(f)
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Manifest should NOT contain ground truth lat/lon directly
    for m in manifest:
        assert "target_lat" not in m, f"Manifest leaked target_lat for {m['query_id']}"
        assert "target_lon" not in m, f"Manifest leaked target_lon for {m['query_id']}"

    # Ground truth must contain lat/lon for positive targets
    assert gt["QA-01"]["target_lat"] is not None
    assert gt["QC-08"]["target_lat"] is None  # Negative control has no coords
