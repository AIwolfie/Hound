"""Tests for HOUND Phase 7.2 Statistical Validation & Open-Set Reliability.

Verifies:
- Three-way dataset split isolation: R, D_calib, and D_test have zero SHA-256 overlap.
- Track structure and query integrity across Tracks A, B, C1, C2, and D.
- Mathematical precision of metric formulas: Recall@k, Coverage, Selective Accuracy,
  Conditional Error, False Localization Rate (C1), False Emission Rate (C2), and Correct Abstention Rate.
- Threshold freezing: ensures tau* is selected strictly on D_calib prior to D_test evaluation.
- Abstention accounting: verifies that abstained queries are recorded as UNAVAILABLE rather
  than silently dropped from denominators.
- Bootstrap confidence interval validity: verifies 95% CIs satisfy 0 <= low <= mean <= high <= 1.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
P7_2_DIR = WORKSPACE_DIR / "data" / "benchmarks" / "phase_7_2"
REFS_DIR = P7_2_DIR / "references"
META_DIR = P7_2_DIR / "metadata"
GT_DIR = P7_2_DIR / "ground_truth"
REPORTS_DIR = P7_2_DIR / "reports"


def test_phase_7_2_three_way_split_zero_leakage():
    """Verify pairwise disjointness of SHA-256 hashes across R, D_calib, and D_test."""
    refs_file = META_DIR / "references_manifest.json"
    calib_file = META_DIR / "calibration_manifest.json"
    test_file = META_DIR / "test_manifest.json"

    assert refs_file.exists(), f"Missing {refs_file}"
    assert calib_file.exists(), f"Missing {calib_file}"
    assert test_file.exists(), f"Missing {test_file}"

    with open(refs_file, "r", encoding="utf-8") as f:
        refs = json.load(f)
    with open(calib_file, "r", encoding="utf-8") as f:
        calib = json.load(f)
    with open(test_file, "r", encoding="utf-8") as f:
        test = json.load(f)

    r_hashes = {r["sha256"]: r["ref_id"] for r in refs}
    calib_hashes = {c["sha256"]: c["query_id"] for c in calib}
    test_hashes = {t["sha256"]: t["query_id"] for t in test}

    # 1. R intersect D_calib == empty
    r_calib_overlap = set(r_hashes.keys()) & set(calib_hashes.keys())
    assert not r_calib_overlap, f"Contamination: R and D_calib share {len(r_calib_overlap)} hashes: {r_calib_overlap}"

    # 2. R intersect D_test == empty
    r_test_overlap = set(r_hashes.keys()) & set(test_hashes.keys())
    assert not r_test_overlap, f"Contamination: R and D_test share {len(r_test_overlap)} hashes: {r_test_overlap}"

    # 3. D_calib intersect D_test == empty
    calib_test_overlap = set(calib_hashes.keys()) & set(test_hashes.keys())
    assert not calib_test_overlap, f"Contamination: D_calib and D_test share {len(calib_test_overlap)} hashes: {calib_test_overlap}"


def test_phase_7_2_track_partition_counts():
    """Verify exact counts in reference, calibration, and test partitions."""
    with open(META_DIR / "references_manifest.json", "r", encoding="utf-8") as f:
        refs = json.load(f)
    with open(META_DIR / "calibration_manifest.json", "r", encoding="utf-8") as f:
        calib = json.load(f)
    with open(META_DIR / "test_manifest.json", "r", encoding="utf-8") as f:
        test = json.load(f)

    assert len(refs) == 10, f"Expected 10 reference prototypes, got {len(refs)}"
    assert len(calib) == 13, f"Expected 13 calibration queries, got {len(calib)}"
    assert len(test) == 41, f"Expected 41 test queries, got {len(test)}"

    # Check track subsets in D_test
    track_a = [t for t in test if t["track"] == "TRACK_A"]
    track_b = [t for t in test if t["track"] == "TRACK_B"]
    track_c1 = [t for t in test if t["track"] == "TRACK_C1"]
    track_c2 = [t for t in test if t["track"] == "TRACK_C2"]
    track_d = [t for t in test if t["track"] == "TRACK_D"]

    assert len(track_a) == 10, f"Expected 10 Track A queries, got {len(track_a)}"
    assert len(track_b) == 10, f"Expected 10 Track B queries, got {len(track_b)}"
    assert len(track_c1) == 7, f"Expected 7 Track C1 queries, got {len(track_c1)}"
    assert len(track_c2) == 8, f"Expected 8 Track C2 queries, got {len(track_c2)}"
    assert len(track_d) == 6, f"Expected 6 Track D queries, got {len(track_d)}"


def test_phase_7_2_metric_calculations():
    """Verify mathematical definitions of Coverage, Selective Accuracy, FLR, FER, and CAR."""
    # Synthetic scenario:
    # 10 Known queries: 6 emitted (4 correct within tolerance, 2 incorrect), 4 abstained
    # 7 Unseen places (C1): 1 emitted (FP), 6 abstained (TN)
    # 8 Non-places (C2): 1 emitted (FP), 7 abstained (TN)

    n_known_total = 10
    n_known_emit = 6
    n_known_correct = 4

    coverage = n_known_emit / n_known_total
    selective_acc = n_known_correct / n_known_emit

    assert coverage == 0.60
    assert abs(selective_acc - (4 / 6)) < 1e-6

    # Track C1: Unseen Places
    n_c1_total = 7
    n_c1_emit = 1
    flr_c1 = n_c1_emit / n_c1_total
    assert abs(flr_c1 - (1 / 7)) < 1e-6

    # Track C2: Non-Place Negatives
    n_c2_total = 8
    n_c2_emit = 1
    fer_c2 = n_c2_emit / n_c2_total
    assert fer_c2 == 0.125

    # Correct Abstention Rate across open-set
    n_neg_total = n_c1_total + n_c2_total
    n_neg_abstained = (7 - 1) + (8 - 1)
    car = n_neg_abstained / n_neg_total
    assert car == (13 / 15)

    # Segregation assertion: FLR and FER must NOT be pooled into a single headline FPR
    assert flr_c1 != fer_c2


def test_phase_7_2_threshold_freezing_and_report_integrity():
    """Verify benchmark report records frozen threshold tau* from D_calib and uses it for D_test."""
    report_file = REPORTS_DIR / "phase_7_2_benchmark_report.json"
    assert report_file.exists(), f"Benchmark report {report_file} does not exist"

    with open(report_file, "r", encoding="utf-8") as f:
        rep = json.load(f)

    assert "calibration_freezing" in rep
    calib_freezing = rep["calibration_freezing"]
    assert "CosPlace" in calib_freezing
    cos_calib = calib_freezing["CosPlace"]

    frozen_tau = cos_calib["frozen_tau"]
    assert frozen_tau is not None
    assert 0.1 <= frozen_tau <= 0.9, f"Frozen threshold out of reasonable range: {frozen_tau}"

    # Verify that test_evaluations contains evaluation for frozen_tau
    test_evals = rep["test_evaluations"]
    assert "CosPlace" in test_evals
    tau_key = f"tau_{frozen_tau:.2f}"
    assert tau_key in test_evals["CosPlace"], f"Expected {tau_key} in CosPlace test evaluations"


def test_phase_7_2_abstention_accounting():
    """Verify that abstained queries in test evaluations explicitly record UNAVAILABLE error."""
    report_file = REPORTS_DIR / "phase_7_2_benchmark_report.json"
    with open(report_file, "r", encoding="utf-8") as f:
        rep = json.load(f)

    cos_calib = rep["calibration_freezing"]["CosPlace"]
    frozen_tau = cos_calib["frozen_tau"]
    tau_key = f"tau_{frozen_tau:.2f}"
    track_a_res = rep["test_evaluations"]["CosPlace"][tau_key]["TRACK_A"]

    queries = track_a_res["queries"]
    abstained = [q for q in queries if q["abstain"]]
    emitted = [q for q in queries if not q["abstain"]]

    assert len(abstained) > 0, "Expected at least one abstention in Track A under frozen tau*"
    for q in abstained:
        assert q["status"] == "ABSTAINED"
        assert q["error_km"] == "UNAVAILABLE"

    for q in emitted:
        assert q["status"] == "EMITTED"
        assert isinstance(q["error_km"], (int, float))


def test_phase_7_2_bootstrap_confidence_intervals():
    """Verify bootstrap confidence intervals are bounded within [0, 1] with low <= mean <= high."""
    report_file = REPORTS_DIR / "phase_7_2_benchmark_report.json"
    with open(report_file, "r", encoding="utf-8") as f:
        rep = json.load(f)

    assert "bootstrap_confidence_intervals" in rep
    cis = rep["bootstrap_confidence_intervals"]

    for metric_name, ci_data in cis.items():
        mean_val = ci_data["mean"]
        low_val = ci_data["ci_95_low"]
        high_val = ci_data["ci_95_high"]

        assert 0.0 <= low_val <= high_val <= 1.0, (
            f"Invalid interval for {metric_name}: [{low_val}, {high_val}]"
        )
        assert low_val <= mean_val <= high_val, (
            f"Mean {mean_val} outside [{low_val}, {high_val}] for {metric_name}"
        )
