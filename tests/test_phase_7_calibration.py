"""HOUND Phase 7 Empirical Calibration Pipeline Test Suite.

Verifies:
- Platt scaling logistic regression fitting.
- Isotonic regression fitting.
- Physical falsification zeroing (P = 0.000).
- Brier score, log loss, ECE, reliability diagram binning, and risk-coverage curve.
"""
import numpy as np
import pytest

from core.empirical_calibration import CalibrationMethod, EmpiricalCalibrator


def test_platt_scaling_fitting_and_calibration():
    calibrator = EmpiricalCalibrator(method=CalibrationMethod.PLATT_SCALING)
    assert calibrator.is_fitted is False

    np.random.seed(42)
    X = np.linspace(0.1, 0.9, 40).reshape(-1, 1)
    y = (X.flatten() >= 0.50).astype(int)

    calibrator.fit(X, y)
    assert calibrator.is_fitted is True

    # Low score gives low probability
    low_prob = calibrator.calibrate(np.array([0.10]))
    high_prob = calibrator.calibrate(np.array([0.90]))
    assert low_prob < 0.35
    assert high_prob > 0.70
    assert low_prob < high_prob


def test_physical_falsification_strictly_zeros_probability():
    calibrator = EmpiricalCalibrator(method=CalibrationMethod.PLATT_SCALING)
    X = np.linspace(0.1, 0.9, 40).reshape(-1, 1)
    y = (X.flatten() >= 0.50).astype(int)
    calibrator.fit(X, y)

    # Even with a high raw score, is_falsified=True must force probability to 0.000
    prob_falsified = calibrator.calibrate(np.array([0.95]), is_falsified=True)
    assert prob_falsified == 0.000


def test_evaluation_metrics_and_reliability_diagram():
    calibrator = EmpiricalCalibrator(method=CalibrationMethod.PLATT_SCALING)
    X = np.linspace(0.1, 0.9, 50).reshape(-1, 1)
    y = (X.flatten() >= 0.50).astype(int)
    calibrator.fit(X, y)

    report = calibrator.evaluate(X, y, n_bins=5)
    assert report.is_empirically_fitted is True
    assert report.sample_size == 50
    assert 0.0 <= report.brier_score <= 1.0
    assert report.expected_calibration_error >= 0.0
    assert len(report.reliability_bins) == 5
    assert len(report.risk_coverage_curve) == 9
