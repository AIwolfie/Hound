"""HOUND Empirical Geolocation Calibration Pipeline.

Replaces heuristic sigmoid parameters with mathematically fitted empirical calibration:
- Platt Scaling (L2-regularized logistic regression) and Isotonic Regression via scikit-learn.
- Rigorous scoring metrics: Brier Score, Log Loss, Expected Calibration Error (ECE), Maximum Calibration Error (MCE).
- Reliability Diagrams (empirical binning) and Risk-Coverage Curves for selective classification.
- Enforces strict scientific labeling: never labels uncalibrated scores as calibrated probabilities.
- Enforces physical falsification dominance: active contradiction strictly clamps probability to 0.000.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from core.logger import get_logger

logger = get_logger("hound.calibration.empirical")

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available; EmpiricalCalibrator running in fallback mode.")


class CalibrationMethod(str, Enum):
    PLATT_SCALING = "platt_scaling"
    ISOTONIC = "isotonic_regression"
    RAW_HEURISTIC = "raw_heuristic"


@dataclass
class ReliabilityBin:
    bin_lower: float
    bin_upper: float
    sample_count: int
    mean_predicted_prob: float
    empirical_accuracy: float
    calibration_error: float


@dataclass
class RiskCoveragePoint:
    confidence_threshold: float
    coverage: float  # Fraction of population where confidence >= threshold
    selective_accuracy: float  # Accuracy over retained predictions
    selective_risk: float  # 1.0 - selective_accuracy


@dataclass
class CalibrationEvaluationReport:
    """Comprehensive mathematical evaluation of a calibration model on held-out data."""
    method: str
    is_empirically_fitted: bool
    sample_size: int
    brier_score: float
    log_loss_value: float
    expected_calibration_error: float
    maximum_calibration_error: float
    reliability_bins: List[ReliabilityBin]
    risk_coverage_curve: List[RiskCoveragePoint]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "is_empirically_fitted": self.is_empirically_fitted,
            "sample_size": self.sample_size,
            "brier_score": round(self.brier_score, 4),
            "log_loss": round(self.log_loss_value, 4),
            "expected_calibration_error": round(self.expected_calibration_error, 4),
            "maximum_calibration_error": round(self.maximum_calibration_error, 4),
            "reliability_bins": [asdict(b) for b in self.reliability_bins],
            "risk_coverage_curve": [asdict(p) for p in self.risk_coverage_curve],
        }


class EmpiricalCalibrator:
    """Empirical calibration engine for geographic certainty estimation."""

    def __init__(self, method: CalibrationMethod = CalibrationMethod.PLATT_SCALING):
        self.method = method
        self.is_fitted = False
        self._platt_model: Optional[LogisticRegression] = None
        self._isotonic_model: Optional[IsotonicRegression] = None
        self._fitted_params: Dict[str, Any] = {}

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Fit calibration model on geographically independent training/validation partition.

        Args:
            features: 2D array of shape (N, D) where columns represent:
                      [raw_score, independent_modalities_count, score_margin] or 1D array of scores.
            labels: 1D binary array of shape (N,) where 1 = correct localization, 0 = incorrect.
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required to fit empirical calibration models.")

        X = np.atleast_2d(features)
        if X.shape[0] == 1 and features.ndim == 1:
            X = X.T
        y = np.array(labels, dtype=int).flatten()

        if len(np.unique(y)) < 2:
            raise ValueError(f"Calibration fitting requires at least two classes in labels, found {np.unique(y)}.")

        if self.method == CalibrationMethod.PLATT_SCALING:
            # L2-regularized logistic regression (Platt scaling)
            self._platt_model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            self._platt_model.fit(X, y)
            self._fitted_params = {
                "intercept": float(self._platt_model.intercept_[0]),
                "coefficients": self._platt_model.coef_[0].tolist(),
            }
            self.is_fitted = True
            logger.info(f"Platt scaling fitted successfully: Intercept={self._fitted_params['intercept']:.3f}, Coefs={self._fitted_params['coefficients']}")

        elif self.method == CalibrationMethod.ISOTONIC:
            # Isotonic regression (1D non-parametric monotonic fit)
            self._isotonic_model = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
            X_1d = X[:, 0] if X.shape[1] > 1 else X.flatten()
            self._isotonic_model.fit(X_1d, y)
            self.is_fitted = True
            logger.info("Isotonic regression fitted successfully.")

    def calibrate(
        self,
        features: np.ndarray,
        is_falsified: bool = False,
    ) -> float:
        """Calibrate a single candidate's score into an empirical probability.

        Enforces physical falsification dominance: if is_falsified is True, returns 0.000.
        """
        if is_falsified:
            return 0.000

        if not self.is_fitted:
            # If model is not fitted, return raw score normalized but NEVER label as empirical probability
            logger.debug("Calibrator not fitted; applying fallback bounding.")
            raw_s = float(features[0] if isinstance(features, (list, np.ndarray)) else features)
            return round(min(0.39, max(0.01, raw_s * 0.40)), 3)

        X = np.atleast_2d(features)
        if X.shape[0] == 1 and np.array(features).ndim == 1:
            X = X.T

        if self.method == CalibrationMethod.PLATT_SCALING and self._platt_model is not None:
            prob = float(self._platt_model.predict_proba(X)[0, 1])
            return round(min(0.99, max(0.01, prob)), 3)

        elif self.method == CalibrationMethod.ISOTONIC and self._isotonic_model is not None:
            X_1d = X[:, 0] if X.shape[1] > 1 else X.flatten()
            prob = float(self._isotonic_model.predict(X_1d)[0])
            return round(min(0.99, max(0.01, prob)), 3)

        return 0.010

    def evaluate(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 5,
    ) -> CalibrationEvaluationReport:
        """Evaluate calibration performance on a geographically independent held-out test split."""
        X = np.atleast_2d(features)
        if X.shape[0] == 1 and features.ndim == 1:
            X = X.T
        y_true = np.array(labels, dtype=int).flatten()
        n = len(y_true)

        # Generate predicted probabilities
        if self.is_fitted and self._platt_model is not None:
            y_prob = self._platt_model.predict_proba(X)[:, 1]
        elif self.is_fitted and self._isotonic_model is not None:
            X_1d = X[:, 0] if X.shape[1] > 1 else X.flatten()
            y_prob = self._isotonic_model.predict(X_1d)
        else:
            # Fallback
            y_prob = np.clip(X[:, 0], 0.01, 0.99)

        # 1. Brier Score
        brier = float(np.mean((y_prob - y_true) ** 2))

        # 2. Log Loss (cross-entropy)
        eps = 1e-15
        p_clipped = np.clip(y_prob, eps, 1 - eps)
        ll = float(-np.mean(y_true * np.log(p_clipped) + (1 - y_true) * np.log(1 - p_clipped)))

        # 3. Expected Calibration Error (ECE) & Reliability Bins
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bins_data: List[ReliabilityBin] = []
        ece = 0.0
        mce = 0.0

        for b_idx in range(n_bins):
            b_low = bin_edges[b_idx]
            b_high = bin_edges[b_idx + 1]

            if b_idx == n_bins - 1:
                mask = (y_prob >= b_low) & (y_prob <= b_high)
            else:
                mask = (y_prob >= b_low) & (y_prob < b_high)

            count = int(np.sum(mask))
            if count > 0:
                mean_p = float(np.mean(y_prob[mask]))
                acc = float(np.mean(y_true[mask]))
                cal_err = abs(mean_p - acc)
                ece += (count / n) * cal_err
                mce = max(mce, cal_err)
            else:
                mean_p = (b_low + b_high) / 2.0
                acc = 0.0
                cal_err = 0.0

            bins_data.append(
                ReliabilityBin(
                    bin_lower=round(b_low, 2),
                    bin_upper=round(b_high, 2),
                    sample_count=count,
                    mean_predicted_prob=round(mean_p, 4),
                    empirical_accuracy=round(acc, 4),
                    calibration_error=round(cal_err, 4),
                )
            )

        # 4. Risk-Coverage Curve (Selective Classification)
        thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
        rc_points: List[RiskCoveragePoint] = []

        for th in thresholds:
            retained = y_prob >= th
            retained_count = int(np.sum(retained))
            cov = retained_count / max(1, n)
            if retained_count > 0:
                sel_acc = float(np.mean(y_true[retained]))
            else:
                sel_acc = 1.0  # Perfect precision when zero claims are made
            rc_points.append(
                RiskCoveragePoint(
                    confidence_threshold=th,
                    coverage=round(cov, 4),
                    selective_accuracy=round(sel_acc, 4),
                    selective_risk=round(1.0 - sel_acc, 4),
                )
            )

        return CalibrationEvaluationReport(
            method=self.method.value,
            is_empirically_fitted=self.is_fitted,
            sample_size=n,
            brier_score=round(brier, 4),
            log_loss_value=round(ll, 4),
            expected_calibration_error=round(ece, 4),
            maximum_calibration_error=round(mce, 4),
            reliability_bins=bins_data,
            risk_coverage_curve=rc_points,
        )
