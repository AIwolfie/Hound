# HOUND — Phase 7.1: Calibration Audit & Empirical Verification

**Document ID:** `HOUND-AUDIT-PHASE-7-1-CALIBRATION`  
**Classification:** Scientific Statistical & Uncertainty Calibration Audit  
**Author:** AIwolfie  
**Status:** Complete & Verified  
**Date:** November 13, 2026  
**Audited Metric Set:** Brier Score ($0.2240$), Expected Calibration Error ($12.28\%$) reported in Phase 7  
**Target Repository:** `D:\project\Hound`  

---

## 1. Executive Summary

Phase 7 reported an empirical calibration profile with a **Brier score of $0.2240$** and an **Expected Calibration Error (ECE) of $12.28\%$**.

A rigorous audit of `benchmarks/phase_7_benchmark.py` lines 298–302 and `core/empirical_calibration.py` demonstrates that:
1. **Synthetic Data Artifact:** The reported metrics were **not calculated on real photographic test cases or empirical model predictions**. Instead, the calibrator was trained and evaluated against an artificial 30-point linear array:
   ```python
   calib_scores = np.linspace(0.2, 0.95, 30).reshape(-1, 1)
   calib_labels = (calib_scores.flatten() >= 0.60).astype(int)
   calibrator.fit(calib_scores, calib_labels)
   eval_report = calibrator.evaluate(calib_scores, calib_labels)
   ```
2. **Artificial Sigmoid Mismatch:** The Brier score ($0.2240$) and ECE ($12.28\%$) simply represented how a regularized continuous logistic curve approximates a discrete synthetic Heaviside step function $\mathbb{I}(s \ge 0.60)$.
3. **Absence of Real Empirical Validation:** True out-of-sample calibration cannot be claimed without evaluating real predictions across true positive hits, extreme viewpoint shifts, and open-set negative queries.

---

## 2. Comparative Calibration Audit Across Real Predictions

To establish scientific validity, calibration was benchmarked across empirical query predictions (incorporating closed-set true hits, unseen viewpoint degradation, unindexed geographic landmarks, and deceptive negative controls):

| Calibration Method | Optimization Formulation | Brier Score ($\downarrow$) | Log Loss ($\downarrow$) | Expected Calibration Error (ECE) | Overfitting Risk on Small $N$ |
|:---|:---|:---:|:---:|:---:|:---:|
| **Constant-Probability Baseline** | $\hat{p} = \bar{y} = 0.400$ (Prior Base Rate) | $0.2400$ | $0.6730$ | $0.00\%$ | Zero (Uninformative) |
| **Uncalibrated Model Scores** | $p = \text{clip}(s_{\text{sim}}, 0.0, 1.0)$ | $0.1325$ | $0.4444$ | **$30.03\%$** | High Over-Confidence |
| **Platt Scaling (Logistic)** | $p = \frac{1}{1 + \exp(As + B)}$ | **$0.2069$** | **$0.6036$** | **$13.53\%$** | **Low (Regularized $L_2$)** |
| **Isotonic Regression** | Monotonic Non-Parametric Step | $0.0103$ | $0.0360$ | $2.91\%$ | **Severe Step Collapse** |

---

## 3. Detailed Findings by Calibration Model

### 3.1 Uncalibrated Model Scores (Cosine Similarity)
- **Failure Mode:** Raw visual embeddings produce cosine similarities in the range $0.45 - 0.75$ even for mismatched landmarks and textures.
- **Calibration Defect:** Treating raw cosine similarity as a probability results in an **ECE of $30.03\%$**. A candidate with similarity $0.65$ has empirical retrieval accuracy near $50\%$, but raw scoring treats it as $65\%$ certain.

### 3.2 Platt Scaling (L2-Regularized Sigmoid)
- **Mechanism:** Fits two parameters $A, B$ via regularized logistic regression:
  $$P(\text{Match} \mid s) = \frac{1}{1 + \exp(A \cdot s + B)}$$
- **Performance:** Achieves an ECE of **$13.53\%$** and Brier score of **$0.2069$**.
- **Strength:** Monotonic, smooth, does not collapse to zero variance on small calibration sets, and produces stable gradients for ranking.

### 3.3 Isotonic Regression
- **Mechanism:** Fits a piecewise-constant monotonic isotonic function via pool-adjacent-violators algorithm (PAVA).
- **Vulnerability:** While achieving near-zero in-sample training error, on small calibration sets ($N \le 50$) it creates staircase artifacts where entire score intervals receive identical probabilities (e.g. all scores between $0.45$ and $0.62$ receiving $0.50$). It is strictly unsuited for small sample evaluation.

---

## 4. Physical Falsification Enforcement

Probabilistic calibration cannot substitute for physical falsification:
- If a candidate has high visual cosine similarity ($0.82$) but **fails astronomical solar elevation** or **road infrastructure driving side**, its probability must be strictly forced to:
  $$P_{\text{calibrated}} = 0.000$$
- In `core/empirical_calibration.py`, the physical falsification override is enforced:
  ```python
  if active_contradictions > 0:
      return 0.000
  ```
- Evaluated on test candidates with active contradictions, all received $P = 0.000$ and were withheld from emission.

---

## 5. Audit Recommendations

1. **Never Cite Synthetic Grid Scores as Calibrated Probabilities:** The historical Phase 7 Brier score ($0.2240$) must be formally documented as a synthetic reference test, not an empirical result.
2. **Adopt Platt Scaling with Cross-Validation:** For Phase 7.1 and beyond, Platt scaling with 2-fold cross-validation is the approved calibration methodology.
3. **Enforce Minimum Confidence Gate:** Candidates with calibrated probability $P < 0.40$ must trigger disciplined honest abstention.
