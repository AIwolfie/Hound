# HOUND — PHASE 7.2 CALIBRATION & UNCERTAINTY QUANTIFICATION AUDIT

**Author:** AIwolfie  
**Auditor / Operator:** RIOT  
**Project:** HOUND — Autonomous Visual Geolocation & OSINT Agent  
**Date:** 2026-11-13  
**Status:** COMPLETE & INDEPENDENTLY AUDITED  
**Workspace:** `D:/project/Hound`

---

## 1. Executive Summary & Calibration Objectives

In open-world visual place recognition and OSINT investigation, raw cosine similarity between neural embeddings cannot be treated as a calibrated probability of geographic identity. Cosine similarity reflects relative orientation in a high-dimensional feature space; it is uncalibrated, sensitive to visual frequency distribution, and lacks probabilistic meaning.

Phase 7.2 executes a rigorous calibration and uncertainty evaluation:
1. **Separation of Calibration from Evaluation:** All calibration parameters (threshold $\tau^*$, Platt logistic coefficients) were derived strictly on the calibration partition $\mathcal{D}_{\text{calib}}$ ($N = 13$) and frozen before testing on $\mathcal{D}_{\text{test}}$ ($N = 41$).
2. **Objective Threshold Derivation:** Rather than arbitrarily selecting thresholds, $\tau^*$ was derived via an explicit objective function balancing selective accuracy against false emissions.
3. **Multi-Model Calibration Auditing:** Raw cosine similarities, Platt logistic scaling, Temperature scaling, and Constant base-rate predictions were evaluated using proper scoring rules (Brier score and Expected Calibration Error).
4. **Murphy Decomposition:** Dissecting Brier score into Reliability, Resolution, and Uncertainty components to expose why low base-rate domains can flatter naive predictors.

---

## 2. Calibration Partition & Threshold Derivation

### 2.1 The Objective Function
Threshold selection on $\mathcal{D}_{\text{calib}}$ was governed by:
$$J(\tau) = \text{SelectiveAccuracy}(\tau) - \text{FalseEmissionRate}(\tau)$$
where:
$$\text{SelectiveAccuracy}(\tau) = \frac{\sum_{i \in \text{In-Set}} \mathbb{I}(s_i \ge \tau \land \text{correct}_i)}{\sum_{i \in \text{In-Set}} \mathbb{I}(s_i \ge \tau)}$$
$$\text{FalseEmissionRate}(\tau) = \frac{\sum_{j \in \text{Negative}} \mathbb{I}(s_j \ge \tau)}{N_{\text{Negative}}}$$

### 2.2 Empirical Sweep on $\mathcal{D}_{\text{calib}}$ ($N = 13$)

| Threshold $\tau$ | In-Set Coverage | In-Set Selective Acc. | Non-Place $\text{FER}$ | Objective $J(\tau)$ | Status |
|---|---|---|---|---|---|
| $0.20$ | $100.0\%$ (5/5) | $55.6\%$ (5/9) | $50.0\%$ (2/4) | $+0.056$ | Sub-optimal (high false emissions) |
| $0.22$ | $100.0\%$ (5/5) | $55.6\%$ (5/9) | $50.0\%$ (2/4) | $+0.056$ | Sub-optimal |
| $0.24$ | $100.0\%$ (5/5) | $55.6\%$ (5/9) | $50.0\%$ (2/4) | $+0.056$ | Sub-optimal |
| $0.26$ | $100.0\%$ (5/5) | $55.6\%$ (5/9) | $50.0\%$ (2/4) | $+0.056$ | Sub-optimal |
| **$0.28$ ($\tau^*$)** | **$100.0\%$ (5/5)** | **$100.0\%$ (5/5)** | **$0.0\%$ (0/4)** | **$+1.000$** | **Optimal Operating Threshold** |
| $0.30$ | $100.0\%$ (5/5) | $100.0\%$ (5/5) | $0.0\%$ (0/4) | $+1.000$ | Redundant margin |
| $0.40$ (Historical) | $80.0\%$ (4/5) | $100.0\%$ (4/4) | $0.0\%$ (0/4) | $+1.000$ | Overly conservative (drops valid recall) |
| $0.55$ (Strict) | $40.0\%$ (2/5) | $100.0\%$ (2/2) | $0.0\%$ (0/4) | $+1.000$ | Severe recall loss |

**Derivation Verdict:** $\tau^* = 0.28$ is the optimal operating threshold. Below $0.28$, false emissions on negative controls jump immediately to $50\%$. At and above $0.28$, all negative controls are successfully rejected ($\text{FER} = 0\%$) while retaining maximum in-set coverage.

---

## 3. Parametric Probability Calibration (Platt Scaling)

To transform raw cosine similarities $s \in [-1, 1]$ into a calibrated probability $P(Y = 1 \mid s) \in [0, 1]$, Platt scaling was fitted on $\mathcal{D}_{\text{calib}}$:
$$P(Y = 1 \mid s) = \sigma(\beta s + \alpha) = \frac{1}{1 + \exp(-(\beta s + \alpha))}$$

Fitted parameters:
- **Slope ($\beta$):** $1.5988$
- **Intercept ($\alpha$):** $-1.2709$

```
Mapping Curve:
Cosine Score 0.20  --> Calibrated Prob: 0.2787
Cosine Score 0.28* --> Calibrated Prob: 0.3051  (Threshold Boundary)
Cosine Score 0.40  --> Calibrated Prob: 0.3472
Cosine Score 0.60  --> Calibrated Prob: 0.4227
Cosine Score 0.80  --> Calibrated Prob: 0.5020
```

---

## 4. Calibration Auditing on Untouched Test Partition ($\mathcal{D}_{\text{test}}$, $N = 41$)

The frozen models were evaluated across the 41 independent test cases.

### Table 2: Proper Scoring Rules and Calibration Errors

| Calibration Method | Formula / Parameters | Brier Score ($BS$) $\downarrow$ | ECE (10 Bins) $\downarrow$ | Sharpness / Resolution |
|---|---|---|---|---|
| **Constant Base-Rate** | $p_i = \bar{y} \approx 0.0488$ | $0.0464$ | $0.0000$ | Zero (uninformative) |
| **Raw Cosine Similarity** | $p_i = s_i$ | $0.0890$ | $0.2352$ | Moderate ($s \in [0.15, 0.66]$) |
| **Platt Scaling** | $\sigma(1.5988 \cdot s - 1.2709)$ | $0.1039$ | $0.2512$ | Compressed ($p \in [0.26, 0.45]$) |

---

## 5. Statistical Interpretation: The Murphy Decomposition Paradox

The Brier score of the Constant Base-Rate predictor ($0.0464$) appears superior to both Raw Scores ($0.0890$) and Platt Scaling ($0.1039$). This paradox is resolved by the **Murphy Decomposition**:
$$BS = \text{Reliability} - \text{Resolution} + \text{Uncertainty}$$

1. **Uncertainty ($\bar{y}(1 - \bar{y})$):**
   In $\mathcal{D}_{\text{test}}$, the true positive success rate under extreme open-world conditions is $2 / 41 \approx 4.88\%$. The intrinsic base uncertainty is $\bar{y}(1 - \bar{y}) = 0.0488 \times 0.9512 = 0.0464$.
2. **Resolution:**
   The constant predictor outputs $p = 0.0488$ for every query. Its $\text{Resolution} = 0$. It cannot differentiate between an exact landmark photograph and indoor drywall. Its low Brier score is entirely an artifact of class imbalance.
3. **Platt Scaling Under Extreme Shift:**
   Because the training calibration partition $\mathcal{D}_{\text{calib}}$ had a higher positive prevalence ($5/13 \approx 38.5\%$) than the uncurated open-set test partition ($4.9\%$), Platt scaling produced conservative probabilities centered around $0.30 - 0.45$. On out-of-sample open-set queries that passed the threshold, these probabilities incurred non-zero squared error penalties.

---

## 6. Engineering Decisions for Calibrated Geolocation

1. **Never Report Raw Similarity as Probability:** Raw cosine similarity is an uncalibrated ranking metric, not an odds ratio.
2. **Decouple Thresholding from Probability Output:**
   - Decision to EMIT vs ABSTAIN is governed strictly by the metric margin boundary $\tau^* = 0.28$.
   - Confidence probability is conditioned on emission and must incorporate geometric verification inliers.
3. **Physical Falsification Overrides Probability:**
   No matter how high the statistical score $P(Y = 1 \mid s)$ appears, if downstream physical falsification (sun angle inconsistent with latitude, architectural era contradiction, or zero local feature inliers) fails, the candidate's confidence is hard-zeroed and the system abstains.
