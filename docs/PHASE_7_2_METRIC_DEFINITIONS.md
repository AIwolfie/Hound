# HOUND — Phase 7.2: Mathematical Metric Definitions & Evaluation Protocol

**Document ID:** `HOUND-PROTOCOL-PHASE-7-2-METRICS`  
**Classification:** Scientific Methodology & Evaluation Specification  
**Author:** AIwolfie  
**Status:** Canonical Reference Standard  
**Date:** November 13, 2026  
**Target Repository:** `D:\project\Hound`  

---

## 1. Executive Summary

A core failure in geospatial visual place recognition benchmarks is metric conflation:
1. **Conflating Retrieval with Decision:** A model may rank a reference image correctly in top-$k$, but an autonomous agent must decide whether to *emit* that location or *abstain*. Evaluating only retrieval ranks hides false positive hallucination; evaluating only emitted points hides underlying model representation capacity.
2. **Conflating Open-Set Unseen Places with Non-Places:** An unseen real-world landmark (e.g. Taj Mahal when absent from index) presents a different failure mode than a featureless macro texture (e.g. tree bark). Combining them into one monolithic "False Positive Rate" obscures why an agent failed.
3. **Selective Exclusion in Error Metrics:** Calculating mean or median geographic error only on successful queries creates a survivor bias, reporting artificially low errors while hiding queries where the system had no idea where it was.

This document formalizes the exact mathematical equations, operational definitions, and accounting rules enforced across HOUND Phase 7.2.

---

## 2. Formal Mathematical Definitions

Let:
- $Q$ denote a set of query images.
- $R$ denote the reference database indexed with georeferenced coordinates $\text{coord}(r) = (\text{lat}_r, \text{lon}_r)$.
- $\text{coord}^*(q) = (\text{lat}_q^*, \text{lon}_q^*)$ denote the true ground-truth geographic coordinate of query $q$.
- $\text{tol}(q)$ denote the geographic ground-truth tolerance radius (typically $25.0\text{ km}$).
- $\text{Top-}k(q) = [r_1, r_2, \dots, r_k]$ denote the ranked list of retrieved references from $R$ in descending similarity order.
- $s(q, r_1)$ denote the top-1 retrieval confidence/similarity score.
- $\tau \in [0.0, 1.0]$ denote the calibrated decision threshold.
- $\text{emit}(q) = \mathbb{I}(s(q, r_1) \ge \tau)$ denote the binary decision to emit coordinates versus abstain.

---

### 2.1 Pure Retrieval Quality (Independent of Emission Gate)

#### 1. Recall@$k$ ($\text{Rec}@k$)
The fraction of closed-set queries where at least one reference within the top-$k$ retrieved candidates falls within the ground-truth geographic tolerance radius:

$$\text{Recall}@k = \frac{1}{|Q_{\text{closed}}|} \sum_{q \in Q_{\text{closed}}} \mathbb{I}\left(\exists r \in \text{Top-}k(q) \text{ s.t. } d_{\text{haversine}}(\text{coord}(r), \text{coord}^*(q)) \le \text{tol}(q)\right)$$

*Note:* Evaluated on closed-set queries ($Q_{\text{closed}} = \text{Track A} \cup \text{Track B}$) regardless of whether the agent chose to emit or abstain.

#### 2. Geographic Accuracy within Radius $d$ ($\text{Acc}_{\le d}$)
The proportion of queries whose top-1 retrieved candidate is within distance $d \in \{1\text{ km}, 5\text{ km}, 25\text{ km}, 100\text{ km}\}$ of ground truth:

$$\text{Acc}_{\le d} = \frac{1}{|Q_{\text{closed}}|} \sum_{q \in Q_{\text{closed}}} \mathbb{I}\left(d_{\text{haversine}}(\text{coord}(r_1), \text{coord}^*(q)) \le d\right)$$

---

### 2.2 Geographic Error Accounting (No Silent Exclusions)

For every query $q \in Q_{\text{closed}}$, the raw geographic error is:

$$e(q) = d_{\text{haversine}}(\text{coord}(r_1), \text{coord}^*(q))$$

#### Accounting Rule:
- For every emitted query ($\text{emit}(q) = 1$), the localization error is recorded as $e(q)$.
- For every abstained query ($\text{emit}(q) = 0$), the localization error is formally recorded as `UNAVAILABLE (ABSTAINED)`. It must **never** be silently deleted or replaced with $0.0\text{ km}$.

#### 3. Conditional Median & Mean Geographic Error
Evaluated strictly over the emitted subset $Q_{\text{emitted}} = \{q \in Q_{\text{closed}} \mid \text{emit}(q) = 1\}$:

$$\text{Median Error} = \text{median}\left(\{e(q) \mid q \in Q_{\text{emitted}}\}\right)$$

$$\text{Mean Error} = \frac{1}{|Q_{\text{emitted}}|} \sum_{q \in Q_{\text{emitted}}} e(q)$$

#### 4. Maximum Geographic Error ($\text{Max Error}$)
$$\text{Max Error} = \max\left(\{e(q) \mid q \in Q_{\text{emitted}}\}\right)$$

---

### 2.3 Calibrated Decision & Selective Geolocation

#### 5. Prediction Coverage ($\text{Cov}$)
The proportion of total queries for which the system made an active prediction rather than abstaining:

$$\text{Coverage}(\tau) = \frac{|\{q \in Q \mid \text{emit}(q) = 1\}|}{|Q|} = \frac{|Q_{\text{emitted}}|}{|Q|}$$

#### 6. Selective Accuracy ($\text{SelAcc}$)
The accuracy of the system restricted strictly to the queries it chose to emit:

$$\text{SelAcc}(\tau) = \frac{|\{q \in Q_{\text{emitted}} \mid e(q) \le \text{tol}(q)\}|}{|Q_{\text{emitted}}|}$$

#### 7. Empirical Risk versus Coverage
The risk (error rate) incurred at a given coverage level:

$$\text{Risk}(\tau) = 1 - \text{SelAcc}(\tau) = \frac{|\{q \in Q_{\text{emitted}} \mid e(q) > \text{tol}(q)\}|}{|Q_{\text{emitted}}|}$$

The Risk-Coverage curve is plotted by varying $\tau \in [0.0, 1.0]$. A well-calibrated agent exhibits monotonically decreasing risk as coverage is reduced.

---

### 2.4 Open-Set & Adversarial Metrics (Strictly Separated)

#### 8. False Localization Rate on Unseen Places ($\text{FLR}_{\text{C1}}$)
Evaluated on **Track C1** (real photographs of genuine physical locations absent from $R$):
$$\text{FLR}_{\text{C1}}(\tau) = \frac{|\{q \in Q_{\text{C1}} \mid \text{emit}(q) = 1\}|}{|Q_{\text{C1}}|}$$
Measures the frequency with which an agent hallucinates a known database location when inspecting a real, unfamiliar location on Earth.

#### 9. Correct Abstention Rate on Unseen Places ($\text{CAR}_{\text{C1}}$)
$$\text{CAR}_{\text{C1}}(\tau) = 1 - \text{FLR}_{\text{C1}}(\tau) = \frac{|\{q \in Q_{\text{C1}} \mid \text{emit}(q) = 0\}|}{|Q_{\text{C1}}|}$$

#### 10. False Emission Rate on Non-Place Negatives ($\text{FER}_{\text{C2}}$)
Evaluated on **Track C2** (indoor rooms, blank surfaces, bark, asphalt, fog, ocean water):
$$\text{FER}_{\text{C2}}(\tau) = \frac{|\{q \in Q_{\text{C2}} \mid \text{emit}(q) = 1\}|}{|Q_{\text{C2}}|}$$
Measures the frequency with which non-place noise is assigned geographic coordinates.

#### 11. Correct Abstention Rate on Non-Places ($\text{CAR}_{\text{C2}}$)
$$\text{CAR}_{\text{C2}}(\tau) = 1 - \text{FER}_{\text{C2}}(\tau) = \frac{|\{q \in Q_{\text{C2}} \mid \text{emit}(q) = 0\}|}{|Q_{\text{C2}}|}$$

#### 12. Adversarial False Localization Rate ($\text{FLR}_{\text{Adv}}$)
Evaluated on **Track D** (similar dome architecture, lookalike suspension bridges, stratovolcanoes, lookalike rail catenary, misleading road signage):
$$\text{FLR}_{\text{Adv}}(\tau) = \frac{|\{q \in Q_{\text{Adv}} \mid \text{emit}(q) = 1\}|}{|Q_{\text{Adv}}|}$$

---

## 3. Strict Reporting Matrix

In all Phase 7.2 benchmarks, reporting must conform to this schema:

```
========================================================================================================================
TRACK A (Closed-Set Independent)  : R@1, R@5, R@10, Acc@25km, Cond. Med Err, Cond. Mean Err, Coverage, Selective Acc
TRACK B (Unseen Viewpoints)       : R@1, R@5, R@10, Acc@25km, Cond. Med Err, Cond. Mean Err, Coverage, Selective Acc
TRACK C1 (Unseen Places)          : False Localization Rate (FLR), Correct Abstention Rate (CAR), Mean Emitted Score
TRACK C2 (Non-Place Negatives)    : False Emission Rate (FER), Correct Abstention Rate (CAR), Max Emitted Score
TRACK D (Adversarial Challenging) : False Localization Rate (FLR), Correct Abstention Rate (CAR), Failure Modes
========================================================================================================================
```

$\text{FLR}_{\text{C1}}$ and $\text{FER}_{\text{C2}}$ must **never** be averaged into a single "FPR" number.
