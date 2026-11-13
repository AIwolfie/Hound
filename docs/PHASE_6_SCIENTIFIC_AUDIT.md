# HOUND — Phase 6: Independent Scientific & Empirical Audit Report

**Document ID:** `HOUND-AUDIT-PHASE-6`  
**Classification:** Independent Scientific & Methodological Audit  
**Author:** AIwolfie  
**Auditor System:** HOUND Forensic Audit Suite  
**Status:** Complete & Approved  
**Date:** November 13, 2026  
**Target Codebase:** `D:\project\Hound` (Commit `1ee45d3`)  
**Audit Scope:** Benchmark Integrity, Metric Consistency, Calibration Validity, Model Robustness, and Benchmark Scalability  

---

## 1. Executive Summary & Audit Mandate

This document provides a scientific, evidence-grounded audit of the Phase 6 implementation of **HOUND (Autonomous Visual Geolocation & OSINT Agent)**. 

Phase 6 represented a fundamental milestone in HOUND's evolution: moving from in-memory synthetic color canvases (Phases 5 and 5.1) to genuine real-world photographic sensor data (`data/benchmarks/phase_6/`). While Phase 6 demonstrated crucial engineering advancements—specifically the integration of an adversarial text sanitizer and a calibrated abstention layer—this audit was commissioned to rigorously investigate reported metric inconsistencies, evaluate whether the calibration parameters are empirically valid, audit ground-truth isolation, establish model baselines, and design an expanded evaluation protocol for Phase 7.

### Primary Audit Findings

1. **Benchmark Integrity & Non-Contamination: VERIFIED CLEAN**
   - All 12 real-world photographs in `data/benchmarks/phase_6/` cryptographically match their registered SHA-256 digests in `manifest.json`.
   - All images originate from documented Creative Commons (CC BY, CC BY-SA, Public Domain) sources on Wikimedia Commons with complete author attributions.
   - EXIF headers and GPS tags are strictly absent from benchmark imagery.
   - Ground truth (`ground_truth.json`) is physically isolated in a distinct directory; static analysis confirms that zero production modules (`agent/`, `core/`, `investigators/`, `storage/`) import or reference ground truth.
   - Zero benchmark target images or embeddings leaked into the reference database (`hound.db`), gazetteer, or cache.

2. **Resolution of Reported Metric Inconsistencies: IDENTIFIED & RESOLVED**
   - The reported combination: **Berlin Hbf localized within 1.94 km**, **Abstention Rate: 100%**, and **False-Positive Coordinate Rate: 0%** was audited down to its raw execution trace.
   - **Root Cause:** A dual-layer evaluation pipeline with mismatched reporting denominators.
     - The agent's candidate generator formulated a top candidate (`Area: Berlin, Deutschland` via station signage OCR) that was **1.94 km** from ground truth.
     - The benchmark harness computed `top1_accuracy_25km` on the *unfiltered raw candidate pool* prior to calibration ($1 / 10 = 10.0\%$).
     - Concurrently, the newly implemented `CalibrationEngine` evaluated the candidate, identified that it lacked multi-modal corroboration, assigned an `INSUFFICIENT` confidence tier ($P = 0.312 < 0.40$), and flagged `abstention_recommended = True`.
     - Because `abstention_recommended` was True across all 12 cases, the benchmark reported `abstention_rate_pct = 100.0%` and `localization_coverage_pct = 0.0%`.
     - **Reporting Flaw:** Presenting "10% Top-1 Accuracy" alongside "100% Abstention" creates a false impression that coordinates were emitted. In operational reality, HOUND withheld coordinates on all 12 cases. The metric must be formally reported as **Candidate Generator Recall@1 = 10.0%** and **Deployed Calibrated Accuracy = 0.0% (with 100% Abstention)**.

3. **Calibration Engine Audit: HEURISTIC TRANSFER, NOT EMPIRICALLY FITTED**
   - The logistic sigmoid parameters ($\text{Intercept} = -3.20$, $\text{Score} = 0.065$, $\text{Modality} = 0.85$, $\text{Contradiction} = -2.50$) were **manually selected by expert design**, not fitted via Maximum Likelihood Estimation (Platt scaling) or isotonic regression on a held-out dataset.
   - **Empirical Brier Score:** $0.0774$.
   - **Expected Calibration Error (ECE):** $0.0982$ ($9.82\%$).
   - Physical falsification was audited and verified: active contradiction strictly clamps probability to $P = 0.0$ and forces immediate abstention.
   - Until parameters are fitted against hundreds of real cases in Phase 7, these scores must be designated as **heuristic confidence scores formatted via a sigmoid transfer function**, rather than true empirical posterior probabilities.

4. **Model Baseline Audit: HANDCRAFTED COLOR PYRAMIDS FAIL REAL PHOTOGRAPHY**
   - Handcrafted 1224-d color-spatial pyramids execute in **2.25 ms** on CPU, but their cosine similarity drops to **0.5366** under a +40% illumination shift.
   - PyTorch ResNet-18 embeddings execute in **11.85 ms** on CUDA and maintain **0.9772** cosine similarity under the same illumination shift.
   - Neither existing baseline provides true instance-level place recognition across unindexed open-world photographs, proving that Phase 7 must integrate pre-trained metric-learning VPR architectures (CosPlace, MixVPR, or AnyLoc).

---

## 2. Benchmark Integrity & Contamination Audit

### 2.1. Dataset Manifest & Cryptographic Verification

The Phase 6 benchmark consists of 12 real photographs acquired from Wikimedia Commons, standardized to a maximum dimension of 1024 pixels, and stored in `data/benchmarks/phase_6/images/`.

All 12 files were verified against `data/benchmarks/phase_6/metadata/manifest.json`:

```bash
python benchmarks/audit_phase6.py
```

**Audit Output:**
```text
Total manifest entries: 12
Total ground truth entries: 12
VERIFIED: REAL-01 | SHA-256: 5a71a4de88376e23... | License: Public domain | Author: Benh LIEU SONG
VERIFIED: REAL-02 | SHA-256: 638823df68c41aea... | License: CC BY-SA 3.0 | Author: Zacharie Grossen / chil
VERIFIED: REAL-03 | SHA-256: b0e9f07f975d7e93... | License: CC BY-SA 3.0 | Author: Diliff
VERIFIED: REAL-04 | SHA-256: 9bbae4c097de52be... | License: CC BY-SA 2.0 | Author: Benh LIEU SONG
VERIFIED: REAL-05 | SHA-256: 04df1e4f5e50a70e... | License: CC BY 2.0   | Author: hrs51
VERIFIED: REAL-06 | SHA-256: 2cc4426e523b43b7... | License: CC BY-SA 4.0 | Author: Pierre André Leclercq
VERIFIED: REAL-07 | SHA-256: d8c995681d9761e3... | License: CC BY-SA 3.0 | Author: King of Hearts
VERIFIED: REAL-08 | SHA-256: d3f8ce1413d816bd... | License: CC BY-SA 2.0 | Author: Eric Kilby
VERIFIED: REAL-09 | SHA-256: de1f0e2a93108335... | License: CC BY-SA 3.0 | Author: Paxson Woelber
VERIFIED: REAL-10 | SHA-256: fb9b52569cec5370... | License: CC BY-SA 4.0 | Author: Diego Delso
VERIFIED: REAL-11 | SHA-256: 9fdf48be75c4fc88... | License: CC BY-SA 4.0 | Author: Tttrung
VERIFIED: REAL-12 | SHA-256: 82f450b2825daf00... | License: CC BY-SA 4.0 | Author: Thomas Fuhrmann
SUCCESS: All 12 benchmark images cryptographically match manifest.
```

- **Hash Integrity:** 100% (12/12 matching SHA-256 digests).
- **Licensing Compliance:** 100% open-source / Creative Commons compliant. Zero proprietary, copyrighted, or unlicensed assets.
- **EXIF Sanitization:** Automated inspection confirmed zero EXIF tags, zero GPS coordinates, and zero camera serial numbers exist in the benchmark directory.

### 2.2. Ground-Truth Isolation Audit

To ensure no data leakage occurs during autonomous execution, the codebase was audited for ground-truth references:

1. **Filesystem Isolation:** Ground truth is located exclusively at `data/benchmarks/phase_6/ground_truth/ground_truth.json`.
2. **Static Code Analysis:** A regex scan for `ground_truth` across all production source directories (`agent/`, `core/`, `investigators/`, `storage/`) returned **0 matches**. Ground truth is referenced strictly in test harnesses (`tests/test_phase_6_leakage.py`, `benchmarks/phase_6_benchmark.py`).
3. **Image Filename Analysis:** Benchmark image filenames are formatted as anonymous alphanumeric tokens (`REAL-01.jpg` to `REAL-12.jpg`). Zero geographic names, country codes, or coordinates exist in filenames.

### 2.3. Reference Database & Cache Leakage Audit

A potential vulnerability in geolocation systems is reference index memorization (the target image or its exact coordinates existing in the training/retrieval set).

- **VPR Prototype Database:** `core/visual_place_recognition.py` contains 10 reference prototypes (`REF-FRA-PARIS`, `REF-USA-ROUTE66`, etc.). These prototypes are **procedurally painted synthetic canvases** generated by NumPy color blocks (e.g., `ref_canvas[:128, :] = [210, 150, 80]`). None of the 12 real photographic images or their neural feature embeddings were ever inserted into the prototype database.
- **Gazetteer Fallback Cache:** The offline gazetteer contains 7 fallback administrative entries. It was verified that target coordinates for `REAL-05` (Berlin Central Station) were resolved dynamically via forward geocoding on the OCR-extracted token `"Berlin"`, not retrieved from a hardcoded ground-truth cache.

### 2.4. Geographic Independence of Splits

The 12 benchmark targets were audited for spatial autocorrelation:
- **Europe:** `REAL-01` (Paris, FR), `REAL-02` (Matterhorn, CH), `REAL-05` (Berlin, DE), `REAL-08` (Tuscany, IT), `REAL-10` (Tromsø, NO). The minimum pairwise distance between European cases is **680 km** (Paris to Matterhorn).
- **North America:** `REAL-06` (Seligman, AZ), `REAL-07` (Bixby Bridge, CA), `REAL-09` (Death Valley, CA). Minimum distance is **370 km** between distinct biomes (Pacific coast vs high desert vs Mojave basin).
- **Asia / Pacific:** `REAL-04` (Tokyo, JP), `REAL-03` (Sydney, AU).
- **Negative Controls:** `REAL-11` (Indoor corporate meeting room), `REAL-12` (Erg Chebbi sand dunes, Morocco).

Zero spatial overlap or near-duplicate camera viewpoints exist between cases.

---

## 3. Resolution of Metric Inconsistencies

### 3.1. The Apparent Paradox

The Phase 6 benchmark results reported the following combination of figures:
- **Berlin Central Station (`REAL-05`):** Localized within **1.94 km**.
- **Top-1 Accuracy ($\le 25\text{ km}$):** **10.0%** (1/10 positive cases).
- **Overall Abstention Rate:** **100.0%** (12/12 cases).
- **False Positive Rate on Negative Controls:** **0.0%**.
- **Localization Coverage:** **0.0%**.

An external auditor would reasonably ask: *How can a system achieve 10% accuracy while simultaneously abstaining on 100% of cases and achieving 0% coverage?*

### 3.2. Code Trace & Denominator Analysis

The audit traced this paradox directly to `benchmarks/phase_6_benchmark.py`:

```python
# Lines 473, 481, 488, 498 in benchmarks/phase_6_benchmark.py
w_25km = sum(1 for r in pos if r.within_25km)
abstained = sum(1 for r in results if r.abstention_recommended or r.final_status == "insufficient_evidence")
coverage = sum(1 for r in pos if r.predicted_lat is not None and not r.abstention_recommended)

top1_accuracy_25km = round(w_25km / max(1, pos_count) * 100.0, 1)
localization_coverage_pct = round(coverage / max(1, pos_count) * 100.0, 1)
abstention_rate_pct = round(abstained / tot * 100.0, 1)
```

1. **How Berlin Hbf Achieved 1.94 km:**
   - In case `REAL-05`, the agent ran 8 iterations.
   - Multi-pass OCR extracted text including `"Berlin"` from the station environment.
   - The candidate generator forward-geocoded `"Berlin"`, creating candidate `Area: Berlin, Deutschland` at $(52.5173885^\circ\text{N}, 13.3951309^\circ\text{E})$.
   - The ground truth in `ground_truth.json` is $(52.5251^\circ\text{N}, 13.3694^\circ\text{E})$.
   - Haversine distance between candidate and ground truth is **$1,940\text{ meters} = 1.94\text{ km}$**.
   - Because $1.94\text{ km} \le 25.0\text{ km}$, `r.within_25km` was marked `True`.

2. **Why the Case Was Abstained:**
   - `CalibrationEngine.evaluate_candidate` evaluated the candidate:
     - Raw heuristic score: $S = 6.4$.
     - Independent modalities: $M_{\text{indep}} = 2$ (`infrastructure`, `ocr`).
     - Calibrated probability formula yielded:
       $$z = -3.20 + 0.065 \cdot (6.4) + 0.85 \cdot (2) = -1.084$$
       $$P = \frac{1}{1 + e^{1.084}} \approx 0.312$$
     - Because $P = 0.312 < 0.40$, the engine assigned tier `INSUFFICIENT` and flagged `abstention_recommended = True`.

3. **The Denominator Mismatch:**
   - **`top1_accuracy_25km`** was computed over all positive targets ($N_{\text{pos}} = 10$) using the raw candidate distance, **completely ignoring the abstention flag**.
   - **`abstention_rate_pct`** was computed over all cases ($N_{\text{total}} = 12$) based on the calibration engine's recommendation.
   - **`localization_coverage_pct`** was computed as positive cases where coordinates were present *and* abstention was not recommended ($0 / 10 = 0.0\%$).

### 3.3. Standardized Terminology Correction

To eliminate ambiguity, HOUND's reporting must separate candidate generation recall from calibrated system output:

| Metric Name | Formula | Phase 6 Value | Operational Meaning |
|:---|:---:|:---:|:---|
| **Candidate Generator Recall@1 ($\le 25\text{ km}$)** | $\frac{N(\text{top cand } \le 25\text{km})}{N_{\text{pos}}}$ | **$10.0\%$** (1/10) | The agent formulated a candidate within 25 km of ground truth in its ranked hypothesis ledger. |
| **Calibrated Localization Coverage** | $\frac{N(\text{emitted coordinates})}{N_{\text{pos}}}$ | **$0.0\%$** (0/10) | Fraction of positive cases where the agent possessed sufficient calibrated certainty to emit coordinates. |
| **Calibrated Selective Accuracy** | $\frac{N(\text{correct emitted})}{N(\text{emitted})}$ | **Undefined / N/A** | Accuracy on cases where the agent actually claimed coordinates (0 predictions made). |
| **End-to-End Calibrated Accuracy** | $\frac{N(\text{correct emitted})}{N_{\text{pos}}}$ | **$0.0\%$** (0/10) | Deployed end-to-end accuracy under calibration gating. |
| **Honest Abstention Rate** | $\frac{N(\text{abstained})}{N_{\text{total}}}$ | **$100.0\%$** (12/12) | System recognized that available evidence was insufficient to claim pinpoint coordinates. |
| **Negative Control False-Positive Rate** | $\frac{N(\text{neg with emitted coords})}{N_{\text{neg}}}$ | **$0.0\%$** (0/2) | The agent emitted zero false coordinates on featureless/deceptive scenes. |

**Audit Conclusion:** There is no arithmetic bug in the code, but there was a reporting ambiguity in presenting candidate recall as system accuracy alongside 100% abstention.

---

## 4. Calibration Engine Scientific Audit

### 4.1. Parameter Origin Analysis

`core/calibration_engine.py` implements the following calibration formula:

$$z = \text{INTERCEPT} + w_{\text{score}} \cdot S + w_{\text{mod}} \cdot \min(3, M) - w_{\text{contra}} \cdot C$$

$$P = \frac{1}{1 + e^{-z}}$$

With static constants:
- $\text{INTERCEPT} = -3.20$
- $w_{\text{score}} = 0.065$
- $w_{\text{mod}} = 0.85$
- $w_{\text{contra}} = 2.50$

**Audit Finding:** These parameters were **manually selected by heuristic engineering**, not fitted through statistical regression (e.g., maximum likelihood estimation on cross-validation folds). They encode sensible domain heuristics (single clues cannot exceed $P = 0.35$; contradictions penalize by $2.50$), but calling the resulting value an "empirically calibrated probability" is methodologically inaccurate. It is a **heuristic certainty score mapped through a logistic sigmoid transfer function**.

### 4.2. Quantitative Calibration Metrics

To establish a mathematical baseline, the Phase 6 benchmark predictions were audited using `benchmarks/calculate_calibration_metrics.py`.

Ground truth outcome $y_i \in \{0, 1\}$ was defined as: $y_i = 1$ if the case is a positive target localized within $25\text{ km}$, and $y_i = 0$ otherwise.

```bash
python benchmarks/calculate_calibration_metrics.py
```

#### Results:
- **Empirical Brier Score:** **$0.0774$**  
  $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (P_i - y_i)^2 = 0.0774$$
  *(A Brier score of 0.0774 indicates strong overall penalty minimization, primarily driven by the engine correctly assigning low probabilities ($P < 0.32$) to unlocalized cases).*

- **Expected Calibration Error (ECE):** **$0.0982$ ($9.82\%$)** across 5 uniform bins $[0.0, 1.0]$.

#### Reliability Diagram Data:

| Probability Bin | Sample Count ($N_b$) | Mean Predicted Prob ($\bar{P}_b$) | Empirical Accuracy ($\bar{y}_b$) | Absolute Error ($|\bar{P}_b - \bar{y}_b|$) |
|:---:|:---:|:---:|:---:|:---:|
| **$[0.0, 0.2)$** | $7$ | $0.0890$ | $0.0000$ | $0.0890$ |
| **$[0.2, 0.4)$** | $5$ | $0.3110$ | $0.2000$ (1/5: Berlin Hbf) | $0.1110$ |
| **$[0.4, 0.6)$** | $0$ | N/A | N/A | N/A |
| **$[0.6, 0.8)$** | $0$ | N/A | N/A | N/A |
| **$[0.8, 1.0]$** | $0$ | N/A | N/A | N/A |

### 4.3. Physical Falsification Enforcement

The audit verified lines 67–68 and 155–157 of `core/calibration_engine.py`:
- When a candidate is marked `is_falsified = True` (e.g. via solar ephemeris ceiling or road curvature contradiction), the probability is strictly clamped:
  $$\text{If } \text{is\_falsified} \implies P = 0.000$$
- `abstention_recommended` is set to `True`, and `confidence_tier` is forced to `INSUFFICIENT`.
- Unit test verification in `tests/test_phase_6_calibration.py::test_active_falsification_zeros_probability` passed deterministically.

### 4.4. Decoupling Resolution Level from Probability

The audit confirmed that `GeographicResolutionLevel` (`LANDMARK`, `STREET`, `DISTRICT`, `CITY`, `REGION`, `COUNTRY`, `UNLOCALIZED`) is strictly decoupled from `calibrated_probability`. A candidate can have high probability at the `COUNTRY` level while remaining `UNLOCALIZED` at the `POINT` level, preventing the system from hallucinating unwarranted spatial precision.

---

## 5. Model Evaluation & Empirical Baseline

### 5.1. Measured Performance on Target Hardware

Hardware Environment:
- **Host Platform:** Windows 11
- **GPU:** NVIDIA GeForce RTX 2050 (4GB VRAM, CUDA 11.8)
- **CPU:** Intel Core i5-12450H @ 2.50 GHz (8 cores, 12 threads)
- **Frameworks:** PyTorch 2.7.1+cu118, Torchvision 0.22.1+cu118, OpenCV 4.13.0

Using `benchmarks/model_evaluation_baseline.py`, the existing visual descriptors were benchmarked across extraction latency, matching speed, and perturbation robustness on real benchmark imagery:

| Model / Descriptor | Type | Dimensionality | Compute Device | Extraction Latency | Pairwise Match Latency |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Hybrid Color-Spatial** | Handcrafted (HSV + Sobel) | $1224\text{ dims}$ | CPU | **$2.25\text{ ms}$** | $0.05\text{ ms}$ (Cosine) |
| **ResNet-18 (ImageNet)** | Deep CNN (AvgPool) | $512\text{ dims}$ | CUDA | **$11.85\text{ ms}$** | $0.03\text{ ms}$ (Cosine) |
| **ResNet-50 (ImageNet)** | Deep CNN (AvgPool) | $2048\text{ dims}$ | CUDA | **$17.65\text{ ms}$** | $0.04\text{ ms}$ (Cosine) |
| **ORB + RANSAC** | Local Keypoint | Variable ($N \times 32$) | CPU | **$14.47\text{ ms}$** | $19.22\text{ ms}$ (Homography) |
| **SIFT + RANSAC** | Local Keypoint | Variable ($N \times 128$) | CPU | **$93.55\text{ ms}$** | $164.16\text{ ms}$ (Homography) |

### 5.2. Invariance Stress Testing

Each descriptor was evaluated on target photograph `REAL-01.jpg` against three standard photographic perturbations:
1. **Illumination Shift:** $+40\%$ brightness increase with $+30$ intensity offset.
2. **Field-of-View Shift:** $80\%$ center crop.
3. **In-Plane Rotation:** $15^\circ$ affine rotation.

| Perturbation | Hybrid Color-Spatial (Cosine Sim) | ResNet-18 Neural (Cosine Sim) |
|:---|:---:|:---:|
| **Baseline (Self-Identity)** | $1.0000$ | $1.0000$ |
| **Brightness Increase (+40%)** | **$0.5366$** (Severe Degradation) | **$0.9772$** (Robust) |
| **Center Crop (80%)** | $0.9681$ | $0.9349$ |
| **In-Plane Rotation (15°)** | $0.8473$ | $0.8422$ |

**Scientific Takeaway:** The handcrafted color pyramid experiences catastrophic failure under illumination shifts (dropping from 1.0 to 0.5366 under daylight variation), explaining why real photographs failed to match synthetic database prototypes in Phase 6. ResNet-18 exhibits strong photometric invariance, but because it was trained on ImageNet category classification, its feature representations cluster semantically (e.g. grouping all bridges together) rather than geographically.

---

## 6. Audit Verdict & Non-Negotiable Remediation Plan

### Final Audit Verdict: CONDITIONAL PASS

Phase 6 successfully introduced real photographic evaluation, cryptographic integrity, adversarial prompt-injection protection, and a principled abstention mechanism. However, before proceeding with Phase 7 feature development, the following non-negotiable scientific remediations must be enacted:

1. **Reporting Clarification:**
   In all public reports and documentation, replace raw "Top-1 Accuracy" with **Candidate Generator Recall@1** and report **Deployed Calibrated Accuracy** as $0.0\%$ with $100.0\%$ honest abstention.
2. **Empirical Calibration Requirement:**
   Do not describe heuristic sigmoid scores as empirical probabilities until parameters are fitted on a designated validation split of $\ge 50$ real-world scenes.
3. **VPR Architecture Upgrade:**
   Deprecate handcrafted color pyramids as the primary visual place recognition mechanism for real photography. Phase 7 must integrate a dedicated metric-learning VPR model (**CosPlace** or **AnyLoc**).
4. **Benchmark Expansion:**
   Expand the real-world benchmark from 12 cases to a 50-case geographically balanced planetary evaluation dataset.
