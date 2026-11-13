# HOUND — PHASE 7.2 OPEN-SET & ADVERSARIAL RELIABILITY EVALUATION

**Author:** AIwolfie  
**Auditor / Operator:** RIOT  
**Project:** HOUND — Autonomous Visual Geolocation & OSINT Agent  
**Date:** 2026-11-13  
**Status:** COMPLETE & EMPIRICALLY VALIDATED  
**Workspace:** `D:/project/Hound`

---

## 1. Executive Summary & Core Principle

In operational geolocation and visual intelligence (OSINT), **abstention is a primary valid outcome**. An autonomous agent that honestly admits an image cannot be located is infinitely more trustworthy than one that confidently hallucinates coordinates thousands of kilometers away.

In prior phases, benchmark evaluations frequently grouped all non-matching inputs into a monolithic "negative" or "open-set" category. Phase 7.2 dismantles this flawed aggregation by introducing three distinct, non-overlapping open-set tracks:
1. **Track C1 — Unseen Geographic Locations ($N = 7$):** Genuine photographs of prominent global places that do not exist in the reference database $\mathcal{R}$.
2. **Track C2 — Non-Place Negative Controls ($N = 8$):** Macro textures, indoor walls, open water, featureless skies, and blur devoid of geographic information.
3. **Track D — Adversarial Decoys & Lookalikes ($N = 6$):** Visually deceptive real-world scenes engineered to mimic reference landmarks (dome architecture, suspension bridges, stratovolcanoes, rail catenaries, misleading foreign road signage).

### Headline Principle
> **Rule:** Never combine Track C1 and Track C2 into a single headline False Positive Rate. A model can exhibit near-zero false localization on unseen landmarks while catastrophically emitting coordinates on indoor drywall, or vice versa.

---

## 2. Empirical Results Across Evaluation Tracks

Table 1 summarizes performance across all 6 evaluated visual models on the untouched test partition $\mathcal{D}_{\text{test}}$ ($N = 41$).

### Table 1: Comprehensive Model Comparison on $\mathcal{D}_{\text{test}}$

| Model / Pipeline | Track A Coverage | Track A Sel. Acc. | Track B Coverage | Track B Sel. Acc. | Track C1 $\text{FLR}$ (Unseen) | Track C2 $\text{FER}$ (Non-Place) | Track D $\text{FLR}$ (Adversarial) | Overall Correct Abstention ($\text{CAR}$) |
|---|---|---|---|---|---|---|---|---|
| **CosPlace ($\tau^* = 0.28$)** | **60.0%** | **66.7%** | **40.0%** | **75.0%** | **14.3% (1/7)** | **12.5% (1/8)** | **16.7% (1/6)** | **85.7% (18/21)** |
| CosPlace ($\tau = 0.40$) | 40.0% | 75.0% | 10.0% | 100.0% | 0.0% (0/7) | 0.0% (0/8) | 0.0% (0/6) | 100.0% (21/21) |
| CosPlace ($\tau = 0.55$) | 20.0% | 100.0% | 0.0% | N/A | 0.0% (0/7) | 0.0% (0/8) | 0.0% (0/6) | 100.0% (21/21) |
| **AnyLoc (DINOv2 ViT-S/14)** | 100.0% | 70.0% | 100.0% | 90.0% | **100.0% (7/7)** | **100.0% (8/8)** | **100.0% (6/6)** | **0.0% (0/21)** |
| **MixVPR (ResNet-50)** | 20.0% | 50.0% | 10.0% | 100.0% | **0.0% (0/7)** | **75.0% (6/8)** | **0.0% (0/6)** | **61.9% (13/21)** |
| **ResNet-18 (ImageNet)** | 100.0% | 50.0% | 100.0% | 50.0% | 100.0% | 25.0% | 100.0% | 28.6% |
| **Hybrid Color/Spatial** | 100.0% | 20.0% | 100.0% | 50.0% | 100.0% | 87.5% | 100.0% | 4.8% |
| **Multi-Engine Fusion** | 60.0% | 50.0% | 40.0% | 75.0% | 14.3% (1/7) | 12.5% (1/8) | 16.7% (1/6) | 85.7% (18/21) |

---

## 3. Dissecting the Divergent Failure Modes: C1 vs C2

The empirical results conclusively prove why $\text{FLR}_{\text{C1}}$ and $\text{FER}_{\text{C2}}$ must be kept separate:

### 3.1 The MixVPR Asymmetry
- On **Track C1 (Unseen Real Places)**, MixVPR achieved a flawless **$0.0\%$ False Localization Rate** (7/7 abstained). Its feature-mixing stages correctly suppressed unseen architectural forms because they did not match the global spatial layout of the index prototypes.
- On **Track C2 (Non-Place Controls)**, MixVPR collapsed to a catastrophic **$75.0\%$ False Emission Rate** (6/8 false positives). High-contrast asphalt grains, corrugated metal lines, and wall outlets activated the feature mixer's row/column projection channels, generating high artificial similarity with structured landmarks.
- *If C1 and C2 had been pooled into an aggregate FPR*, MixVPR would have scored $40\%$, completely masking the fact that it is safe against unseen cities but dangerously hallucinatory on generic textures.

### 3.2 The Foundation Model (DINOv2) Open-Set Collapse
AnyLoc (DINOv2 ViT-S/14) exhibits a well-documented failure mode of self-supervised foundation models in open-world retrieval:
- **Zero Abstention ($0.0\%$ CAR):** Emitted coordinates for 100% of all queries across Track C1, Track C2, and Track D.
- **The Representation Trap:** DINOv2 learns a dense semantic patch embedding space without explicit classification margins or background contrastive boundaries. Unseen natural images (e.g., Mount Fuji or an asphalt road) yield cosine similarities $> 0.70$ against arbitrary index landmarks.
- Without external metric calibration or margin gating, foundation model descriptors cannot distinguish between "semantically coherent imagery" and "geographic identity."

---

## 4. Track D: Adversarial Case Study

Track D introduces 6 real-world deceptive cases designed to exploit visual shortcuts.

| Query ID | Case Description | Target Geolocation | Visual Shortcut Tested | CosPlace Score | Decision under $\tau^*=0.28$ | Outcome |
|---|---|---|---|---|---|---|
| `ADV-01` | St Paul's Cathedral London | London, UK | Baroque dome lookalike to St. Peter's / European domes | $0.245$ | **ABSTAIN** | **True Negative** |
| `ADV-02` | Ponte 25 de Abril Lisbon | Lisbon, Portugal | International Orange suspension bridge lookalike to Golden Gate SF | $0.252$ | **ABSTAIN** | **True Negative** |
| `ADV-03` | Mount Rainier Washington | Washington, USA | Isolated snow-capped stratovolcano lookalike to Mount Fuji | $0.218$ | **ABSTAIN** | **True Negative** |
| `ADV-04` | Paris Gare de Lyon tracks | Paris, France | Overhead rail catenary & track layout lookalike to Berlin Hbf | $0.298$ | **EMIT** | **False Positive** (matched Big Ben / London rail) |
| `ADV-05` | Route 66 Steakhouse Bavaria | Bavaria, Germany | US Highway 66 shield signage situated in rural Germany | $0.187$ | **ABSTAIN** | **True Negative** |
| `ADV-06` | Ludkovice concrete underpass | Moravia, Czechia | Featureless brutalist concrete underpass | $0.211$ | **ABSTAIN** | **True Negative** |

### Forensic Breakdown of ADV-04 (Gare de Lyon False Positive)
- `ADV-04` represents a classic infrastructure decoy. The photograph features heavy overhead electrical catenary wires, multiple railway tracks, and track ballast.
- CosPlace registered a top match against `REF-04` (London Big Ben / Westminster Bridge area) with a score of $0.298$, barely exceeding $\tau^* = 0.28$.
- **Cause:** Linear overhead perspective cues (parallel track rails and catenary grids) share spatial frequency features with bridge suspension cables and Gothic vertical tracery.
- **Remedy for Phase 8:** Integration of semantic segmentation to flag transport infrastructure and local homography verification (ORB/SIFT geometric consistency) to reject candidates that lack planar point-to-point correspondence.

---

## 5. Rejection Dynamics & Margin Gating (CosPlace)

CosPlace succeeds where foundation models fail because of its training objective:
$$\mathcal{L}_{\text{CosFace}} = -\log \frac{\exp(s \cdot (\cos \theta_{y} - m))}{\exp(s \cdot (\cos \theta_{y} - m)) + \sum_{j \ne y} \exp(s \cdot \cos \theta_j)}$$

The angular margin $m$ forces the network to concentrate class intra-variance tightly around prototype vectors while repelling unassigned classes into null regions.

Under the frozen operating threshold $\tau^* = 0.28$:
- **Track C1:** 6 of 7 unseen landmarks fell into the null region ($\text{similarity} < 0.28$), achieving **$85.7\%$ correct abstention**.
- **Track C2:** 7 of 8 non-place textures were rejected, achieving **$87.5\%$ correct abstention**.
- **Track D:** 5 of 6 adversarial lookalikes were rejected, achieving **$83.3\%$ correct abstention**.

---

## 6. Summary & Recommendations for Phase 8

1. **Retain Margin-Trained Descriptors:** CosPlace remains the primary retrieval backbone. Raw foundation models (DINOv2) must never be used for final emission without downstream calibration or verification.
2. **Multi-Stage Gating:**
   - *Stage 1:* Metric VPR retrieval ($\tau^* \ge 0.28$).
   - *Stage 2:* Local feature geometric verification (SIFT/ORB RANSAC inlier count $\ge 12$).
   - *Stage 3:* Semantic sanity check (rejecting landmark assignments when query contains zero architectural features).
3. **Statistical Assurance:** The segregated reporting of $\text{FLR}_{\text{C1}}$, $\text{FER}_{\text{C2}}$, and Track D provides an honest, leak-free foundation for open-world operational deployment.
