# HOUND — PHASE 7.2 DATASET SPLIT & ISOLATION AUDIT

**Author:** AIwolfie  
**Auditor / Operator:** RIOT  
**Project:** HOUND — Autonomous Visual Geolocation & OSINT Agent  
**Date:** 2026-11-13  
**Status:** COMPLETE & INDEPENDENTLY AUDITED  
**Workspace:** `D:/project/Hound`

---

## 1. Executive Summary & Objective

In Phase 7.1, an independent forensic audit exposed exact-file reference leakage in the original Phase 7 benchmark (where test query images were identical copies of reference index images, resulting in an artificial $100\%$ Recall@1 and $0.0\text{ km}$ median error). Phase 7.1 rectified this by establishing three separate tracks (Closed-set Track A, Viewpoint-shift Track B, and Open-set Track C).

Phase 7.2 formalizes **strict scientific validation protocols** through a mathematically isolated **Three-Way Dataset Partitioning Scheme**:
1. **Reference Prototype Index ($\mathcal{R}$):** 10 curated geographic prototypes defining the searchable knowledge base.
2. **Calibration / Tuning Partition ($\mathcal{D}_{\text{calib}}$):** 13 independent queries used exclusively to fit probability calibrators (Platt scaling / temperature) and select decision thresholds ($\tau^*$).
3. **Untouched Final Test Partition ($\mathcal{D}_{\text{test}}$):** 41 strictly isolated evaluation queries across five tracks (Track A, Track B, Track C1, Track C2, and Track D).

Under strict scientific integrity rules:
$$\mathcal{R} \cap \mathcal{D}_{\text{calib}} = \emptyset, \quad \mathcal{R} \cap \mathcal{D}_{\text{test}} = \emptyset, \quad \mathcal{D}_{\text{calib}} \cap \mathcal{D}_{\text{test}} = \emptyset$$

No parameter, threshold, hyperparameter, or calibration model was tuned, selected, or modified using any image in $\mathcal{D}_{\text{test}}$.

---

## 2. Partition Cryptographic Verification

All 64 photographs across all three partitions were subjected to cryptographic SHA-256 hashing post-sanitization (EXIF stripped, re-encoded).

| Partition | Symbol | Total Items | Unique SHA-256 Hashes | Duplicate Hashes Within Partition | Overlap With Other Partitions |
|---|---|---|---|---|---|
| **Reference Prototypes** | $\mathcal{R}$ | 10 | 10 | 0 | **0 (Zero)** |
| **Calibration Set** | $\mathcal{D}_{\text{calib}}$ | 13 | 13 | 0 | **0 (Zero)** |
| **Final Test Partition** | $\mathcal{D}_{\text{test}}$ | 41 | 41 | 0 | **0 (Zero)** |

### Pairwise Disjointness Matrix

$$\begin{pmatrix}
|\mathcal{R} \cap \mathcal{R}| = 10 & |\mathcal{R} \cap \mathcal{D}_{\text{calib}}| = 0 & |\mathcal{R} \cap \mathcal{D}_{\text{test}}| = 0 \\
|\mathcal{D}_{\text{calib}} \cap \mathcal{R}| = 0 & |\mathcal{D}_{\text{calib}} \cap \mathcal{D}_{\text{calib}}| = 13 & |\mathcal{D}_{\text{calib}} \cap \mathcal{D}_{\text{test}}| = 0 \\
|\mathcal{D}_{\text{test}} \cap \mathcal{R}| = 0 & |\mathcal{D}_{\text{test}} \cap \mathcal{D}_{\text{calib}}| = 0 & |\mathcal{D}_{\text{test}} \cap \mathcal{D}_{\text{test}}| = 41
\end{pmatrix}$$

**Audit Conclusion:** Pairwise hash intersection is **identically zero**. There is no image reuse between the index, the calibration partition, and the test partition.

---

## 3. Reference Prototype Index ($\mathcal{R}$)

The reference index $\mathcal{R}$ consists of 10 landmark prototypes indexed into the vector database.

| Ref ID | Landmark Name | Country | Latitude | Longitude | Tolerance | SHA-256 Prefix | License |
|---|---|---|---|---|---|---|---|
| `REF-01` | Eiffel Tower, Paris | France | 48.8584 | 2.2945 | 5.0 km | `f9479b18...` | CC BY-SA 4.0 |
| `REF-02` | Colosseum, Rome | Italy | 41.8902 | 12.4922 | 5.0 km | `ec76fbc0...` | CC BY-SA 3.0 |
| `REF-03` | Brandenburg Gate, Berlin | Germany | 52.5163 | 13.3777 | 5.0 km | `f047ff95...` | CC BY-SA 4.0 |
| `REF-04` | Big Ben / Elizabeth Tower, London | UK | 51.5007 | -0.1246 | 5.0 km | `76f7f2b1...` | CC BY-SA 4.0 |
| `REF-05` | Sagrada Familia, Barcelona | Spain | 41.4036 | 2.1744 | 5.0 km | `39f37cbb...` | CC BY-SA 4.0 |
| `REF-06` | Sydney Opera House, Sydney | Australia | -33.8568 | 151.2153 | 5.0 km | `5ba16499...` | CC BY-SA 4.0 |
| `REF-07` | Taj Mahal, Agra | India | 27.1751 | 78.0421 | 5.0 km | `da0e5276...` | CC BY-SA 4.0 |
| `REF-08` | Statue of Liberty, New York | USA | 40.6892 | -74.0445 | 5.0 km | `324208a1...` | CC BY-SA 4.0 |
| `REF-09` | Parthenon, Athens | Greece | 37.9715 | 23.7266 | 5.0 km | `8c6fc7d9...` | CC BY-SA 4.0 |
| `REF-10` | Tower Bridge, London | UK | 51.5055 | -0.0754 | 5.0 km | `132be04f...` | CC BY-SA 4.0 |

---

## 4. Calibration Partition ($\mathcal{D}_{\text{calib}}$)

The calibration partition contains 13 photographic queries designed to calibrate score-to-probability mapping and evaluate operating thresholds:
- **5 Positive In-Set Cases (`CALIB_POS-01` to `CALIB_POS-05`):** Independent photographs of index landmarks (`REF-01` to `REF-05`).
- **4 Open-Set Place Cases (`CALIB_OPEN-01` to `CALIB_OPEN-04`):** Real photographs of world landmarks absent from $\mathcal{R}$ (Leaning Tower of Pisa, St. Mark's Basilica, Christ the Redeemer, Golden Gate Bridge).
- **4 Non-Place Negative Cases (`CALIB_NEG-01` to `CALIB_NEG-04`):** Negative controls without geographic features (indoor drywall, macro asphalt, blank overcast sky, blur).

| Calib ID | Category | Target Location | True Ref | Latitude | Longitude | Expected Action | SHA-256 Prefix |
|---|---|---|---|---|---|---|---|
| `CALIB_POS-01` | Known Place | Eiffel Tower (Champ de Mars view) | `REF-01` | 48.8584 | 2.2945 | EMIT (`REF-01`) | `aa935661...` |
| `CALIB_POS-02` | Known Place | Colosseum (Night exterior) | `REF-02` | 41.8902 | 12.4922 | EMIT (`REF-02`) | `fe5fb0fa...` |
| `CALIB_POS-03` | Known Place | Brandenburg Gate (East side) | `REF-03` | 52.5163 | 13.3777 | EMIT (`REF-03`) | `073e04e9...` |
| `CALIB_POS-04` | Known Place | Big Ben (Westminster Bridge view) | `REF-04` | 51.5007 | -0.1246 | EMIT (`REF-04`) | `b99e74ce...` |
| `CALIB_POS-05` | Known Place | Sagrada Familia (Nativity facade) | `REF-05` | 41.4036 | 2.1744 | EMIT (`REF-05`) | `cf619a9a...` |
| `CALIB_OPEN-01`| Unseen Place | Leaning Tower of Pisa, Italy | None | 43.7230 | 10.3966 | ABSTAIN | `895c2b3e...` |
| `CALIB_OPEN-02`| Unseen Place | St. Mark's Basilica, Venice | None | 45.4344 | 12.3397 | ABSTAIN | `5d6b42b1...` |
| `CALIB_OPEN-03`| Unseen Place | Christ the Redeemer, Rio | None | -22.9519 | -43.2105 | ABSTAIN | `f1a8c886...` |
| `CALIB_OPEN-04`| Unseen Place | Golden Gate Bridge, SF | None | 37.8199 | -122.4783 | ABSTAIN | `ae4e432a...` |
| `CALIB_NEG-01` | Non-Place | Painted drywall indoor texture | None | None | None | ABSTAIN | `f7347a8a...` |
| `CALIB_NEG-02` | Non-Place | Macro coarse asphalt texture | None | None | None | ABSTAIN | `07f59d4c...` |
| `CALIB_NEG-03` | Non-Place | Featureless overcast gray sky | None | None | None | ABSTAIN | `734e9e03...` |
| `CALIB_NEG-04` | Non-Place | Heavy uniform Gaussian blur | None | None | None | ABSTAIN | `0ca0956d...` |

---

## 5. Final Untouched Test Partition ($\mathcal{D}_{\text{test}}$)

The test partition comprises 41 queries divided into 5 strictly isolated evaluation tracks:

```
data/benchmarks/phase_7_2/test/
├── track_a/   (10 queries: QA-01 to QA-10 — Closed-set independent real photos)
├── track_b/   (10 queries: QB-01 to QB-10 — Unseen viewpoints & severe environmental shifts)
├── track_c1/  ( 7 queries: QC1-01 to QC1-07 — Genuine unseen real-world landmarks absent from index)
├── track_c2/  ( 8 queries: QC2-01 to QC2-08 — Non-place negative controls)
└── track_d/   ( 6 queries: ADV-01 to ADV-06 — Adversarial challenging lookalikes & decoys)
```

### 5.1 Track A: Closed-Set Independent Real Photographs (10 queries)
Independent real photographs of the 10 index locations taken by different photographers under standard conditions:
- `QA-01`: Eiffel Tower from Pont de Bir-Hakeim (`f0249ba9...`)
- `QA-02`: Colosseum south facade (`a3746a81...`)
- `QA-03`: Brandenburg Gate daytime frontal (`1b0928ff...`)
- `QA-04`: Big Ben / Palace of Westminster (`8dc38151...`)
- `QA-05`: Sagrada Familia Passion facade (`d946d87e...`)
- `QA-06`: Sydney Opera House harbor view (`9d273752...`)
- `QA-07`: Taj Mahal reflecting pool view (`83be605e...`)
- `QA-08`: Statue of Liberty frontal view (`d7f8a706...`)
- `QA-09`: Parthenon south colonnade (`2ea60c91...`)
- `QA-10`: Tower Bridge walkway view (`7e891395...`)

### 5.2 Track B: Unseen Viewpoints & Severe Environmental Shifts (10 queries)
Challenging viewpoints of the 10 index locations featuring extreme angles, nocturnal lighting, seasonal weather, or interior perspectives:
- `QB-01`: Eiffel Tower from Champ de Mars under snow (`6aa17088...`)
- `QB-02`: Colosseum interior arena floor perspective (`c1d4715f...`)
- `QB-03`: Brandenburg Gate dramatic nocturnal illumination (`ea862025...`)
- `QB-04`: Big Ben architectural detail / clock face close-up (`80b3d81b...`)
- `QB-05`: Sagrada Familia interior hyperbolic vaults (`c98a39a0...`)
- `QB-06`: Sydney Opera House tile shell texture macro (`9cb4df31...`)
- `QB-07`: Taj Mahal western mosque perspective (`1f4c718b...`)
- `QB-08`: Statue of Liberty crown & torch close crop (`631682ea...`)
- `QB-09`: Parthenon eastern pediment detail (`c5ce110c...`)
- `QB-10`: Tower Bridge bascule opening at twilight (`918f0376...`)

### 5.3 Track C1: Unseen Geographic Locations (7 queries)
Real photographs of prominent world landmarks that are **completely absent** from the reference database $\mathcal{R}$:
- `QC1-01`: Neuschwanstein Castle, Bavaria, Germany (`1a84f3df...`)
- `QC1-02`: Petra Treasury (Al-Khazneh), Jordan (`7702f306...`)
- `QC1-03`: Machupicchu Citadel, Peru (`a2f4ab27...`)
- `QC1-04`: Saint Basil's Cathedral, Moscow, Russia (`69466e3f...`)
- `QC1-05`: Sydney Harbour Bridge, Australia (`6889eb06...`)
- `QC1-06`: Mount Fuji, Japan (`79eb17e1...`)
- `QC1-07`: Pyramids of Giza, Egypt (`745f616e...`)

*Ground-Truth Expectation:* System must report **ABSTAIN**. Any emitted coordinate is evaluated as a **False Localization ($\text{FLR}_{\text{C1}}$)**.

### 5.4 Track C2: Non-Place Negative Controls (8 queries)
Images devoid of geographic discriminators, tested to verify robustness against hallucinatory emissions:
- `QC2-01`: Macro coarse asphalt texture (`481d6aa3...`)
- `QC2-02`: Indoor painted drywall with electrical outlet (`9d424b94...`)
- `QC2-03`: Featureless uniform overcast sky (`583ef5fc...`)
- `QC2-04`: Open water surface with gentle wave chop (`0105ff7d...`)
- `QC2-05`: Synthetic white noise with Gaussian blur (`dfd5e1dc...`)
- `QC2-06`: Corrugated shipping container metal panel (`30a84e27...`)
- `QC2-07`: Smooth polished concrete floor indoors (`da23a232...`)
- `QC2-08`: Generic lush grass lawn without landmarks (`9952fe21...`)

*Ground-Truth Expectation:* System must report **ABSTAIN**. Any emitted coordinate is evaluated as a **False Emission ($\text{FER}_{\text{C2}}$)**.

### 5.5 Track D: Adversarial Challenging Cases (6 queries)
Hard negative and decoy cases exhibiting visual feature mimicry designed to deceive naive VPR models:
- `ADV-01`: St Paul's Cathedral London — Baroque dome lookalike to St. Peter's / European domes (`f8373b98...`)
- `ADV-02`: Ponte 25 de Abril Lisbon — Red suspension bridge lookalike to Golden Gate SF (`3a48e7ba...`)
- `ADV-03`: Mount Rainier Washington — Symmetrical snow-capped stratovolcano lookalike to Mount Fuji (`34ec62be...`)
- `ADV-04`: Paris Gare de Lyon tracks — Overhead rail catenary & platform layout lookalike to Berlin Hbf (`9b109033...`)
- `ADV-05`: Route 66 Steakhouse Bavaria — US highway shield signage located in Germany (`2518e1ef...`)
- `ADV-06`: Ludkovice concrete underpass — Brutalist concrete underpass without skyline features (`883eb268...`)

*Ground-Truth Expectation:* System must report **ABSTAIN**.

---

## 6. Pre-Computation and Freeze Protocol

1. **Threshold Derivation ($\tau^*$):**
   - Derived exclusively on $\mathcal{D}_{\text{calib}}$ by maximizing the objective function:
     $$J(\tau) = \text{SelectiveAcc}(\tau) - \text{FER}(\tau)$$
   - Optimal operating threshold derived: $\tau^* = 0.28$.
2. **Probability Calibrator Fitting:**
   - Logistic regression (Platt scaling) fitted exclusively on $(s_i, y_i) \in \mathcal{D}_{\text{calib}}$:
     $$\sigma(z) = \frac{1}{1 + \exp(-(1.5988 \cdot s - 1.2709))}$$
3. **Freezing:**
   - Both $\tau^*$ and Platt parameters were stored in `data/benchmarks/phase_7_2/reports/phase_7_2_benchmark_report.json` under `calibration_freezing`.
   - $\mathcal{D}_{\text{test}}$ was subsequently evaluated in a single forward pass without alteration.

---

## 7. Audit Certification

This audit certifies that:
1. Ground truth coordinates, geographic labels, and query filenames never entered model inference or feature extraction pipelines.
2. The three-way dataset split is cryptographically disjoint with zero SHA-256 hash overlap.
3. The evaluation on $\mathcal{D}_{\text{test}}$ represents a clean, unbiased out-of-sample benchmark of visual place recognition and open-set calibration.
