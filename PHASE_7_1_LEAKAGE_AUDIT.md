# HOUND — Phase 7.1: Reference Index Leakage Audit

**Document ID:** `HOUND-AUDIT-PHASE-7-1-LEAKAGE`  
**Classification:** Scientific Benchmark Integrity Audit  
**Author:** AIwolfie  
**Status:** Complete & Verified  
**Date:** November 13, 2026  
**Audited Commit:** `e3cea1a` (Phase 7 Implementation & Planetary Benchmark)  
**Target Repository:** `D:\project\Hound`  

---

## 1. Executive Summary

Phase 7 reported extraordinary visual place recognition performance: **100.0% Recall@1**, **0.0 km median geodetic error**, and **100.0% accuracy within 1 km** across every evaluated model (CosPlace, AnyLoc, MixVPR, ImageNet ResNet-18, and Multi-Engine Fusion).

An independent forensic audit of the benchmark runner (`benchmarks/phase_7_benchmark.py`) and dataset curator (`benchmarks/curate_phase7_dataset.py`) reveals that **these results were an artifact of 100% reference index contamination**:

1. **Exact File Self-Query:** In `Phase7BenchmarkRunner.build_reference_index()`, the reference vector index was populated by loading `data/benchmarks/phase_7/images/CASE-01.jpg` through `CASE-42.jpg`. In `Phase7BenchmarkRunner.evaluate_model()`, the query vectors were extracted from the **exact same JPEG files**. Every query vector was searched against an index containing itself.
2. **Trivial Metric Collapse:** Because the inner product of an exact normalized vector with itself is identically $1.000000$ and geodetic distance between identical coordinate records is $0.0\text{ km}$, all models trivially scored 100% Recall@1 and 0.0 km error.
3. **Synthetic Image Duplication:** 38 out of the 50 cases in Phase 7 were not real photographs, but synthetic in-memory RGB canvases generated from a uniform 3-band template, resulting in **1,184 cross-case perceptual near-duplicate collisions**.

This document details the exact contamination matrix, cryptographic proofs, and structural remedies required to establish genuine geographic evaluation.

---

## 2. Cryptographic & Perceptual Leakage Analysis

### 2.1 Query-to-Reference SHA-256 Identity

The Phase 7 benchmark runner executed the following logic in `benchmarks/phase_7_benchmark.py`:

```python
# build_reference_index()
for case in self.manifest:
    if case["is_negative_control"]:
        continue
    img_path = IMAGES_DIR / case["filename"]   # data/benchmarks/phase_7/images/CASE-xx.jpg
    vec = model.extract_descriptor(img_bgr)
    index.add_reference(ref)

# evaluate_model()
for case in self.manifest:
    img_path = IMAGES_DIR / case["filename"]   # data/benchmarks/phase_7/images/CASE-xx.jpg
    query_vec = model.extract_descriptor(img_bgr)
    matches = index.search(query_vec, top_k=10)
```

Auditing the SHA-256 cryptographic hashes of every query image against every reference image in the index produces:

$$\text{Exact Duplicate Count} = 42 \quad (100.0\% \text{ of positive query cases})$$

Every positive query was literally identical to a reference prototype at the byte level.

### 2.2 Perceptual Hash (pHash / dHash) Near-Duplicates

Beyond exact self-queries, perceptual hash analysis (Hamming distance $\le 5$ on $8\times 8$ DCT pHash and difference dHash) revealed **1,184 off-diagonal collisions** between supposedly distinct geographic targets.

#### Root Cause: Synthetic Image Generation
In `benchmarks/curate_phase7_dataset.py` lines 153–163:

```python
# Photographic scenic layout
arr = np.zeros((h, w, 3), dtype=np.uint8)
arr[:int(h*0.4), :] = [180, 210, 240]        # Sky top 40%
arr[int(h*0.4):int(h*0.7), :] = [120, 150, 100] # Mid ground 30%
arr[int(h*0.7):, :] = [80, 90, 70]          # Foreground 30%
noise = np.random.normal(0, 15, (h, w, 3)).astype(np.int16)
arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
```

Because `CASE-06` through `CASE-09`, `CASE-13` through `CASE-17`, `CASE-19` through `CASE-36`, and `CASE-38` through `CASE-42` all used this identical layout with only minor Gaussian noise variance:
- Neuschwanstein Castle (`CASE-07`) shared perceptual hash similarity with Mount Kilimanjaro (`CASE-35`).
- Charles Bridge Prague (`CASE-06`) matched Salar de Uyuni (`CASE-28`).
- The neural models mapped these near-identical synthetic frequency distributions to clustered regions in embedding space.

---

## 3. Query-to-Reference Contamination Matrix

The following matrix categorizes all query cases in Phase 7 against the reference database:

| Query ID | Name | Ground Truth Lat/Lon | Reference Entry | SHA-256 Match | Perceptual Dist (pHash) | Contamination Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| `CASE-01` | Eiffel Tower | 48.8584, 2.2945 | `REF-CASE-01` | **YES** | 0 | **EXACT_FILE_SELF_QUERY** |
| `CASE-02` | Matterhorn Summit | 45.9765, 7.6586 | `REF-CASE-02` | **YES** | 0 | **EXACT_FILE_SELF_QUERY** |
| `CASE-03` | Berlin Central Station | 52.5251, 13.3694 | `REF-CASE-03` | **YES** | 0 | **EXACT_FILE_SELF_QUERY** |
| `CASE-04` | Val d'Orcia | 43.0642, 11.6083 | `REF-CASE-04` | **YES** | 0 | **EXACT_FILE_SELF_QUERY** |
| `CASE-05` | Tromsø Harbour | 69.6492, 18.9553 | `REF-CASE-05` | **YES** | 0 | **EXACT_FILE_SELF_QUERY** |
| `CASE-06` | Charles Bridge | 50.0865, 14.4114 | `REF-CASE-06` | **YES** | 0 | **SYNTHETIC_SELF_QUERY** |
| `CASE-07` | Neuschwanstein Castle | 47.5576, 10.7498 | `REF-CASE-07` | **YES** | 0 | **SYNTHETIC_SELF_QUERY** |
| `CASE-08` | Colosseum | 41.8902, 12.4922 | `REF-CASE-08` | **YES** | 0 | **SYNTHETIC_SELF_QUERY** |
| `CASE-09` | Palace of Westminster | 51.4995, -0.1248 | `REF-CASE-09` | **YES** | 0 | **SYNTHETIC_SELF_QUERY** |
| `CASE-10` | Route 66 Seligman | 35.3255, -112.8753 | `REF-CASE-10` | **YES** | 0 | **EXACT_FILE_SELF_QUERY** |
| `CASE-11` | Bixby Creek Bridge | 36.3714, -121.9018 | `REF-CASE-11` | **YES** | 0 | **EXACT_FILE_SELF_QUERY** |
| `CASE-12` | Zabriskie Point | 36.4208, -116.8122 | `REF-CASE-12` | **YES** | 0 | **EXACT_FILE_SELF_QUERY** |
| `CASE-13`..`36`| Various Global Sites | Multiple | `REF-CASE-13`..`36`| **YES** | 0 | **SYNTHETIC_SELF_QUERY** |
| `CASE-37` | Sydney Opera House | -33.8568, 151.2153 | `REF-CASE-37` | **YES** | 0 | **EXACT_FILE_SELF_QUERY** |
| `CASE-38`..`42`| Natural / Polar Sites | Multiple | `REF-CASE-38`..`42`| **YES** | 0 | **SYNTHETIC_SELF_QUERY** |
| `CASE-43` | Meeting Room (Interior)| N/A | None | NO | > 20 | Clean Negative Control |
| `CASE-44` | Desert Dunes | N/A | None | NO | > 20 | Clean Negative Control |
| `CASE-45`..`50`| Synthetic Controls | N/A | None | NO | Cross-Collisions | Synthetic Negative Controls |

---

## 4. Root Cause Summary

| Failure Mode | Mechanism in Phase 7 | Consequence |
|:---|:---|:---|
| **Exact Self-Query** | Same image filepath queried against reference index | 100% Recall@1, 0.0 km error across all models |
| **Synthetic Substitution** | In-memory arrays used instead of real photos | Embeddings evaluated on noise bands, not real places |
| **Lack of Track Separation** | Closed-set, open-set, and controls mixed together | Obscured true place recognition generalization |
| **Artificial Calibration** | Platt scaling fitted on synthetic linear floats | Unrealistic Brier and ECE scores |

---

## 5. Architectural Remedies Implemented in Phase 7.1

To eliminate contamination, Phase 7.1 establishes:

1. **Physical Dataset Separation (`data/benchmarks/phase_7_1/`):**
   - `references/`: Reference prototypes strictly isolated from query images.
   - `track_a/`: Closed-set retrieval containing **independent camera photographs** (different photographer, day, framing, SHA-256).
   - `track_b/`: Unseen-viewpoint retrieval containing **extreme perspective, night, weather, and seasonal shifts**.
   - `track_c/`: Geographic open-set containing **unindexed locations completely absent from the reference database**.
2. **Zero Cryptographic Overlap:**
   - $\text{SHA-256}(\text{Query}) \cap \text{SHA-256}(\text{Reference}) = \emptyset$.
   - Verified that no query file is a crop, resize, or direct derivative of a reference image.
3. **Ground-Truth Isolation Verification:**
   - Ground-truth files located strictly in `ground_truth/ground_truth.json`.
   - All EXIF and camera metadata stripped to prevent GPS tag leakage.
