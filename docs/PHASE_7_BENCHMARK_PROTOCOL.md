# HOUND — Phase 7: Empirical Benchmark Protocol & Evaluation Standards

**Document ID:** `HOUND-PROTOCOL-PHASE-7`  
**Classification:** Standardized Scientific Evaluation Protocol  
**Author:** AIwolfie  
**Status:** Approved for Implementation  
**Date:** November 13, 2026  
**Target Dataset:** `data/benchmarks/phase_7/`  
**Sample Size:** 50 Stratified Photographic Cases (42 Positive Targets + 8 Deceptive Controls)  

---

## 1. Scientific Principles & Anti-Contamination Mandates

To ensure that HOUND's geolocation performance is rigorously measured without data leakage or reporting bias, all Phase 7 benchmark evaluations must adhere to the following non-negotiable rules:

1. **Photographic Authenticity:** Zero synthetic canvases, zero unit-test mockups, and zero AI-generated images. All positive and negative benchmark items must be genuine camera photographs from documented Creative Commons or Public Domain sources.
2. **Metadata Sanitization:** Every benchmark image must undergo cryptographic verification and automated stripping of all EXIF metadata, GPS tags, camera make/model identifiers, and original timestamps before ingestion.
3. **Filesystem Isolation of Ground Truth:** Ground truth records (WGS-84 coordinates, administrative boundaries, tolerance radii) must reside in a dedicated `ground_truth/` directory that is inaccessible to the agent execution pipeline. Production modules (`agent/`, `core/`, `investigators/`, `storage/`) must not contain import statements or references to ground-truth files.
4. **Zero Reference Index Leakage:** Target benchmark photographs must never be added to the georeferenced reference prototype database, the offline gazetteer cache, or model fine-tuning sets.
5. **Decoupled Metric Reporting:** Metrics must explicitly distinguish between **Candidate Generator Recall** (presence of a correct hypothesis in the candidate ledger) and **Calibrated Agent Prediction** (coordinates actually emitted past the calibration abstention gate).

---

## 2. 50-Case Geographically Diverse Dataset Specification

The Phase 7 benchmark suite expands evaluation from 12 to 50 rigorously curated photographs, stratified across **all 7 continents**, 5 distinct environment classes, 5 atmospheric/lighting conditions, and 4 camera viewpoints:

```
Total Evaluation Corpus: 50 Cases
├── Positive Geographic Targets: 42 Cases
│   ├── Urban Environments & Skylines: 10 Cases
│   ├── Rural Farmland, Steppe & Savannah: 8 Cases
│   ├── Natural Landscapes & High-Relief Topography: 10 Cases
│   ├── Transportation Infrastructure (Rail, Bridges, Ports): 8 Cases
│   └── Historic & Vernacular Architecture: 6 Cases
└── Negative Controls (Unlocalizable & Ambiguous): 8 Cases
    ├── Windowless Indoor Rooms: 2 Cases
    ├── Close-Up Macro Textures (Bark, Asphalt): 2 Cases
    ├── Featureless Overcast Sky / Clouds: 1 Case
    ├── Ambiguous Open Ocean / Water: 1 Case
    └── Ambiguous Windblown Desert Sand Dunes: 2 Cases
```

### 2.1. Geographic Distribution Matrix (42 Positive Targets)

| Continent | Target Count | Representative Sample Locations |
|:---|:---:|:---|
| **Africa** | 6 | Table Mountain (ZA), Victoria Falls (ZW/ZM), Giza Plateau (EG), Djenné Mosque (ML), Atlas Mountains (MA), Rift Valley (KE) |
| **Asia** | 8 | Shibuya Crossing (JP), Taj Mahal (IN), Marina Bay (SG), Pamir Highway (TJ), Huangshan (CN), Petra (JO), Shinjuku Station (JP), Sigiriya (LK) |
| **Europe** | 9 | Berlin Hauptbahnhof (DE), Matterhorn (CH), Mont Saint-Michel (FR), Giant's Causeway (UK), Acropolis (GR), Tromsø Port (NO), Neuschwanstein (DE), Charles Bridge (CZ), Val d'Orcia (IT) |
| **North America** | 8 | Bixby Bridge (US), Banff National Park (CA), Route 66 Seligman (US), Chichen Itza (MX), Golden Gate (US), Zabriskie Point (US), Manhattan Skyline (US), Copper Canyon (MX) |
| **South America** | 5 | Machu Picchu (PE), Iguazu Falls (AR/BR), Salar de Uyuni (BO), Torres del Paine (CL), Christ the Redeemer (BR) |
| **Oceania** | 4 | Sydney Opera House (AU), Milford Sound (NZ), Uluru / Kata Tjuta (AU), Twelve Apostles (AU) |
| **Antarctica** | 2 | McMurdo Station coastal apron, Lemaire Channel glacial peaks |

### 2.2. Environmental & Viewpoint Stratification

Every image is tagged in `metadata/manifest.json` across three orthogonal axes:

1. **Atmospheric / Lighting Condition:**
   - `direct_noon_sun`: Overhead shadows, high dynamic range.
   - `golden_hour_low_sun`: Long cast shadow vectors for ephemeris validation.
   - `night_artificial_lighting`: Sodium/LED streetlamps, commercial neon.
   - `overcast_diffuse`: Zero cast shadows, low contrast.
   - `winter_snow_cover`: High albedo, obscured ground textures.

2. **Optical Viewpoint:**
   - `pedestrian_street_level`: Standard human eye level (1.5m - 1.8m), perspective distortion.
   - `telephoto_detail`: Narrow FOV ($< 25^\circ$), zoomed architectural features.
   - `wide_angle_panorama`: Ultra-wide FOV ($> 85^\circ$), horizon curvature.
   - `elevated_vantage_point`: Balcony, hilltop, or overlook viewing urban/terrain expanse.

3. **Difficulty Tier:**
   - `EASY`: High-profile global landmark with distinct architectural geometry or visible signage.
   - `MEDIUM`: Regional infrastructure, distinctive terrain silhouette, or standard highway corridor.
   - `HARD`: Unindexed rural road, non-distinctive mountain ridge, or multi-national vernacular architecture.
   - `NEGATIVE_CONTROL`: Intrinsically unlocalizable scene.

---

## 3. Standardized Metric Formulations & Closed Denominators

To permanently resolve the reporting flaw identified in Phase 6, Phase 7 establishes standard closed-denominator formulas:

### 3.1. Candidate Generation Recall (Pre-Calibration)
Measures the ability of the agent's multi-modal tools (OCR, VPR, OSM, Solar, DEM) to generate hypotheses that capture the target within the search tree:

$$\text{Recall@}K = \frac{1}{N_{\text{pos}}} \sum_{i=1}^{N_{\text{pos}}} \mathbb{I}\left( \min_{j \le K} d(\mathbf{cand}_{i,j}, \mathbf{gt}_i) \le \tau_i \right)$$

Where $\tau_i$ is the target tolerance radius (default: $25.0\text{ km}$ for regional, $2.0\text{ km}$ for landmark). Evaluated at $K \in \{1, 5, 20\}$.

### 3.2. Calibrated Coverage
Measures the proportion of positive cases where the agent possessed sufficient calibrated certainty ($P \ge 0.40$, non-falsified) to emit coordinates:

$$\text{Coverage} = \frac{N_{\text{emitted}}}{N_{\text{pos}}} = \frac{\sum_{i=1}^{N_{\text{pos}}} \mathbb{I}(\text{not } \mathbf{r}_i.\text{abstention\_recommended})}{N_{\text{pos}}}$$

### 3.3. Calibrated Selective Accuracy
Measures the accuracy strictly over instances where coordinates were emitted to the user:

$$\text{Selective Accuracy} = \begin{cases} \frac{\sum_{i \in \text{Emitted}} \mathbb{I}(d(\mathbf{pred}_i, \mathbf{gt}_i) \le \tau_i)}{N_{\text{emitted}}} & \text{if } N_{\text{emitted}} > 0 \\ \text{N/A} & \text{if } N_{\text{emitted}} = 0 \end{cases}$$

### 3.4. Deployed End-to-End Accuracy
Measures overall system success under operational calibration gating:

$$\text{End-to-End Accuracy} = \frac{1}{N_{\text{pos}}} \sum_{i=1}^{N_{\text{pos}}} \mathbb{I}(\text{not } \mathbf{r}_i.\text{abstention\_recommended} \land d(\mathbf{pred}_i, \mathbf{gt}_i) \le \tau_i)$$

### 3.5. Honest Abstention Precision on Negative Controls
Measures protection against false localization claims:

$$\text{Negative Abstention Rate} = \frac{1}{N_{\text{neg}}} \sum_{j=1}^{N_{\text{neg}}} \mathbb{I}(\mathbf{r}_j.\text{abstention\_recommended} \lor \mathbf{r}_j.\text{final\_status} == \text{"insufficient\_evidence"})$$

$$\text{False Positive Rate (FPR)} = 1.0 - \text{Negative Abstention Rate}$$

Target for deployment: $\text{FPR} = 0.0\%$.

---

## 4. Execution Harness & Reproducibility Standard

The Phase 7 benchmark must be executable via a single deterministic CLI command:

```bash
# Run complete Phase 7 evaluation in offline mode
python benchmarks/phase_7_benchmark.py --dataset data/benchmarks/phase_7 --offline

# Run with online OSM and Nominatim validation
python benchmarks/phase_7_benchmark.py --dataset data/benchmarks/phase_7 --online --rate-limit 1.5
```

### Required JSON Output Artifact
Every benchmark execution generates `data/benchmarks/phase_7/reports/phase_7_benchmark_report.json` containing:
1. Environment metadata (Python version, PyTorch version, CUDA status, timestamp, Git commit SHA).
2. Global summary metrics table (Recall@1, Recall@5, Coverage, Selective Accuracy, End-to-End Accuracy, Abstention Rate, FPR).
3. Calibration metrics (Brier score, ECE, Reliability diagram bin counts).
4. Full execution traces per case (iteration count, tool invocation list, execution duration, candidate progression, calibration breakdown).
