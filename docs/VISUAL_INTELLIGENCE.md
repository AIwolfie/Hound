# HOUND — Visual Intelligence Engine

## 1. Introduction

Visual intelligence in HOUND is designed around a single investigative rule: **Observation precedes hypothesis**.

Rather than querying a black-box foundation model to guess where an image was taken, HOUND systematically catalogues the visual, structural, and astronomical evidence in the frame:
1. Physical infrastructure (utility poles, road markings, signboards).
2. Astronomical ephemeris and cast shadows (solar azimuth, elevation, latitude bounds).
3. Textual evidence with exact bounding boxes and script identification.
4. Optical parameters (equivalent 35mm focal length and horizontal field of view).

---

## 2. Evidence Taxonomy

All visual intelligence findings are partitioned according to HOUND's strict four-tier evidence hierarchy:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ READ                                                                   │
│ Direct textual facts extracted via OCR (e.g. "NH 48", "Main St").      │
│ Status: Ground truth observation. Not yet a geographic candidate.      │
├────────────────────────────────────────────────────────────────────────┤
│ OBSERVED                                                               │
│ Direct visual features recognized in the frame:                        │
│ • "Detected 4 utility pole instances (avg conf 0.82)"                   │
│ • "Yellow painted road lane markings detected in lower visual field"   │
│ Likelihood ratio impact on candidates clamped to [0.20, 5.0].          │
├────────────────────────────────────────────────────────────────────────┤
│ INFERRED                                                               │
│ Deductions derived from observations (e.g. "Plate format resembles BC")│
│ Likelihood ratio impact on candidates clamped to [0.33, 3.0].          │
├────────────────────────────────────────────────────────────────────────┤
│ COMPUTED                                                               │
│ Derived via verifiable mathematical or geometric calculations:         │
│ • Solar elevation and azimuth ephemeris                                │
│ • Perspective-n-Point camera resectioning                              │
│ • DEM horizon cross-correlation                                        │
│ ONLY computed evidence backed by an artifact file can execute hard     │
│ geographic exclusions.                                                 │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Infrastructure Recognition Framework

Implemented in `investigators/infrastructure/`:
- **Utility Poles & Vertical Infrastructure:** Uses Hough line transformation with vertical clustering ($|\theta| \le 12^\circ$ from vertical). Detects catenary masts, wooden telephone poles, and metal lighting stanchions.
- **Road Signage Panels:** Geometric contour analysis identifying high-contrast circular, rectangular, and triangular signboards.
- **Roadway Markings:** Color-space segmentation in HSV (white vs. yellow) in the lower half of the visual field. Detects solid and dashed lane dividers, crosswalk markings, and edge lines.
- **Extensibility:** Built on `BaseInfrastructureDetector` to allow lightweight OpenCV classical detectors alongside optional future deep-learning object detectors (YOLO / vision transformers).

---

## 4. Solar Ephemeris & Shadow Analysis

Implemented in `investigators/shadows/solar_analyzer.py`:
- **Astronomical Calculation:** Employs NOAA solar ephemeris algorithms to calculate true solar elevation, solar azimuth (relative to True North), solar declination, and hour angle given timestamp and coordinates.
- **Shadow Geometry:** For a vertical post on level ground, shadow azimuth is $\alpha_{\text{shadow}} = (\alpha_{\text{sun}} + 180^\circ) \bmod 360^\circ$ and shadow length ratio is $L/H = 1 / \tan(\theta_{\text{sun}})$.
- **Unanchored Shadow Detection:** Uses adaptive HSV luminance/saturation thresholding and Principal Component Analysis (PCA) on dark contours to detect primary shadow axes without assuming image-up is geographic north.
- **Explicit Uncertainty:** Every shadow observation carries an angular tolerance ($\pm 20^\circ$) and notes stating whether compass heading calibration is required.

---

## 5. Visual Artifact Visualization

Implemented in `reports/evidence_visualizer.py`:
- `ocr_overlay.jpg`: Color-coded semi-transparent bounding boxes with category pills.
- `infrastructure_overlay.jpg`: Bounding boxes highlighting detected poles, signs, and road markings with confidence scores.
- `shadow_overlay.jpg`: Primary detected shadow vectors with anchor dots.
- `comparison_sheet.jpg`: Multi-variant forensic contact sheet comparing original, CLAHE, grayscale, edge, and sharpened variants.
