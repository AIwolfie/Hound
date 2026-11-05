# REFERENCE GEOLOCATION RESEARCH AUDIT

**Date:** 2026-11-05  
**Auditor:** Lead Architect, HOUND Development Team  
**Scope:** Evaluation of upstream open-source GIS and visual geolocation research algorithms  
**License Standards:** MIT License (code), CC BY-SA 4.0 (derived reference tables)  
**Attribution:** Reference algorithms by Oldcircle; Wikipedia contributors  

---

## 1. Executive Summary

Upstream open-source geolocation research demonstrates the feasibility of evidence-based image localization (geolocation and chronolocation). Rather than relying on black-box visual guesses, structured investigative systems separate analysis into three explicit layers:
1. **Decision Layer:** Structured hypothesis management tracking clues, likelihood ratios, exclusions, and rank calculation.
2. **Perception & Computation Layer:** Mathematical utilities performing EXIF parsing, image preprocessing, multi-scale OCR, reverse image search, sun/shadow geometry, OpenStreetMap Overpass mining, digital elevation model (DEM) horizon/ridge matching, perspective camera pose estimation, and satellite/street-view ranking.
3. **Judgment Layer:** An autonomous agent or human analyst selecting among machine-ranked top candidates, evaluating qualitative scene clues, and closing verification loops.

While reference implementations provide proven mathematical, geometric, and GIS algorithms, standalone script toolkits typically lack:
- A unified, asynchronous Python API package architecture.
- Robust cross-platform console encoding resilience (Windows default `cp1252` stdout crashes on non-ASCII place names without `PYTHONUTF8=1`).
- Multi-provider fallback orchestration (e.g., if Overpass or search engines time out, external pipelines stall or require human intervention).
- Enterprise persistence, relational database storage, audit trail logging, and strict contradictory-evidence engines.

HOUND was designed from the ground up as a native, modular, production-grade autonomous agent architecture that implements these intelligence capabilities independently.

---

## 2. Environment & Dependency Audit

| Component | Status | Details |
|---|---|---|
| **Python Runtime** | Available | Python 3.13.7 (Verified) |
| **Package Manager** | Available | `uv` 0.10.2 (Verified) |
| **Network Client** | Available | System `curl` available |
| **Browser Engine** | Available | Playwright Chromium headless automation |
| **OCR Backend** | Configured | RapidOCR (ONNX runtime cross-platform fallback) |
| **Vision / ML Models** | Lazy-Loaded | PyTorch (CUDA / CPU), torchvision, Hugging Face Hub (CLIP, DINOv2). Weights downloaded on first use. |
| **GIS / Elevation Tiles**| Available | AWS Terrarium elevation tiles (HTTP 200 confirmed) |
| **Tile Mosaic Servers** | Available | Google Satellite / OSM Tile endpoints reachable |
| **Search Engines** | Mixed | Yandex Images HTTP 200; Baidu Image search high latency |
| **OpenStreetMap** | High Load | Public Overpass mirrors periodically return HTTP 406/504/timeouts; Nominatim requires strict user agents |

---

## 3. Algorithm Inventory & Architecture Mapping

| Capability | Methodology & Mathematical Basis | Implementation in HOUND |
|---|---|---|
| **Forensic Diagnostics** | System health, environment, encoding, and network diagnostic checks. | Native `hound doctor` and `hound geo doctor` CLI commands. |
| **EXIF & Optics Analysis** | Extracts GPS coordinates, timestamps, camera make/model, focal length, 35mm equivalent, heading. | Native `investigators/metadata/exif_analyzer.py` with typed provenance models. |
| **Image Preprocessing** | Edge crops, zoomed details, perspective normalization, and contrast enhancement. | Native `core/image_preprocessor.py` for direct in-memory `PIL.Image` manipulation. |
| **Solar Chronolocation** | Solar ephemeris, declination, and azimuth from shadows and timestamps. | Native `investigators/shadows/solar_analyzer.py` with strict physical ceiling checks. |
| **OSM Infrastructure Mining** | Overpass QL querying for highways, railways, bridges, powerlines, and waterways. | Resilient native client in `integrations/openstreetmap/` with mirror failover and persistent SQLite caching. |
| **Gazetteer & Geocoding** | Administrative boundaries and location resolution via Nominatim. | Native client in `integrations/geocoding/` with SQLite caching and polygon boundary parsing. |
| **Terrain & DEM Analysis** | Terrarium tile decoding and ray-cast horizon synthesis with curvature and refraction. | Native client in `integrations/elevation/` with continuous bilinear sub-pixel interpolation. |
| **Ridge Matching** | Cross-correlation optimization matching observed visual skylines to synthetic DEM horizons. | Native `RidgeMatcher` in `integrations/elevation/ridge_matcher.py` with normalized RMSE scoring. |
| **Candidate Generation & Ranking** | Multi-scale hypothesis generation (point, corridor, area) and non-circular scoring. | Native hypothesis engines in `agent/candidate_generator.py` and `agent/candidate_comparator.py`. |

---

## 4. Licensing Obligations

Reference software and research are licensed under the **MIT License**:
```text
Copyright (c) 2026 Oldcircle
Permission is hereby granted, free of charge, to any person obtaining a copy...
```
Reference data tables originating from Wikipedia are licensed under **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**.

### HOUND Compliance:
1. Provide full attribution in `THIRD_PARTY_NOTICES.md` acknowledging Oldcircle and OpenStreetMap contributors (ODbL).
2. Maintain clean separation between independent architectural implementations and algorithmic references.

---

## 5. Identified Vulnerabilities & Mitigations in HOUND

1. **Overpass API Rate Limits:** Public Overpass mirrors are heavily rate-limited. HOUND implements:
   - A multi-mirror rotating pool (`overpass-api.de`, `overpass.private.coffee`, `maps.mail.ru`).
   - Persistent SQLite disk caching (`data/hound.db`).
   - Configurable custom Overpass endpoints.
2. **Reverse Image Search Fragility:** Headless browser scraping is vulnerable to bot detection and CAPTCHAs. HOUND isolates reverse image search behind an abstract provider adapter and supports structured search APIs alongside local analysis.
3. **Windows Encoding Traps:** All file reads, writes, and subprocess invocations mandate `encoding="utf-8", errors="replace"`.
4. **Circular Evidence Bias:** HOUND separates seed evidence items from independent corroboration to prevent self-reinforcing score inflation.
