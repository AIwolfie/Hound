# GEO-SLEUTH REPOSITORY AUDIT

**Date:** 2026-11-05  
**Auditor:** Lead Architect, HOUND Development Team  
**Source Path:** `C:\Users\mayan\Downloads\geo-sleuth-main\geo-sleuth-main`  
**License:** MIT License (code), CC BY-SA 4.0 (derived reference tables)  
**Upstream Repository:** `Oldcircle/geo-sleuth` (v2 Agent Skill)

---

## 1. Executive Summary

Geo-Sleuth is a structured agent skill designed for automated and semi-automated geographic image localization (geolocation and chronolocation). Rather than relying on black-box visual guesses, Geo-Sleuth separates the investigation into three explicit layers:
1. **Decision Layer (`board.py`):** Structured hypothesis board managing clues, likelihood ratios, exclusions, and rank calculation.
2. **Perception & Computation Layer:** 20+ specialized CLI utilities performing EXIF parsing, image preprocessing, multi-scale OCR, reverse image search, sun/shadow geometry, OpenStreetMap Overpass mining, digital elevation model (DEM) horizon/ridge matching, perspective camera pose estimation, and satellite/street-view neural ranking (CLIP zero-shot and DINOv2 + SIFT).
3. **Judgment Layer:** An autonomous agent or human analyst selecting among machine-ranked top candidates, evaluating qualitative scene clues, and closing verification loops.

Geo-Sleuth provides an outstanding catalog of proven mathematical, geometric, and GIS algorithms. However, as an Agent Skill designed for interactive terminal execution (via `uv run script.py`), it lacks:
- A unified, asynchronous Python API package architecture (each script is standalone with PEP 723 inline dependencies).
- Robust cross-platform console encoding resilience (Windows default `cp1252` stdout crashes on non-ASCII place names without `PYTHONUTF8=1`).
- Multi-provider fallback orchestration (e.g., if Overpass or Baidu times out, external pipelines stall or require human intervention).
- Enterprise persistence, audit trail databases, and strict contradictory-evidence engines.

HOUND will adopt the algorithmic strengths of Geo-Sleuth through decoupled integration adapters while building a modular, production-grade autonomous agent architecture.

---

## 2. Environment & Dependency Audit

| Component | Status | Details |
|---|---|---|
| **Python Runtime** | Available | Python 3.13.7 (Verified) |
| **Package Manager** | Available | `uv` 0.10.2 (Verified) |
| **Network Client** | Available | System `curl` available |
| **Browser Engine** | Available | Playwright Chromium launched and verified via `doctor.py` |
| **OCR Backend** | Configured | RapidOCR (ONNX runtime fallback for non-macOS systems) |
| **Vision / ML Models** | Lazy-Loaded | PyTorch (CUDA / CPU), torchvision, Hugging Face Hub (CLIP, DINOv2). Weights downloaded on first use. |
| **GIS / Elevation Tiles**| Available | AWS Terrarium elevation tiles (HTTP 200 confirmed) |
| **Tile Mosaic Servers** | Available | Google Satellite / OSM Tile endpoints reachable |
| **Search Engines** | Mixed | Yandex Images HTTP 200; Baidu Image search timed out (curl 28); Google Lens not implemented in Geo-Sleuth v2 |
| **OpenStreetMap** | High Load | `overpass-api.de` and mirrors periodically return HTTP 406/504/timeouts; Nominatim requires strict headers |

---

## 3. Script Inventory & Classification

Geo-Sleuth components are classified according to the directive's 5 tiers:
- **Tier 1:** Reusable without modification (as external CLI tools or via adapter wrappers)
- **Tier 2:** Reusable with modifications (adapting to in-memory Python library APIs, fixing encoding/timeouts)
- **Tier 3:** Requires replacement (fragile scrapers, unsupported or single-point-of-failure services)
- **Tier 4:** Requires additional testing (heavy ML pipelines, large-scale spatial queries)
- **Tier 5:** Not relevant to HOUND (agent skill meta-wrappers, vendor-specific IDE configs)

### Complete Inventory

| Script / Module | Classification | Purpose & Description | Integration Strategy for HOUND |
|---|---|---|---|
| `scripts/doctor.py` | **Tier 2: Reusable with modifications** | System health, environment, table readability, browser, and network diagnostic utility. | Integrate into HOUND system diagnostics (`hound doctor` command). Replace CLI reporting with structured dictionary output. |
| `scripts/exif.py` | **Tier 1: Reusable without modification** | Extracts GPS coordinates, timestamps, camera make/model, focal length, 35mm equivalent, orientation, heading. | Wrap in HOUND's `investigators/metadata/exif_analyzer.py`. Can be invoked via adapter or imported directly. |
| `scripts/imgprep.py` | **Tier 2: Reusable with modifications** | Generates edge crops (N/S/E/W/corners), zoomed details, tiles, and column detection for regular pier spacing. | Refactor into `core/image_preprocessor.py` to allow direct in-memory `PIL.Image` manipulation rather than writing hundreds of disk files on each pass. |
| `scripts/ocr.py` | **Tier 2: Reusable with modifications** | RapidOCR / Apple Vision multi-pass OCR (original, 2x upscale, edge tiles). | Wrap in `investigators/ocr/ocr_engine.py`. Add confidence scoring, coordinate bounding boxes, and international script detection. |
| `scripts/revimg.py` | **Tier 3: Requires replacement** | Web scrapers for Baidu and Yandex image search using Playwright browser automation. | Replace with a pluggable adapter framework (`integrations/reverse_image/`) supporting Google Lens, Bing Visual Search, Yandex, TinEye, with headless browser as a fallback, preserving lawful rate limits. |
| `scripts/intake.py` | **Tier 2: Reusable with modifications** | Sequential pipeline orchestrating EXIF, edge crops, OCR, and reverse image search into `intake.md`. | Re-implement inside HOUND's `agent/orchestrator.py` Phase 1 (Image Intake) as an async, non-blocking pipeline with structured event emission. |
| `scripts/clues.py` | **Tier 1: Reusable without modification** | Lookup engine for calling codes, driving sides, dependent territories, plate formats, and area codes. | Adapt directly into `investigators/infrastructure/` and `investigators/metadata/`. The bundled data tables in `skills/geo-sleuth/data/` are directly reusable. |
| `scripts/regions.py` | **Tier 1: Reusable without modification** | Region pack registry (CA, CN, IL, JP, KG, LK, MM, MX, NZ, TR, US). | Leverage as an extensible knowledge base for country-specific OSINT clue catalogs. |
| `scripts/board.py` | **Tier 2: Reusable with modifications** | Candidate board tracking hypotheses, likelihood ratios (1/5 to 5), exclusions, and scan ordering. | Crucial design reference for HOUND's `agent/hypothesis_engine.py` and `agent/evidence_manager.py`. Refactor state storage into SQLite / typed JSON-schema with ACID guarantees. |
| `scripts/gazetteer.py` | **Tier 2: Reusable with modifications** | Administrative division hierarchy query tool using OSM boundaries and bounding boxes. | Adapt into `integrations/geocoding/gazetteer_service.py` with multi-tier caching (SQLite/disk) to survive Overpass outages. |
| `scripts/poi.py` | **Tier 2: Reusable with modifications** | Point of Interest and compound name resolver using 360 Maps, Nominatim, and Baidu Suggest. | Adapt into `integrations/geocoding/poi_resolver.py`. Add defensive headers to satisfy Nominatim requirements. |
| `scripts/sun.py` | **Tier 1: Reusable without modification** | Ephemeris and solar geometry engine. Solves latitude bands, capture times, solar azimuth, and camera heading from lit/shaded faces. | Reusable mathematical core. Wrap in HOUND's `investigators/shadows/solar_analyzer.py`. |
| `scripts/osm.py` | **Tier 2: Reusable with modifications** | Advanced Overpass QL generator for linear infrastructure (railways, powerlines, rivers), feature co-occurrence, and crossings. | Wrap in `integrations/openstreetmap/overpass_client.py` with resilient failover across multiple public mirrors and custom local Overpass/Planet endpoints. |
| `scripts/tiles.py` | **Tier 1: Reusable without modification** | Satellite and map tile fetcher (Google Satellite, OSM, etc.) with caching and mosaic assembly. | Adapt into `integrations/satellite/tile_service.py`. |
| `scripts/sat_scan.py` | **Tier 4: Requires additional testing** | Zero-shot satellite tile ranking using OpenAI CLIP (e.g. searching for athletics tracks, factories, quarries, solar farms). | Adapt into `investigators/infrastructure/satellite_classifier.py`. Needs benchmark testing on Windows with CUDA vs. CPU fallback. |
| `scripts/baidu_pano.py` | **Tier 2: Reusable with modifications** | Baidu Panorama API client for China street view imagery and metadata. | Wrap in `integrations/street_imagery/baidu_adapter.py`. |
| `scripts/gsv.py` | **Tier 2: Reusable with modifications** | Google Street View panorama locator and image renderer. | Wrap in `integrations/street_imagery/gsv_adapter.py`. Ensure graceful degradation when official Google Street View API keys are absent. |
| `scripts/match.py` | **Tier 4: Requires additional testing** | Two-stage candidate matching: DINOv2 global embeddings followed by SIFT feature keypoint inlier verification. | Excellent verification mechanism for `verification/candidate_verifier.py`. Requires testing for memory footprint and execution speed on local GPUs. |
| `scripts/terrain.py` | **Tier 1: Reusable without modification** | Digital elevation model (AWS Terrarium) processor: horizon profiling, synthetic viewpoint rendering, ridge extraction, and camera roll/pitch fitting. | Exceptional mathematical asset. Adapt into `investigators/terrain/elevation_matcher.py`. |
| `scripts/pose.py` | **Tier 1: Reusable without modification** | Perspective-n-Point (PnP) solver calculating camera coordinates, altitude, heading, pitch, and roll from known 3D-2D landmarks. | Adapt into `investigators/visual/pose_estimator.py`. |
| `scripts/geo.py` | **Tier 1: Reusable without modification** | Geodetic calculations: Vincenty/Haversine distance, bearings, line-of-sight ray intersection, FOV occlusion checks, pier spacing ruler. | Adapt into `core/geographic_search.py` and `verification/candidate_verifier.py`. |
| `scripts/evidence.py` | **Tier 2: Reusable with modifications** | Generates visual verification cards combining satellite tiles, camera FOV cones, and comparative crops. | Enhance in `reports/evidence_visualizer.py` with modern SVG / high-resolution HTML5 export. |
| `scripts/_browser.py` | **Tier 2: Reusable with modifications** | Playwright Chromium / Chrome launcher with anti-detection args. | Reusable helper for browser-based investigation tasks. |
| `scripts/_device.py` | **Tier 1: Reusable without modification** | Auto-detects PyTorch execution device (`cuda`, `mps`, `cpu`). | Direct reuse in HOUND's ML pipelines. |
| `scripts/_image.py` | **Tier 1: Reusable without modification** | Safe PIL image loader with EXIF auto-rotation and format normalization. | Ingest into `core/image_loader.py`. |
| `scripts/_net.py` | **Tier 2: Reusable with modifications** | Proxy arguments and curl wrapper helpers. | Modernize using Python's `httpx` or `aiohttp` for async I/O. |
| `SKILL.md` | **Tier 5: Not relevant to HOUND** | LLM agent system prompt specific to Claude Code / Agent Skill standard. | Serves as architectural inspiration; replaced by HOUND's autonomous state machine. |
| `.github/` & templates | **Tier 5: Not relevant to HOUND** | Upstream repository issue templates. | Not relevant. |

---

## 4. Empirical Test Results

### 4.1 Diagnostic Health Check (`doctor.py`)
- Python 3.13.7, `uv`, `curl`, and Playwright Chromium passed without issue.
- Global lookup tables (calling codes, driving sides, territories, country names) all passed validation.
- All 11 region packs validated.
- **Failures Detected:**
  - Baidu Image Search timed out (`curl exit 28`).
  - Public Overpass default status endpoint timed out.
  - Terrarium elevation tiles, Yandex, Hugging Face, Google Satellite tiles, and Google Street View were completely functional.

### 4.2 Comprehensive Unit Test Suite (`tests/`)
- Command: `python -m unittest discover -s skills/geo-sleuth/tests -t skills/geo-sleuth/tests`
- Total Tests: 196
- Results:
  - Default Windows environment: 191 Passed, 4 Skipped, 1 Failed (`UnicodeEncodeError` in `test_regions.py`).
  - With `PYTHONUTF8=1`: **192 Passed, 4 Skipped, 0 Failed** (Execution time: 41.2s).
- **Key Insight:** Geo-Sleuth core algorithms (Sun, Terrain, Pose, Board, Geo, Match) are mathematically sound and have extensive regression coverage. Windows installations must strictly mandate UTF-8 stream handling.

---

## 5. Licensing Obligations

Geo-Sleuth is licensed under the **MIT License**:
```text
Copyright (c) 2026 Oldcircle
Permission is hereby granted, free of charge, to any person obtaining a copy...
```
The reference tables in `skills/geo-sleuth/data/` originating from Wikipedia are licensed under **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**.

### HOUND Compliance Strategy:
1. Maintain Geo-Sleuth as an external upstream dependency/reference under `integrations/geo_sleuth/`.
2. Do not mutate or overwrite original files in `C:\Users\mayan\Downloads\geo-sleuth-main\geo-sleuth-main`.
3. Provide full attribution in `HOUND/LICENSES.md` and docs acknowledging Oldcircle and OpenStreetMap contributors (ODbL).

---

## 6. Identified Vulnerabilities & Technical Blockers

1. **Overpass API Rate Limits:** Public Overpass mirrors are heavily rate-limited. HOUND must implement:
   - A multi-mirror rotating pool (`overpass-api.de`, `private.coffee`, `mail.ru`).
   - Aggressive disk/database caching (`.hound_cache/osm/`).
   - Configurable custom Overpass endpoints (e.g. self-hosted Overpass or regional extracts).
2. **Reverse Image Search Fragility:** Headless browser scraping of Baidu/Yandex is vulnerable to bot detection and CAPTCHAs. HOUND must isolate reverse image search behind an abstract provider adapter and support optional official search APIs (SerpApi, Google Custom Search, Bing Visual Search) alongside lawful scraping fallbacks.
3. **ML Initialization Latency:** First-time downloads of DINOv2 and CLIP models can exceed 2.5 GB. HOUND must feature a dedicated model pre-warming command (`hound setup --models`) and allow complete offline execution of geometric/OSM modules when ML packages are not installed.
4. **Windows Encoding Traps:** All file reads, writes, and subprocess invocations must specify `encoding="utf-8", errors="replace"`.

---

## 7. Next Actions

1. Construct `HOUND_ARCHITECTURE.md` defining HOUND's modular agent architecture, state machine, evidence model, and verification matrix.
2. Initialize HOUND directory structure, base configuration, requirements, and development logging.
3. Proceed directly to **PHASE 1: FOUNDATION**.
