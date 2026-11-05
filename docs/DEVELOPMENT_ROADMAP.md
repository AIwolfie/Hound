# HOUND — Multi-Phase Development Roadmap

## Phase 0: Research, Audit & Architecture Specification
- [x] Verify reference GIS toolkits and environment.
- [x] Conduct 20+ script empirical audit and dependency mapping.
- [x] Identify Windows codepage traps and mandate UTF-8 stream handling.
- [x] Design modular HOUND architecture and evidence taxonomy.
- [x] Document third-party licensing and copyright notices (`THIRD_PARTY_NOTICES.md`).

## Phase 1: Foundation Architecture & Image Intake
- [x] Pydantic configuration loader with environment variable overrides.
- [x] Rich forensic console logging and structured file logging.
- [x] Cryptographic SHA-256 hashing and non-destructive image preservation.
- [x] EXIF metadata parser (GPS decimal conversion, camera make/model, lens, optics, ISO).
- [x] Metadata integrity evaluator (detects editing tools, stripped tags).
- [x] SQLite relational storage (investigations, metadata, evidence ledger, candidates).
- [x] Base investigator contract and EXIF metadata investigator.
- [x] Forensic CLI terminal identity (`hound intake`, `hound inspect`, `hound doctor`, `hound list`).
- [x] 13 automated unit tests with 100% pass rate.

## Phase 2: Visual Intelligence Engine (Current Milestone)
- [x] Image preprocessing engine: 7 analysis variants (CLAHE, contrast, grayscale, edges, sharpened, 2x upscale).
- [x] 12 coordinate-mapped spatial crops/tiles with bounding box metadata.
- [x] Multi-pass OCR investigation engine powered by RapidOCR (ONNX Runtime).
- [x] Spatial IoU and string-similarity OCR deduplication.
- [x] Geographic clue classification (`road_number`, `street_name`, `railway_station`, `telephone_number`, `postal_code`, etc.).
- [x] NOAA astronomical solar position calculation and shadow length ratio solver.
- [x] Adaptive HSV + PCA shadow vector candidate extraction.
- [x] CPU-compatible classical computer vision infrastructure detectors (utility poles, signage panels, road markings).
- [x] Visual evidence overlays (OCR bounding boxes, infrastructure boxes, shadow vectors, comparison sheets).
- [x] Extended SQLite schema with `ocr_observations`, `infrastructure_detections`, `shadow_observations`.
- [x] CLI commands: `hound preprocess`, `hound ocr`, `hound analyze`.
- [x] 30 automated unit tests with 100% pass rate.

## Phase 3: Geographic Intelligence Engine (Next Milestone)
- [ ] OpenStreetMap Overpass client with rotating mirror pool and persistent caching.
- [ ] Administrative boundary gazetteer and Nominatim geocoding adapter.
- [ ] Digital elevation model (Terrarium DEM) horizon synthesizer.
- [ ] Ridge line cross-correlation and camera roll/pitch solver.
- [ ] Initial geographic candidate generation (Country -> Region -> City -> Area).

## Phase 4: Autonomous Investigation & Hypothesis Management
- [ ] Master agent orchestrator state machine.
- [ ] Hypothesis engine and candidate tree management.
- [ ] Cost-aware task planner (ordering tests from cheap to expensive).
- [ ] Likelihood-ratio evidence integration.

## Phase 5: Adversarial Verification & Anti-Hallucination
- [ ] Contradiction engine with active falsification.
- [ ] Occlusion and line-of-sight verification.
- [ ] Tiered confidence scoring engine (City, Area, Road, Building, Pose).

## Phase 6: Reporting & Visual Deliverables
- [ ] Forensic investigation report generator (Markdown, HTML5, structured JSON).
- [ ] Satellite overlay compositor with camera field-of-view wedges.
- [ ] Audit trail exporter.

## Phase 7: Interactive Investigation Dashboard
- [ ] Professional dark-mode analyst dashboard (FastAPI backend).
- [ ] Interactive Leaflet/MapLibre map viewer with candidate markers.
- [ ] Side-by-side evidence inspection panel.

## Phase 8: Multi-Frame & Video Investigation
- [ ] Video frame extraction and temporal sampling.
- [ ] Persistent visual feature tracking across frames.
- [ ] Unified multi-frame geographic synthesis.
