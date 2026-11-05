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

## Phase 2: Visual Intelligence Engine
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

## Phase 3: Geographic Intelligence Engine
- [x] Geodetic mathematics (Haversine, forward azimuth, destination points, ray intersection, Web Mercator slippy tile math).
- [x] OpenStreetMap Overpass client with mirror pool, health monitoring, failover, and persistent SQLite query caching (`osm_cache`).
- [x] Domain-specific OSM geometry query builders (highways, railways, stations, power lines, waterways, named streets).
- [x] Administrative boundary gazetteer and Nominatim geocoding with local persistent SQLite cache (`gazetteer_cache`).
- [x] Digital elevation model (AWS Terrarium DEM) decoder with sub-pixel bilinear interpolation.
- [x] 360-degree synthetic terrain horizon generator incorporating Earth curvature and atmospheric refraction ($k=0.13$).
- [x] Mountain ridge line cross-correlation matcher with camera heading and pitch offset solver.
- [x] Provider-independent satellite and map tile service with multi-tile mosaic stitching and geo-to-pixel projection.
- [x] Multi-scale geographic candidate generator (point, building, road corridor, area, city, admin region).
- [x] Astronomical solar elevation active falsification and latitude exclusion.
- [x] Candidate comparator with level weighting multipliers, terrain bonuses, contradiction penalties, and transparent ranking.
- [x] Geographic visualizer generating georeferenced candidate maps and 360° horizon skyline profiles.
- [x] CLI commands: `hound geo doctor`, `hound geo terrain`, `hound geo investigate`, `hound geo candidates`, `hound geo map`.
- [x] 60 automated unit tests with 100% pass rate.

## Phase 4: Autonomous Investigation & Hypothesis Management (Next Milestone)
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
