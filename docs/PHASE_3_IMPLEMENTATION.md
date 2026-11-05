# HOUND — PHASE 3: GEOGRAPHIC INTELLIGENCE ENGINE
## System Architecture & Technical Implementation Document

**HOUND — Autonomous Visual Geolocation & OSINT Agent**  
*Tagline: Every image leaves a trail.*

---

## 1. Executive Summary

Phase 3 transitions HOUND from visual clue extraction into full **Geographic Intelligence**. Visual clues—such as highway numbers, street signs, transit markers, vehicle registration styles, and solar ephemeris bounds—are grounded against authoritative global GIS databases, digital elevation models, satellite imagery, and spatial graph networks.

Rather than synthesizing speculative coordinates, HOUND formulates explicit, testable geographic candidate polygons across spatial hierarchies (point, building, road corridor, area, city, administrative region), cross-references candidate topography against digital elevation models (DEM), queries OpenStreetMap infrastructure for verification, and enforces strict falsification conditions.

---

## 2. Core Architectural Components

```
HOUND Geographic Intelligence Engine
├── core/
│   └── geographic_search.py        # Geodetic math (Haversine, destination point, ray intersection, Web Mercator)
├── integrations/
│   ├── openstreetmap/
│   │   ├── geometry_parser.py     # Overpass JSON parser (nodes, ways, relations, bounding boxes)
│   │   └── overpass_client.py     # Resilient client (mirrors, health monitoring, failover, SQLite caching)
│   ├── geocoding/
│   │   └── gazetteer.py           # Forward/reverse geocoding (Nominatim, hierarchy resolution, persistent cache)
│   ├── elevation/
│   │   ├── terrarium_client.py    # AWS Terrarium tile decoder (RGB to meters), bilinear interpolation
│   │   ├── horizon_generator.py   # 360° synthetic terrain horizon with curvature & refraction (k=0.13)
│   │   └── ridge_matcher.py       # Cross-correlation matching observed vs synthetic skylines
│   └── satellite/
│       └── tile_service.py        # Satellite/map tile fetcher, mosaic stitching, pixel-to-WGS84 projection
├── agent/
│   ├── candidate_generator.py     # Visual clue to candidate polygon synthesis & solar falsification
│   ├── candidate_comparator.py    # Multi-factor scoring, level weights, contradiction penalties
│   └── investigation_manager.py   # End-to-end Phase 3 investigation orchestrator
├── reports/
│   └── geographic_visualizer.py   # High-resolution candidate maps, 360° horizon plots, comparison cards
└── interface/
    └── cli.py                     # `hound geo doctor`, `terrain`, `investigate`, `candidates`, `map`
```

---

## 3. Detailed Component Engineering

### 3.1 Geodetic Mathematics & Web Mercator Geometry (`core/geographic_search.py`)
- **Haversine Distance**: Calculates great-circle distance between coordinates on a WGS-84 ellipsoid sphere ($R = 6,371,000 \text{ m}$).
- **Forward Azimuth / Bearing**: Computes initial true north bearing from observer to target coordinate:
  $$\theta = \text{atan2}(\sin(\Delta \lambda) \cdot \cos(\phi_2), \cos(\phi_1) \cdot \sin(\phi_2) - \sin(\phi_1) \cdot \cos(\phi_2) \cdot \cos(\Delta \lambda))$$
- **Destination Point**: Projects a coordinate along a great circle given an initial coordinate, distance $d$, and bearing $\theta$.
- **Bounding Envelopes (`BoundingBox`)**: Supports spatial bounding boxes, intersection detection, radius expansion, and point containment.
- **Web Mercator Slippy Tile Math**: Converts $(\text{lat}, \text{lon}, z) \leftrightarrow (t_x, t_y)$ with exact sub-pixel boundary projection.

### 3.2 OpenStreetMap Overpass Client (`integrations/openstreetmap/`)
- **Mirror Pool & Automatic Failover**: Actively monitors pool of public Overpass endpoints (`overpass-api.de`, `overpass.private.coffee`, `maps.mail.ru`) with health scoring and failure counts.
- **Persistent SQLite Caching**: Normalizes Overpass QL queries, computes SHA-256 query hashes, and persists full JSON responses into SQLite (`osm_cache`).
- **Structured Geometry Parsing**: Parses Overpass JSON into structured Pydantic primitives (`OSMNode`, `OSMWay`, `OSMRelation`, `OSMResponse`).
- **Pre-Built Domain Query Builders**:
  - `query_highways(bbox, highway_types)`: Primary highways, trunks, motorways, bridges, tunnels.
  - `query_railways(bbox)`: Rail tracks, electrified lines, high-speed corridors.
  - `query_power_lines(bbox)`: Transmission power lines, high-voltage pylons.
  - `query_waterways(bbox)`: Rivers, streams, canals.
  - `query_named_street(bbox, street_name)`: Pinpoints exact street vectors within candidate regions.

### 3.3 Geocoding & Gazetteer Service (`integrations/geocoding/`)
- **Forward & Reverse Geocoding**: Queries OpenStreetMap Nominatim with structured address extraction.
- **Hierarchical Boundary Resolution**: Resolves country, state/province, county, city, district, and exact bounding box envelopes.
- **Local SQLite Caching**: Eliminates repeated queries by hashing query strings and coordinate pairs in `gazetteer_cache`.
- **Strict OSINT Attribution**: Adheres to the Nominatim usage policy with user-agent identification and rate-limiting.

### 3.4 Digital Elevation Modeling & Horizon Synthesis (`integrations/elevation/`)
- **AWS Terrarium DEM Decoding**: Decodes elevation from Terrarium RGB tiles according to the global specification:
  $$H = (R \cdot 256.0 + G + B / 256.0) - 32768.0$$
- **Sub-Pixel Bilinear Interpolation**: Evaluates smooth, continuous elevation surfaces without tile step artifacts.
- **Earth Curvature & Atmospheric Refraction**: Computes apparent angular elevation $\theta$ over distance $d$:
  $$\theta_{\text{apparent}} = \arctan\left(\frac{H_{\text{target}} - H_{\text{observer}}}{d}\right) - \frac{d}{2 R_{\text{eff}}}$$
  where $R_{\text{eff}} = \frac{R_{\text{earth}}}{1 - k}$ with standard terrestrial refraction coefficient $k \approx 0.13$.
- **Ridge Silhouette Matching (`ridge_matcher.py`)**: Uses 1D cross-correlation and root-mean-square error (RMSE) optimization to resolve camera heading and pitch offset against an observed skyline.

### 3.5 Satellite & Map Tile Service (`integrations/satellite/`)
- **Multi-Provider Support**: Supports OpenStreetMap cartography, Carto Light basemaps, and satellite rasters.
- **Dynamic Mosaic Stitching**: Assembles multi-tile matrices into single georeferenced composite images.
- **Tile Budget Protection**: Protects against excessive bandwidth by enforcing configurable tile budgets (e.g., maximum 16 tiles) and automatically stepping down zoom levels for continent-wide envelopes.

### 3.6 Candidate Generator & Active Falsification (`agent/candidate_generator.py`)
- **Direct GPS Anchors**: Elevates verified EXIF GPS coordinates into high-confidence point candidates.
- **Corridor Hypotheses**: Maps OCR road designations (e.g., `US-101`, `Route 66`, `M4`) to linear geographic envelopes.
- **Street Segments**: Converts localized street names into city-level candidate bounding boxes.
- **Vehicle Registration Livery**: Maps detected license plate liveries (e.g., British Columbia) to administrative boundaries.
- **Astronomical Solar Feasibility Exclusion**: Evaluates whether observed solar elevation angles are astronomically possible at candidate latitudes:
  $$\text{Elev}_{\text{max}} = 90^\circ - |\phi - \delta_{\text{sun}}|$$
  If $\text{Elev}_{\text{observed}} > \text{Elev}_{\text{max}} + 5^\circ$, the candidate is permanently marked as `status = "excluded"` with explicit falsification conditions logged.

### 3.7 Candidate Comparator & Transparent Ranking (`agent/candidate_comparator.py`)
- **Level Specificity Multipliers**:
  - `point`: 5.0
  - `building`: 4.0
  - `road`: 3.0
  - `area`: 2.5
  - `city`: 2.0
  - `admin`: 1.5
  - `region`: 1.0
- **Terrain Correlation Bonus**: Multiplies score by $1.0 + (2.0 \cdot S_{\text{terrain}})$.
- **Contradiction Penalty**: Applies exponential penalty $0.10^n$ for each contradictory physical constraint.

---

## 4. CLI Interface Verification

The HOUND CLI provides dedicated Phase 3 commands under `hound geo`:

```bash
# 1. Inspect Geographic endpoints (Overpass, Terrarium, Gazetteer)
python main.py geo doctor

# 2. Compute 360-degree synthetic terrain horizon profile
python main.py geo terrain 27.9881 86.9250 --distance-km 15.0 -o everest_horizon.jpg

# 3. Execute Phase 3 Geographic Intelligence on an investigation
python main.py geo investigate <investigation_id>

# 4. View candidate hypotheses ledger
python main.py geo candidates <investigation_id>

# 5. Generate and render candidate map
python main.py geo map <investigation_id>
```

---

## 5. Verification & Quality Assurance

- **Unit Test Suite**: 60 unit tests across 18 test modules.
- **Pass Rate**: 100% (60 passed, 0 failures, 0 errors).
- **Execution Duration**: ~36 seconds.
- **Safety & Boundary Compliance**: Local-first processing, transparent evidence chains, zero fabricated coordinates.
