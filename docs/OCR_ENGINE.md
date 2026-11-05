# HOUND — Multi-Pass OCR Investigation Engine

## 1. Overview

The OCR Investigation Engine (`investigators/ocr/ocr_engine.py`) extracts visible text from forensic imagery. Rather than running a single flat OCR pass, HOUND executes a multi-pass strategy across preprocessed variants and coordinate-mapped crop pyramids, translates all detected bounding boxes back to the original image coordinate frame, deduplicates overlaps via spatial Intersection-over-Union (IoU) and string similarity, and classifies recognized text into geographic clue categories.

---

## 2. Multi-Pass Investigation Architecture

```mermaid
flowchart TD
    IMG["Ingested Target Image"] --> MANIFEST["Preprocessing Manifest"]

    subgraph PASSES ["Multi-Pass Scanning Pipeline"]
        P1["Pass 1: Full Normalized Resolution"]
        P2["Pass 2: Contrast Enhanced (CLAHE)"]
        P3["Pass 3: 2x Super-Sampled (Lanczos)"]
        P4["Pass 4: 12 Overlapping Spatial Tiles & Edge Crops"]
    end

    MANIFEST --> P1 & P2 & P3 & P4

    subgraph TRANS ["Coordinate Transformation & Unification"]
        P1 --> MAP1["Identity Mapping [x0, y0, x1, y1]"]
        P2 --> MAP2["Identity Mapping [x0, y0, x1, y1]"]
        P3 --> MAP3["Divide Coordinates by 2.0"]
        P4 --> MAP4["Translate Crop Offsets: [ox0 + x0, oy0 + y0, ...]"]
    end

    MAP1 & MAP2 & MAP3 & MAP4 --> RAW_POOL["Unified Detection Pool"]

    subgraph DEDUP ["Deduplication & Classification"]
        RAW_POOL --> IOU["Spatial IoU Clustered Deduplication<br/>(IoU > 0.45 or IoU > 0.20 + String Match)"]
        IOU --> SCRIPT["Unicode Script Classifier<br/>(Latin, Cyrillic, Han, Arabic, Devanagari, Greek, Hangul, Kana)"]
        IOU --> CAT["Geographic Clue Classifier<br/>(Street Name, Road Number, Station, Phone, Postal Code)"]
    end

    SCRIPT & CAT --> DB[("SQLite: ocr_observations")]
    SCRIPT & CAT --> EVID["HOUND Evidence Ledger (Status: 'read')"]
```

---

## 3. Pluggable OCR Backend Architecture

HOUND isolates OCR execution behind an abstract `BaseOCRBackend`:
```python
class BaseOCRBackend(ABC):
    @abstractmethod
    def detect_text(self, image_input: Path | np.ndarray) -> List[RawOCRBox]:
        pass
```
The primary backend is **RapidOCR** (`RapidOCRBackend`), powered by ONNX Runtime with lightweight text detection (DBNet) and text recognition (CRNN/SVTR) models. The backend operates completely offline with zero mandatory cloud API dependencies.

---

## 4. Geographic Clue Classification Catalog

Every recognized text string is analyzed against regex heuristics and international naming patterns:

| Clue Category | Description & Pattern Heuristics | Examples |
|---|---|---|
| **`road_number`** | Highway, interstate, national route, or motorway designation, including optional directional suffixes. | `I-80`, `US-101 North`, `M4`, `A1`, `NH 48`, `G107`, `B402`, `E15` |
| **`street_name`** | Multilingual street/road suffixes, including compound German/Nordic words. | `Main Street`, `Grand Ave`, `Müllerstrasse`, `Rue de Rivoli`, `Avenida Paulista`, `南京东路` |
| **`railway_station`** | Multilingual transit and rail terminus indicators. | `Central Station`, `Hauptbahnhof`, `Gare du Nord`, `Estación Central`, `新宿駅`, `北京南站` |
| **`transport_authority`**| Urban rapid transit, subway, and railway authority branding. | `Metro`, `Subway`, `SNCF`, `DB`, `TfL`, `MTA`, `BART`, `S-Bahn` |
| **`telephone_number`** | Standard international and local telephone dialing sequences. | `+1 (555) 234-5678`, `(02) 9876 5432`, `0800 123 456` |
| **`postal_code`** | Numeric or alphanumeric postal codes. | `90210`, `75008`, `SW1A 1AA`, `K1A 0B1` |
| **`abbreviation`** | 2-3 character capital acronyms. | `FDNY`, `BC`, `CA`, `DHS` |
| **`unknown`** | General descriptive or unclassified signage text. | `Welcome`, `Pharmacy`, `Fresh Bread` |

---

## 5. Spatial & Textual Deduplication

Because text may be detected across multiple passes (e.g. visible both in the full image and inside `tile_r0_c0`), HOUND clusters detections:
1. Sorts raw detections by descending model confidence.
2. For each detection, calculates spatial IoU against existing clusters:
   - If $\text{IoU} > 0.45$, the boxes refer to the same physical text sign.
   - If $\text{IoU} > 0.20$ and normalized strings match, they are merged.
3. Retains the highest-confidence reading and records `passes_detected` (e.g. `["full_normalized", "tile_r0_c0"]`).

---

## 6. Evidence Provenance & Storage

Observations are persisted to the SQLite `ocr_observations` table:
- `investigation_id`, `text`, `confidence`, `box_x0`, `box_y0`, `box_x1`, `box_y1`, `category`, `language_script`, `pass_name`.
- Formally emitted into the HOUND Evidence Ledger as `EvidenceItem` with status **`read`**. Text is strictly treated as an observation, not an assumed location proof.
