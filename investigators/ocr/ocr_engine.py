"""HOUND Multi-Pass OCR Investigation Engine.

Coordinates multi-pass OCR recognition, coordinate transformations across crop pyramids,
IoU spatial-textual deduplication, and geographic clue classification.
"""
from __future__ import annotations

import re
import time
import unicodedata
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from core.image_loader import ImageRecord
from core.image_preprocessor import ImagePreprocessor, PreprocessingManifest
from core.logger import get_logger
from investigators.base import BaseInvestigator
from storage.database import DatabaseManager
from storage.evidence_store import EvidenceItem

logger = get_logger("hound.ocr")


class RawOCRBox(BaseModel):
    text: str
    confidence: float
    box: Tuple[float, float, float, float]  # (x0, y0, x1, y1) in detector coordinate space
    pass_name: str
    backend: str


class OCRObservation(BaseModel):
    id: Optional[int] = None
    investigation_id: str
    text: str
    confidence: float
    box_original: Tuple[float, float, float, float]  # (x0, y0, x1, y1) in original image space
    category: str
    language_script: str
    pass_name: str
    passes_detected: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BaseOCRBackend(ABC):
    """Abstract interface for pluggable OCR engines."""

    @abstractmethod
    def detect_text(self, image_input: Path | np.ndarray) -> List[RawOCRBox]:
        """Recognize text and return bounding boxes and confidence scores."""
        pass


class RapidOCRBackend(BaseOCRBackend):
    """RapidOCR engine implementation powered by ONNX Runtime."""

    def __init__(self):
        try:
            from rapidocr_onnxruntime import RapidOCR

            self._engine = RapidOCR()
            self._available = True
        except ImportError:
            logger.warning("RapidOCR package not available; running with null fallback.")
            self._engine = None
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def detect_text(self, image_input: Path | np.ndarray) -> List[RawOCRBox]:
        if not self._available or self._engine is None:
            return []

        results: List[RawOCRBox] = []
        try:
            if isinstance(image_input, Path):
                img_arg = str(image_input)
            elif isinstance(image_input, np.ndarray):
                img_arg = image_input
            else:
                img_arg = np.array(image_input)

            ocr_res, _ = self._engine(img_arg)
            if not ocr_res:
                return []

            for item in ocr_res:
                # RapidOCR format: [poly_points, text, confidence]
                poly, text, conf = item
                if not text or not str(text).strip():
                    continue

                pts = np.array(poly)
                x0 = float(np.min(pts[:, 0]))
                y0 = float(np.min(pts[:, 1]))
                x1 = float(np.max(pts[:, 0]))
                y1 = float(np.max(pts[:, 1]))

                results.append(
                    RawOCRBox(
                        text=str(text).strip(),
                        confidence=float(conf),
                        box=(x0, y0, x1, y1),
                        pass_name="",
                        backend="RapidOCR",
                    )
                )
        except Exception as e:
            logger.error(f"RapidOCR execution error: {e}")

        return results


def compute_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    """Compute Intersection over Union between two axis-aligned bounding boxes."""
    x0 = max(box_a[0], box_b[0])
    y0 = max(box_a[1], box_b[1])
    x1 = min(box_a[2], box_b[2])
    y1 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x1 - x0)
    inter_h = max(0.0, y1 - y0)
    inter_area = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def detect_script(text: str) -> str:
    """Classify the dominant unicode script in the recognized text."""
    scripts: Dict[str, int] = {
        "Latin": 0,
        "Cyrillic": 0,
        "Han": 0,
        "Arabic": 0,
        "Devanagari": 0,
        "Greek": 0,
        "Hangul": 0,
        "Kana": 0,
    }

    for char in text:
        name = unicodedata.name(char, "")
        if "LATIN" in name:
            scripts["Latin"] += 1
        elif "CYRILLIC" in name:
            scripts["Cyrillic"] += 1
        elif "CJK UNIFIED" in name:
            scripts["Han"] += 1
        elif "ARABIC" in name:
            scripts["Arabic"] += 1
        elif "DEVANAGARI" in name:
            scripts["Devanagari"] += 1
        elif "GREEK" in name:
            scripts["Greek"] += 1
        elif "HANGUL" in name:
            scripts["Hangul"] += 1
        elif "HIRAGANA" in name or "KATAKANA" in name:
            scripts["Kana"] += 1

    dominant = max(scripts, key=scripts.get)  # type: ignore
    return dominant if scripts[dominant] > 0 else "Unknown"


def classify_geographic_clue(text: str) -> str:
    """Categorize text into potential geographic intelligence clue types."""
    cleaned = text.strip()
    upper = cleaned.upper()

    # Telephone numbers
    phone_pattern = r"(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}"
    if re.fullmatch(phone_pattern, cleaned) and len(re.sub(r"\D", "", cleaned)) >= 7:
        return "telephone_number"

    # Road numbers (US Interstate, State Route, German Bundesstrasse, UK Motorway, Asian Highway, National Highway)
    # Allows optional directional suffix e.g. "US-101 North", "I-80 W"
    road_pattern = (
        r"^(I-\d{1,3}|US-\d{1,3}|SR-\d{1,3}|M\d{1,3}|A\d{1,3}|N\d{1,3}|B\d{1,3}|"
        r"NH\s*\d+|G\d{3,4}|S\d{3,4}|D\d{1,3}|E\d{1,3})"
        r"(\s+(NORTH|SOUTH|EAST|WEST|N|S|E|W|NB|SB|EB|WB))?$"
    )
    if re.match(road_pattern, upper):
        return "road_number"

    # Street names (multilingual suffixes, compound suffix match for German/Nordic/etc.)
    street_terms = [
        r"\bSTREET\b", r"\bST\b", r"\bROAD\b", r"\bRD\b", r"\bAVENUE\b", r"\bAVE\b",
        r"\bBOULEVARD\b", r"\bBLVD\b", r"\bLANE\b", r"\bLN\b", r"\bWAY\b", r"\bDRIVE\b",
        r"\bDR\b", r"STRASSE\b", r"STRAßE\b", r"\bSTR\b", r"\bRUE\b", r"\bCALLE\b", r"\bVIA\b",
        r"\bPRACA\b", r"\bAVENIDA\b", r"VEJ\b", r"GATAN\b", r"KATU\b", r"路$", r"街$", r"道$", r"巷$", r"弄$",
    ]
    for term in street_terms:
        if re.search(term, upper, re.IGNORECASE):
            return "street_name"

    # Railway stations
    station_terms = [
        r"\bSTATION\b", r"\bSTN\b", r"\bBAHNHOF\b", r"\bHBF\b", r"\bGARE\b",
        r"\bESTACION\b", r"\bSTAZIONE\b", r"站$", r"火车站",
    ]
    for term in station_terms:
        if re.search(term, upper, re.IGNORECASE):
            return "railway_station"

    # Transport authorities
    transport_terms = [
        r"\bMETRO\b", r"\bSUBWAY\b", r"\bTRANSIT\b", r"\bSNCF\b", r"\bDB\b",
        r"\bMTA\b", r"\bTFL\b", r"\bBART\b", r"\bS-BAHN\b", r"\bU-BAHN\b",
    ]
    for term in transport_terms:
        if re.search(term, upper, re.IGNORECASE):
            return "transport_authority"

    # Postal codes (simple heuristic for 5-6 digit or alphanumeric postal formats)
    if re.fullmatch(r"\b\d{5}(-\d{4})?\b", cleaned) or re.fullmatch(r"\b[A-Z]\d[A-Z]\s*\d[A-Z]\d\b", upper):
        return "postal_code"

    # Short single uppercase or business acronyms
    if len(cleaned) <= 3 and cleaned.isupper():
        return "abbreviation"

    return "unknown"


class OCREngine(BaseInvestigator):
    """Multi-pass OCR investigator extracting text and mapping coordinates."""

    def __init__(self, backend: Optional[BaseOCRBackend] = None, min_confidence: float = 0.30):
        super().__init__(name="ocr_engine", category="ocr")
        self.backend = backend or RapidOCRBackend()
        self.min_confidence = min_confidence
        self.preprocessor = ImagePreprocessor()

    def run(
        self,
        investigation_id: str,
        image_record: ImageRecord,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[EvidenceItem]:
        """Execute multi-pass OCR investigation across image variants and tiles."""
        evidence_list: List[EvidenceItem] = []
        raw_detections: List[Tuple[RawOCRBox, Tuple[float, float, float, float]]] = []

        # Determine workspace directory for preprocessed crops
        workspace = (
            context.get("workspace_dir")
            if context and "workspace_dir" in context
            else image_record.original_path.parent / ".preprocessed"
        )
        preprocessed_dir = Path(workspace) / "preprocessed"
        preprocessed_dir.mkdir(parents=True, exist_ok=True)

        manifest = self.preprocessor.process(
            image_path=image_record.original_path,
            output_dir=preprocessed_dir,
            generate_tiles=True,
            generate_edges=True,
        )

        norm_w, norm_h = manifest.normalized_dimensions

        # -------------------------------------------------------------
        # PASS 1: Full Normalized Resolution
        # -------------------------------------------------------------
        norm_variant = next((v for v in manifest.analysis_variants if v.name == "full_normalized"), None)
        if norm_variant and norm_variant.file_path.is_file():
            boxes = self.backend.detect_text(norm_variant.file_path)
            for b in boxes:
                if b.confidence >= self.min_confidence:
                    b.pass_name = "full_normalized"
                    raw_detections.append((b, b.box))

        # -------------------------------------------------------------
        # PASS 2: Contrast Enhanced (CLAHE)
        # -------------------------------------------------------------
        clahe_variant = next((v for v in manifest.analysis_variants if v.name == "clahe"), None)
        if clahe_variant and clahe_variant.file_path.is_file():
            boxes = self.backend.detect_text(clahe_variant.file_path)
            for b in boxes:
                if b.confidence >= self.min_confidence:
                    b.pass_name = "clahe"
                    raw_detections.append((b, b.box))

        # -------------------------------------------------------------
        # PASS 3: Upscaled 2x (Scale coordinates back down)
        # -------------------------------------------------------------
        upscale_variant = next((v for v in manifest.analysis_variants if v.name == "upscaled_2x"), None)
        if upscale_variant and upscale_variant.file_path.is_file():
            boxes = self.backend.detect_text(upscale_variant.file_path)
            for b in boxes:
                if b.confidence >= self.min_confidence:
                    b.pass_name = "upscaled_2x"
                    # Rescale coordinates down by 2.0 to match base normalized image
                    x0, y0, x1, y1 = b.box
                    mapped_box = (x0 / 2.0, y0 / 2.0, x1 / 2.0, y1 / 2.0)
                    raw_detections.append((b, mapped_box))

        # -------------------------------------------------------------
        # PASS 4: Overlapping Tiles & Edge Crops
        # -------------------------------------------------------------
        for crop in manifest.crops:
            if not crop.file_path.is_file():
                continue
            boxes = self.backend.detect_text(crop.file_path)
            ox0, oy0, _, _ = crop.box_original
            for b in boxes:
                if b.confidence >= self.min_confidence:
                    b.pass_name = crop.name
                    # Translate crop coordinates back to full image frame
                    bx0, by0, bx1, by1 = b.box
                    mapped_box = (ox0 + bx0, oy0 + by0, ox0 + bx1, oy0 + by1)
                    raw_detections.append((b, mapped_box))

        # -------------------------------------------------------------
        # Spatial & Textual Deduplication
        # -------------------------------------------------------------
        deduped = self._deduplicate_observations(raw_detections, investigation_id)

        # Store to database if db_manager is available in context
        db_manager: Optional[DatabaseManager] = context.get("db_manager") if context else None
        if db_manager:
            self._persist_observations(db_manager, deduped)

        # -------------------------------------------------------------
        # Convert Observations to Evidence Items
        # -------------------------------------------------------------
        for obs in deduped:
            coord_str = f"[{int(obs.box_original[0])},{int(obs.box_original[1])},{int(obs.box_original[2])},{int(obs.box_original[3])}]"
            evidence_text = (
                f'Recognized text: "{obs.text}" '
                f"(Category: {obs.category}, Script: {obs.language_script}, Conf: {obs.confidence:.2f}, Box: {coord_str})"
            )
            evidence_list.append(
                EvidenceItem(
                    investigation_id=investigation_id,
                    kind="ocr",
                    status="read",
                    text=evidence_text,
                    confidence_weight=round(obs.confidence, 2),
                )
            )

        logger.info(
            f"OCR Investigation complete for {image_record.filename}: "
            f"{len(raw_detections)} raw hits -> {len(deduped)} deduplicated observations."
        )
        return evidence_list

    def _deduplicate_observations(
        self,
        raw_items: List[Tuple[RawOCRBox, Tuple[float, float, float, float]]],
        investigation_id: str,
    ) -> List[OCRObservation]:
        """Merge spatial overlaps and identical text clusters into distinct observations."""
        sorted_items = sorted(raw_items, key=lambda x: x[0].confidence, reverse=True)
        clusters: List[Dict[str, Any]] = []

        for item, box in sorted_items:
            norm_text = re.sub(r"\s+", " ", item.text.strip().lower())
            matched_cluster = None

            for cluster in clusters:
                c_box = cluster["box"]
                c_norm_text = cluster["norm_text"]
                iou = compute_iou(box, c_box)

                # Match if boxes overlap substantially OR if boxes overlap moderately and strings match
                if iou > 0.45 or (iou > 0.20 and norm_text == c_norm_text):
                    matched_cluster = cluster
                    break

            if matched_cluster is not None:
                matched_cluster["passes"].append(item.pass_name)
                # Keep highest confidence text
                if item.confidence > matched_cluster["confidence"]:
                    matched_cluster["text"] = item.text
                    matched_cluster["norm_text"] = norm_text
                    matched_cluster["confidence"] = item.confidence
                    matched_cluster["box"] = box
            else:
                clusters.append(
                    {
                        "text": item.text,
                        "norm_text": norm_text,
                        "confidence": item.confidence,
                        "box": box,
                        "primary_pass": item.pass_name,
                        "passes": [item.pass_name],
                    }
                )

        observations: List[OCRObservation] = []
        for c in clusters:
            text = c["text"]
            cat = classify_geographic_clue(text)
            script = detect_script(text)
            observations.append(
                OCRObservation(
                    investigation_id=investigation_id,
                    text=text,
                    confidence=c["confidence"],
                    box_original=c["box"],
                    category=cat,
                    language_script=script,
                    pass_name=c["primary_pass"],
                    passes_detected=list(set(c["passes"])),
                )
            )

        return observations

    def _persist_observations(self, db: DatabaseManager, observations: List[OCRObservation]) -> None:
        """Persist OCR observations into SQLite."""
        with db.get_connection() as conn:
            for obs in observations:
                now_str = obs.created_at.isoformat()
                conn.execute(
                    """
                    INSERT INTO ocr_observations (
                        investigation_id, text, confidence, box_x0, box_y0, box_x1, box_y1,
                        category, language_script, pass_name, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        obs.investigation_id,
                        obs.text,
                        obs.confidence,
                        obs.box_original[0],
                        obs.box_original[1],
                        obs.box_original[2],
                        obs.box_original[3],
                        obs.category,
                        obs.language_script,
                        obs.pass_name,
                        now_str,
                    ),
                )
            conn.commit()
