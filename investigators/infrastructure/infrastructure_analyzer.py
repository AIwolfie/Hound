"""HOUND Infrastructure Investigator Module.

Coordinates infrastructure feature detectors, maps visual observations into the
evidence store, and persists detection records to SQLite.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.image_loader import ImageRecord
from core.logger import get_logger
from investigators.base import BaseInvestigator
from investigators.infrastructure.base_detector import BaseInfrastructureDetector, DetectedFeature
from investigators.infrastructure.classical_detector import ClassicalInfrastructureDetector
from storage.database import DatabaseManager
from storage.evidence_store import EvidenceItem

logger = get_logger("hound.investigators.infrastructure")


class InfrastructureAnalyzer(BaseInvestigator):
    """Analyzes utility poles, road signs, traffic markers, and physical fixtures."""

    def __init__(self, detectors: Optional[List[BaseInfrastructureDetector]] = None):
        super().__init__(name="infrastructure_analyzer", category="infrastructure")
        self.detectors = detectors or [ClassicalInfrastructureDetector()]

    def run(
        self,
        investigation_id: str,
        image_record: ImageRecord,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[EvidenceItem]:
        evidence_list: List[EvidenceItem] = []
        all_features: List[DetectedFeature] = []

        # Run registered detectors
        for detector in self.detectors:
            try:
                features = detector.detect(image_record.original_path)
                all_features.extend(features)
            except Exception as e:
                logger.error(f"Detector {detector.name} failed: {e}")

        # Persist detections to database if db_manager is provided
        db_manager: Optional[DatabaseManager] = context.get("db_manager") if context else None
        if db_manager and all_features:
            with db_manager.get_connection() as conn:
                for feat in all_features:
                    now_str = datetime.now(timezone.utc).isoformat()
                    conn.execute(
                        """
                        INSERT INTO infrastructure_detections (
                            investigation_id, class_name, confidence, box_x0, box_y0, box_x1, box_y1,
                            detector_backend, details, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            investigation_id,
                            feat.class_name,
                            feat.confidence,
                            feat.box[0],
                            feat.box[1],
                            feat.box[2],
                            feat.box[3],
                            feat.detector_backend,
                            json.dumps(feat.attributes),
                            now_str,
                        ),
                    )
                conn.commit()

        # Group detected features into structured visual evidence observations
        grouped: Dict[str, List[DetectedFeature]] = {}
        for f in all_features:
            grouped.setdefault(f.class_name, []).append(f)

        for class_name, feats in grouped.items():
            count = len(feats)
            avg_conf = sum(f.confidence for f in feats) / float(count)
            boxes_str = "; ".join(
                f"[{int(f.box[0])},{int(f.box[1])},{int(f.box[2])},{int(f.box[3])}]" for f in feats[:3]
            )
            if count > 3:
                boxes_str += f"; +{count - 3} more"

            readable_name = class_name.replace("_", " ").title()
            text = (
                f"[Visual Infrastructure] Detected {count} {readable_name} instance(s) "
                f"(Avg Conf: {avg_conf:.2f}, Boxes: {boxes_str}). "
                f"{feats[0].description}"
            )

            evidence_list.append(
                EvidenceItem(
                    investigation_id=investigation_id,
                    kind="visual",
                    status="observed",
                    text=text,
                    confidence_weight=round(avg_conf, 2),
                )
            )

        logger.info(
            f"Infrastructure analysis complete for {image_record.filename}: "
            f"{len(all_features)} detected features across {len(grouped)} classes."
        )
        return evidence_list
