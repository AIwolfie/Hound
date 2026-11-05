"""HOUND EXIF Metadata Investigator.

Transforms extracted EXIF tags and integrity assessments into structured evidence items.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
from core.image_loader import ImageRecord
from investigators.base import BaseInvestigator
from storage.evidence_store import EvidenceItem


class ExifAnalyzer(BaseInvestigator):
    """Analyzes image EXIF metadata, camera optics, timestamps, and GPS."""

    def __init__(self):
        super().__init__(name="exif_analyzer", category="metadata")

    def run(
        self,
        investigation_id: str,
        image_record: ImageRecord,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[EvidenceItem]:
        evidence_list: List[EvidenceItem] = []
        exif = image_record.exif
        integrity = image_record.integrity

        # 1. Format and Dimensions
        evidence_list.append(
            EvidenceItem(
                investigation_id=investigation_id,
                kind="metadata",
                status="read",
                text=(
                    f"Image format: {image_record.format}, Dimensions: {image_record.width}x{image_record.height} "
                    f"({image_record.aspect_ratio:.2f}:1), Size: {image_record.file_size_bytes:,} bytes"
                ),
                confidence_weight=1.0,
            )
        )

        # 2. Integrity and Sanitization Findings
        for finding in integrity.findings:
            status = "read" if "Camera identified" in finding or "timestamp recorded" in finding else "inferred"
            evidence_list.append(
                EvidenceItem(
                    investigation_id=investigation_id,
                    kind="metadata",
                    status=status,
                    text=f"[Metadata Integrity] {finding}",
                    confidence_weight=1.0,
                )
            )

        # 3. Embedded GPS Coordinates
        if exif.gps:
            gps = exif.gps
            coord_str = gps.to_formatted_str()
            detail = f"Embedded GPS coordinates: {coord_str} ({gps.latitude:.7f}, {gps.longitude:.7f})"
            if gps.altitude_m is not None:
                detail += f", Altitude: {gps.altitude_m:.1f} m"
            if gps.heading_deg is not None:
                detail += f", Camera Heading: {gps.heading_deg:.1f}°"

            evidence_list.append(
                EvidenceItem(
                    investigation_id=investigation_id,
                    kind="exif",
                    status="read",
                    text=detail,
                    confidence_weight=1.0,
                )
            )

        # 4. Optical Analysis & Field of View Estimation
        if exif.focal_length_35mm:
            # Horizontal Field of View: 2 * arctan(36 / (2 * f_35mm))
            fov_rad = 2.0 * math.atan(36.0 / (2.0 * exif.focal_length_35mm))
            fov_deg = math.degrees(fov_rad)
            evidence_list.append(
                EvidenceItem(
                    investigation_id=investigation_id,
                    kind="metadata",
                    status="computed",
                    text=(
                        f"Equivalent 35mm focal length: {exif.focal_length_35mm}mm. "
                        f"Computed horizontal FOV: {fov_deg:.1f}°"
                    ),
                    confidence_weight=1.0,
                )
            )

        # 5. Temporal Analysis
        if exif.datetime_original:
            ts = exif.datetime_original
            month_names = [
                "", "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December",
            ]
            month_str = month_names[ts.month]
            evidence_list.append(
                EvidenceItem(
                    investigation_id=investigation_id,
                    kind="metadata",
                    status="read",
                    text=(
                        f"Capture timestamp: {ts.strftime('%Y-%m-%d %H:%M:%S')} "
                        f"({month_str}, Month {ts.month})"
                    ),
                    confidence_weight=1.0,
                )
            )

        return evidence_list
