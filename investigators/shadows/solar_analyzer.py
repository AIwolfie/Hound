"""HOUND Solar Ephemeris and Shadow Geometry Engine.

Calculates astronomical solar positions (elevation, azimuth), analyzes cast shadows,
derives solar constraints, and estimates latitude bands while strictly recording uncertainty.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from pydantic import BaseModel, Field

from core.image_loader import ImageRecord
from core.logger import get_logger
from investigators.base import BaseInvestigator
from storage.database import DatabaseManager
from storage.evidence_store import EvidenceItem

logger = get_logger("hound.solar")


class SolarPosition(BaseModel):
    elevation_deg: float
    azimuth_deg: float  # Degrees from True North (0° = North, 90° = East, 180° = South, 270° = West)
    declination_deg: float
    hour_angle_deg: float
    shadow_length_ratio: Optional[float] = None  # L / H = 1 / tan(elevation)


class ShadowCandidate(BaseModel):
    contour_id: int
    center_point: Tuple[float, float]
    area_px: float
    length_px: float
    width_px: float
    aspect_ratio: float
    orientation_deg: float  # Angle in image space [-90°, +90°]
    confidence: float
    notes: str


class SolarAnalyzer(BaseInvestigator):
    """Calculates solar position, examines cast shadows, and derives solar constraints."""

    def __init__(self):
        super().__init__(name="solar_analyzer", category="shadows")

    @staticmethod
    def calculate_solar_declination(day_of_year: int, hour_fraction: float = 12.0) -> float:
        """Compute solar declination angle in degrees for given day of year."""
        gamma = 2.0 * math.pi / 365.0 * (day_of_year - 1 + (hour_fraction - 12.0) / 24.0)
        decl_rad = (
            0.006918
            - 0.399912 * math.cos(gamma)
            + 0.070257 * math.sin(gamma)
            - 0.006758 * math.cos(2.0 * gamma)
            + 0.000907 * math.sin(2.0 * gamma)
            - 0.002697 * math.cos(3.0 * gamma)
            + 0.001480 * math.sin(3.0 * gamma)
        )
        return math.degrees(decl_rad)

    @classmethod
    def compute_max_solar_elevation(
        cls, lat_deg: float, day_of_year: Optional[int] = None
    ) -> float:
        """Compute theoretical maximum possible solar elevation at a given latitude.

        At solar noon (hour angle H=0), sin(h_max) = sin(phi)*sin(delta) + cos(phi)*cos(delta)
        = cos(phi - delta) = sin(90° - |phi - delta|).
        Thus h_max = 90.0° - |phi - delta|.

        If day_of_year is unknown, delta can vary between -23.44° and +23.44° throughout the year:
        - Inside tropics (|phi| <= 23.44°): sun reaches zenith (90.0°) twice a year.
        - Outside tropics (|phi| > 23.44°): max annual elevation is 90° - (|phi| - 23.44°) = 113.44° - |phi|.
        """
        if day_of_year is not None:
            decl_deg = cls.calculate_solar_declination(day_of_year, hour_fraction=12.0)
            noon_elev = 90.0 - abs(lat_deg - decl_deg)
            return max(0.0, min(90.0, round(noon_elev, 2)))

        abs_lat = abs(lat_deg)
        if abs_lat <= 23.44:
            return 90.0
        return round(113.44 - abs_lat, 2)

    @classmethod
    def is_solar_elevation_feasible(
        cls,
        lat_deg: float,
        observed_elev_deg: float,
        day_of_year: Optional[int] = None,
        tolerance_deg: float = 3.0,
    ) -> Tuple[bool, float, str]:
        """Evaluate if an observed sun elevation is astronomically possible at latitude lat_deg."""
        max_possible = cls.compute_max_solar_elevation(lat_deg, day_of_year=day_of_year)
        if observed_elev_deg > (max_possible + tolerance_deg):
            season_info = f"day of year {day_of_year}" if day_of_year is not None else "any time of year"
            return (
                False,
                max_possible,
                f"Observed sun elevation {observed_elev_deg:.1f}° exceeds maximum possible "
                f"astronomical elevation {max_possible:.1f}° at latitude {lat_deg:.1f}° ({season_info}).",
            )
        return (True, max_possible, "")

    @staticmethod
    def calculate_solar_position(
        lat_deg: float,
        lon_deg: float,
        dt_utc: datetime,
    ) -> SolarPosition:
        """Compute solar elevation and azimuth using NOAA astronomical algorithms.

        Args:
            lat_deg: Latitude in decimal degrees [-90 to +90].
            lon_deg: Longitude in decimal degrees [-180 to +180].
            dt_utc: UTC datetime of capture.
        """
        # Day of year and fractional hour
        day_of_year = dt_utc.timetuple().tm_yday
        hour_fraction = dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0

        # Fractional year in radians
        gamma = 2.0 * math.pi / 365.0 * (day_of_year - 1 + (hour_fraction - 12.0) / 24.0)

        # Equation of time (in minutes)
        eot = 229.18 * (
            0.000075
            + 0.001868 * math.cos(gamma)
            - 0.032077 * math.sin(gamma)
            - 0.014615 * math.cos(2.0 * gamma)
            - 0.040849 * math.sin(2.0 * gamma)
        )

        # Solar declination angle (in radians)
        decl = (
            0.006918
            - 0.399912 * math.cos(gamma)
            + 0.070257 * math.sin(gamma)
            - 0.006758 * math.cos(2.0 * gamma)
            + 0.000907 * math.sin(2.0 * gamma)
            - 0.002697 * math.cos(3.0 * gamma)
            + 0.001480 * math.sin(3.0 * gamma)
        )

        # Time offset in minutes
        time_offset = eot + 4.0 * lon_deg
        true_solar_time = (hour_fraction * 60.0 + time_offset) % 1440.0

        # Solar hour angle (in degrees and radians)
        hour_angle_deg = (true_solar_time / 4.0) - 180.0
        hour_angle_rad = math.radians(hour_angle_deg)

        lat_rad = math.radians(lat_deg)

        # Solar zenith and elevation
        cos_zenith = math.sin(lat_rad) * math.sin(decl) + math.cos(lat_rad) * math.cos(decl) * math.cos(hour_angle_rad)
        cos_zenith = max(-1.0, min(1.0, cos_zenith))
        zenith_rad = math.acos(cos_zenith)
        elevation_rad = (math.pi / 2.0) - zenith_rad
        elevation_deg = math.degrees(elevation_rad)

        # Solar azimuth
        cos_azimuth = (math.sin(decl) - math.sin(lat_rad) * math.sin(elevation_rad)) / (
            math.cos(lat_rad) * math.cos(elevation_rad) + 1e-9
        )
        cos_azimuth = max(-1.0, min(1.0, cos_azimuth))
        azimuth_rad = math.acos(cos_azimuth)
        azimuth_deg = math.degrees(azimuth_rad)

        if hour_angle_deg > 0.0:
            azimuth_deg = 360.0 - azimuth_deg

        # Shadow length ratio for a vertical post on level ground
        shadow_ratio = None
        if elevation_deg > 1.0:
            shadow_ratio = 1.0 / math.tan(elevation_rad)

        return SolarPosition(
            elevation_deg=round(elevation_deg, 2),
            azimuth_deg=round(azimuth_deg, 2),
            declination_deg=round(math.degrees(decl), 2),
            hour_angle_deg=round(hour_angle_deg, 2),
            shadow_length_ratio=round(shadow_ratio, 3) if shadow_ratio is not None else None,
        )

    def detect_candidate_shadows(
        self,
        image_path: Path | str,
        min_area: float = 400.0,
        min_aspect_ratio: float = 2.5,
    ) -> List[ShadowCandidate]:
        """Detect elongated dark candidate shadow contours using adaptive thresholding and PCA."""
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return []

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Shadows exhibit low brightness (V) and relatively low/moderate saturation
        _, dark_mask = cv2.threshold(v, 70, 255, cv2.THRESH_BINARY_INV)

        # Morphological clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[ShadowCandidate] = []

        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            # Minimum area bounding box
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), angle = rect
            length = max(rw, rh)
            width = max(1.0, min(rw, rh))
            aspect_ratio = length / width

            if aspect_ratio >= min_aspect_ratio:
                # PCA for dominant orientation vector
                pts = cnt.reshape(-1, 2).astype(np.float32)
                mean, eigenvectors = cv2.PCACompute(pts, mean=None)
                primary_vector = eigenvectors[0]
                rad = math.atan2(primary_vector[1], primary_vector[0])
                deg = math.degrees(rad)

                # Confidence heuristic based on elongation and area
                conf = min(0.85, 0.40 + (aspect_ratio / 10.0) * 0.40)

                candidates.append(
                    ShadowCandidate(
                        contour_id=idx,
                        center_point=(round(float(cx), 1), round(float(cy), 1)),
                        area_px=float(area),
                        length_px=round(float(length), 1),
                        width_px=round(float(width), 1),
                        aspect_ratio=round(float(aspect_ratio), 2),
                        orientation_deg=round(deg, 1),
                        confidence=round(conf, 2),
                        notes="Elongated low-luminance candidate shadow region.",
                    )
                )

        # Sort by area descending, return top candidates
        candidates.sort(key=lambda c: c.area_px, reverse=True)
        return candidates[:15]

    def run(
        self,
        investigation_id: str,
        image_record: ImageRecord,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[EvidenceItem]:
        """Execute solar geometry and shadow investigation."""
        evidence_list: List[EvidenceItem] = []
        db_manager: Optional[DatabaseManager] = context.get("db_manager") if context else None

        # 1. Ephemeris Calculation if Timestamp and Estimated/Known Coordinates are present
        dt = image_record.exif.datetime_original
        gps = image_record.exif.gps

        if dt and gps:
            dt_utc = None
            has_tz = False
            if image_record.exif.offset_time:
                import re
                from datetime import timedelta
                m = re.match(r"([+-])(\d{2}):?(\d{2})", image_record.exif.offset_time)
                if m:
                    sign = -1 if m.group(1) == "-" else 1
                    hrs, mins = int(m.group(2)), int(m.group(3))
                    tz = timezone(sign * timedelta(hours=hrs, minutes=mins))
                    dt_utc = dt.replace(tzinfo=tz).astimezone(timezone.utc)
                    has_tz = True
            if dt_utc is None:
                if dt.tzinfo is not None:
                    dt_utc = dt.astimezone(timezone.utc)
                    has_tz = True
                else:
                    dt_utc = dt.replace(tzinfo=timezone.utc)
                    has_tz = False

            pos = self.calculate_solar_position(gps.latitude, gps.longitude, dt_utc)
            tz_note = "EXIF UTC Offset verified" if has_tz else "EXIF OffsetTime absent (UTC assumed)"

            pos_text = (
                f"[Computed Solar Ephemeris] Sun Elevation: {pos.elevation_deg:.1f}°, "
                f"Sun Azimuth: {pos.azimuth_deg:.1f}° (True North). "
                f"Expected vertical shadow length ratio: {pos.shadow_length_ratio if pos.shadow_length_ratio else 'N/A'}x. "
                f"Declination: {pos.declination_deg:.1f}°. Timezone: {tz_note}."
            )
            evidence_list.append(
                EvidenceItem(
                    investigation_id=investigation_id,
                    kind="sun",
                    status="computed",
                    text=pos_text,
                    confidence_weight=1.0,
                )
            )

        # 2. Computer Vision Shadow Candidate Extraction
        shadows = self.detect_candidate_shadows(image_record.original_path)
        if shadows:
            summary = (
                f"[Visual Shadow Analysis] Detected {len(shadows)} elongated dark candidate shadow contours. "
                f"Dominant candidate axis: {shadows[0].orientation_deg:.1f}° relative to image frame "
                f"(Aspect Ratio: {shadows[0].aspect_ratio:.1f}:1, Confidence: {shadows[0].confidence:.2f}). "
                f"Note: Unanchored to geographic north; requires landmark alignment or compass heading."
            )
            evidence_list.append(
                EvidenceItem(
                    investigation_id=investigation_id,
                    kind="sun",
                    status="observed",
                    text=summary,
                    confidence_weight=0.7,
                )
            )

            # Persist to shadow_observations if db_manager available
            if db_manager:
                with db_manager.get_connection() as conn:
                    for s in shadows:
                        now_str = datetime.now(timezone.utc).isoformat()
                        conn.execute(
                            """
                            INSERT INTO shadow_observations (
                                investigation_id, base_x, base_y, tip_x, tip_y,
                                estimated_azimuth_deg, uncertainty_deg, length_ratio, notes, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                investigation_id,
                                s.center_point[0],
                                s.center_point[1],
                                None,
                                None,
                                s.orientation_deg,
                                20.0,
                                s.aspect_ratio,
                                s.notes,
                                now_str,
                            ),
                        )
                    conn.commit()

        return evidence_list


def calculate_solar_declination(day_of_year: int, hour_fraction: float = 12.0) -> float:
    """Module-level helper to calculate solar declination in degrees."""
    return SolarAnalyzer.calculate_solar_declination(day_of_year, hour_fraction)


def compute_max_solar_elevation(lat_deg: float, day_of_year: Optional[int] = None) -> float:
    """Module-level helper to compute maximum theoretical solar elevation."""
    return SolarAnalyzer.compute_max_solar_elevation(lat_deg, day_of_year)


def is_solar_elevation_feasible(
    lat_deg: float,
    observed_elev_deg: float,
    day_of_year: Optional[int] = None,
    tolerance_deg: float = 3.0,
) -> Tuple[bool, float, str]:
    """Module-level helper evaluating solar elevation feasibility."""
    return SolarAnalyzer.is_solar_elevation_feasible(lat_deg, observed_elev_deg, day_of_year, tolerance_deg)
