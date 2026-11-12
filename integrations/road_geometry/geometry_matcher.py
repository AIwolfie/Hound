"""HOUND Road & Infrastructure Scene Geometry Matching Engine.

Extracts visual roadway geometry (heading, curvature, lane configurations, coastal adjacency)
from photographs and performs geometric alignment against OpenStreetMap vectors to localize
within transportation corridors.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from pydantic import BaseModel, Field

from core.geographic_search import EARTH_RADIUS_METERS, haversine_distance
from core.logger import get_logger
from core.spatial_envelope import GeographicResolutionLevel, SpatialEnvelope

logger = get_logger("hound.road_geometry")


class VisualRoadGeometry(BaseModel):
    """Visual geometry extracted from photographic road scene."""
    vanishing_point: Optional[Tuple[float, float]] = None  # (x, y) in normalized [0, 1] coords
    estimated_camera_heading_offset_deg: float = 0.0      # Offset from road centerline
    curvature_type: str = "straight"                      # "straight", "curve_left", "curve_right", "sharp_turn"
    curvature_radius_m: float = 2000.0                    # Estimated curvature radius in meters
    lane_count: int = 2                                   # Estimated number of lanes
    marking_type: str = "dashed_white"                    # "dashed_white", "solid_yellow", "double_yellow", "none"
    has_median_barrier: bool = False
    coastal_water_side: Optional[str] = None              # "left", "right", "none"
    confidence: float = 0.80


class RoadAlignmentMatch(BaseModel):
    """Match result between visual road geometry and OpenStreetMap corridor segment."""
    segment_id: str
    center_lat: float
    center_lon: float
    segment_heading_deg: float
    segment_curvature_type: str
    curvature_match_score: float   # [0.0, 1.0]
    lane_match_score: float        # [0.0, 1.0]
    coastal_match_score: float     # [0.0, 1.0]
    composite_geometric_score: float # [0.0, 1.0]
    refined_spatial_envelope: SpatialEnvelope
    justification: str


class RoadGeometryMatcher:
    """Extracts roadway geometric parameters and aligns them with OpenStreetMap corridor vectors."""

    @staticmethod
    def extract_visual_geometry(image_np: np.ndarray) -> VisualRoadGeometry:
        """Extract road heading, curvature, and lane parameters from an RGB image."""
        if image_np is None or image_np.size == 0:
            return VisualRoadGeometry()

        h, w = image_np.shape[:2]
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if len(image_np.shape) == 3 else image_np

        # Focus on lower two-thirds of image where road plane lies
        roi = gray[int(h * 0.40) : h, :]
        roi_h, roi_w = roi.shape[:2]

        # 1. Edge and Line Detection
        edges = cv2.Canny(roi, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=20)

        left_lines = []
        right_lines = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                # Left lane lines: negative slope in image coordinates
                if slope < -0.3 and x1 < roi_w * 0.6:
                    left_lines.append(line[0])
                # Right lane lines: positive slope
                elif slope > 0.3 and x2 > roi_w * 0.4:
                    right_lines.append(line[0])

        # 2. Curvature Estimation via Lane Slope Discrepancy
        curvature_type = "straight"
        curvature_radius_m = 2500.0

        if left_lines and right_lines:
            mean_left_slope = np.mean([(y2 - y1) / (x2 - x1) for x1, y1, x2, y2 in left_lines])
            mean_right_slope = np.mean([(y2 - y1) / (x2 - x1) for x1, y1, x2, y2 in right_lines])

            # Asymmetry between left and right line angles indicates perspective curvature
            slope_diff = abs(mean_left_slope) - abs(mean_right_slope)
            if slope_diff > 0.35:
                curvature_type = "curve_right"
                curvature_radius_m = 650.0
            elif slope_diff < -0.35:
                curvature_type = "curve_left"
                curvature_radius_m = 650.0
            elif abs(slope_diff) > 0.70:
                curvature_type = "sharp_turn"
                curvature_radius_m = 250.0
        elif left_lines and not right_lines:
            curvature_type = "curve_left"
            curvature_radius_m = 900.0
        elif right_lines and not left_lines:
            curvature_type = "curve_right"
            curvature_radius_m = 900.0

        # 3. Coastal Water Adjacency Check
        # Inspect left and right upper quadrants for high-saturation / blue ocean hue (RGB or BGR agnostic)
        coastal_water_side = "none"
        if len(image_np.shape) == 3:
            hsv1 = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
            hsv2 = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
            for hsv in (hsv1, hsv2):
                left_quad = hsv[: int(h * 0.6), : int(w * 0.35)]
                right_quad = hsv[: int(h * 0.6), int(w * 0.65) :]

                blue_left = np.sum((left_quad[:, :, 0] >= 95) & (left_quad[:, :, 0] <= 135) & (left_quad[:, :, 1] >= 40))
                blue_right = np.sum((right_quad[:, :, 0] >= 95) & (right_quad[:, :, 0] <= 135) & (right_quad[:, :, 1] >= 40))

                left_ratio = blue_left / max(1, left_quad.size // 3)
                right_ratio = blue_right / max(1, right_quad.size // 3)

                if left_ratio > 0.15 and left_ratio > right_ratio * 1.4:
                    coastal_water_side = "left"
                    break
                elif right_ratio > 0.15 and right_ratio > left_ratio * 1.4:
                    coastal_water_side = "right"
                    break

        # 4. Lane Count Estimation based on road width in ROI
        total_detected_lines = len(left_lines) + len(right_lines)
        lane_count = 2
        if total_detected_lines >= 8:
            lane_count = 4
        elif total_detected_lines >= 14:
            lane_count = 6

        return VisualRoadGeometry(
            curvature_type=curvature_type,
            curvature_radius_m=curvature_radius_m,
            lane_count=lane_count,
            coastal_water_side=coastal_water_side if coastal_water_side != "none" else None,
            confidence=0.85 if (left_lines or right_lines) else 0.50,
        )

    @classmethod
    def match_geometry_to_corridor(
        cls,
        visual_geo: VisualRoadGeometry,
        corridor_waypoints: List[Tuple[float, float]],
        corridor_name: str = "Target Highway",
        candidate_segment_length_m: float = 3000.0,
    ) -> List[RoadAlignmentMatch]:
        """Align extracted visual geometry against OpenStreetMap road waypoints along a corridor.

        Splits the corridor into discrete segments, computes their geographical heading and curvature,
        and scores compatibility against the observed visual geometry.
        """
        if len(corridor_waypoints) < 3:
            return []

        matches: List[RoadAlignmentMatch] = []

        # Iterate along waypoints in sliding windows of ~3-5 points (segments)
        window_size = 4
        step = 2

        for i in range(0, len(corridor_waypoints) - window_size + 1, step):
            window = corridor_waypoints[i : i + window_size]
            p_start = window[0]
            p_mid = window[len(window) // 2]
            p_end = window[-1]

            # 1. Segment Heading (degrees from True North: 0 - 360)
            d_lon = math.radians(p_end[1] - p_start[1])
            lat1 = math.radians(p_start[0])
            lat2 = math.radians(p_end[0])
            y = math.sin(d_lon) * math.cos(lat2)
            x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
            bearing_deg = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

            # 2. Segment Curvature (Turn Angle between first half and second half)
            y1 = math.sin(math.radians(p_mid[1] - p_start[1])) * math.cos(math.radians(p_mid[0]))
            x1 = math.cos(lat1) * math.sin(math.radians(p_mid[0])) - math.sin(lat1) * math.cos(math.radians(p_mid[0])) * math.cos(math.radians(p_mid[1] - p_start[1]))
            h1 = (math.degrees(math.atan2(y1, x1)) + 360.0) % 360.0

            y2 = math.sin(math.radians(p_end[1] - p_mid[1])) * math.cos(lat2)
            x2 = math.cos(math.radians(p_mid[0])) * math.sin(lat2) - math.sin(math.radians(p_mid[0])) * math.cos(lat2) * math.cos(math.radians(p_end[1] - p_mid[1]))
            h2 = (math.degrees(math.atan2(y2, x2)) + 360.0) % 360.0

            turn_angle = (h2 - h1 + 540.0) % 360.0 - 180.0  # Positive = Turn Right, Negative = Turn Left

            seg_curvature = "straight"
            if turn_angle > 6.0:
                seg_curvature = "curve_right"
            elif turn_angle < -6.0:
                seg_curvature = "curve_left"
            if abs(turn_angle) > 25.0:
                seg_curvature = "sharp_turn"

            # 3. Compute Curvature Compatibility Score
            curv_score = 0.50
            if visual_geo.curvature_type == seg_curvature:
                curv_score = 0.95
            elif visual_geo.curvature_type == "straight" and abs(turn_angle) < 8.0:
                curv_score = 0.90
            elif "curve" in visual_geo.curvature_type and "curve" in seg_curvature:
                curv_score = 0.80 if visual_geo.curvature_type == seg_curvature else 0.20
            else:
                curv_score = 0.30

            # 4. Coastal Adjacency Compatibility
            coastal_score = 1.0
            if visual_geo.coastal_water_side:
                # Pacific coast in North America: traveling North has ocean on Left (West)
                # Traveling South has ocean on Right (West)
                is_traveling_north = (bearing_deg >= 315.0 or bearing_deg <= 45.0)
                is_traveling_south = (135.0 <= bearing_deg <= 225.0)

                if visual_geo.coastal_water_side == "left" and is_traveling_north:
                    coastal_score = 0.95
                elif visual_geo.coastal_water_side == "right" and is_traveling_south:
                    coastal_score = 0.95
                else:
                    coastal_score = 0.65

            lane_score = 0.85
            composite = 0.55 * curv_score + 0.30 * coastal_score + 0.15 * lane_score

            # Create refined spatial envelope for this matching segment
            refined_env = SpatialEnvelope.from_point_and_radius(
                lat=p_mid[0],
                lon=p_mid[1],
                radius_m=candidate_segment_length_m,
                level=GeographicResolutionLevel.DISTRICT,
                area_name=f"{corridor_name} (Segment near {p_mid[0]:.4f}, {p_mid[1]:.4f})",
                modalities=["road_geometry", "osm_alignment"],
                justification=(
                    f"Aligned visual road curvature ({visual_geo.curvature_type}) and bearing "
                    f"({bearing_deg:.1f}°) with OSM way geometry (Composite Score: {composite:.2f})."
                ),
            )

            matches.append(
                RoadAlignmentMatch(
                    segment_id=f"SEG-{i}",
                    center_lat=p_mid[0],
                    center_lon=p_mid[1],
                    segment_heading_deg=round(bearing_deg, 1),
                    segment_curvature_type=seg_curvature,
                    curvature_match_score=round(curv_score, 3),
                    lane_match_score=lane_score,
                    coastal_match_score=round(coastal_score, 3),
                    composite_geometric_score=round(composite, 3),
                    refined_spatial_envelope=refined_env,
                    justification=refined_env.justification,
                )
            )

        matches.sort(key=lambda m: m.composite_geometric_score, reverse=True)
        return matches
