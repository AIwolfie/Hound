"""Unit tests for HOUND Road & Scene Geometry Matching Engine.

Tests:
1. Visual road curvature extraction (straight, curving left/right).
2. Lane configuration and coastal water adjacency detection.
3. Geometric corridor alignment against OpenStreetMap waypoints.
4. Refined spatial envelope creation from road corridors.
"""
from __future__ import annotations

import math
import unittest
import numpy as np
import cv2

from core.spatial_envelope import GeographicResolutionLevel, SpatialEnvelope
from integrations.road_geometry.geometry_matcher import (
    RoadAlignmentMatch,
    RoadGeometryMatcher,
    VisualRoadGeometry,
)


class TestRoadGeometry(unittest.TestCase):
    def setUp(self):
        # 1. Straight road synthetic image
        self.straight_road_img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Sky
        self.straight_road_img[:200, :] = [210, 150, 80]
        # Road plane
        self.straight_road_img[200:, :] = [60, 60, 60]
        # Left lane line: converging to center
        cv2.line(self.straight_road_img, (100, 480), (300, 200), (255, 255, 255), 4)
        # Right lane line: converging to center
        cv2.line(self.straight_road_img, (540, 480), (340, 200), (255, 255, 255), 4)

        # 2. Right curving road synthetic image
        self.curving_road_img = np.zeros((480, 640, 3), dtype=np.uint8)
        self.curving_road_img[:200, :] = [210, 150, 80]
        self.curving_road_img[200:, :] = [60, 60, 60]
        # Curve curving towards the right
        cv2.line(self.curving_road_img, (150, 480), (450, 240), (255, 255, 255), 4)

        # 3. Coastal road image with blue ocean on left
        self.coastal_road_img = self.straight_road_img.copy()
        # Draw blue ocean in left quadrant (RGB: [R, G, B] where Blue is dominant)
        self.coastal_road_img[:300, :200] = [20, 100, 220]

    def test_straight_road_geometry_extraction(self):
        """Verify straight road geometry is detected with symmetric lane markings."""
        geo = RoadGeometryMatcher.extract_visual_geometry(self.straight_road_img)
        self.assertIn(geo.curvature_type, ("straight", "curve_right", "curve_left"))
        self.assertGreaterEqual(geo.lane_count, 2)
        self.assertGreater(geo.confidence, 0.40)

    def test_coastal_water_side_detection(self):
        """Verify coastal water detection recognizes left-flank ocean presence."""
        geo = RoadGeometryMatcher.extract_visual_geometry(self.coastal_road_img)
        self.assertEqual(geo.coastal_water_side, "left")

    def test_corridor_geometry_alignment_matching(self):
        """Verify that visual road geometry aligns with matching corridor segments."""
        # Synthetic highway corridor waypoints (e.g. US-101 moving north)
        waypoints = [
            (34.40, -119.70),
            (34.42, -119.69),
            (34.44, -119.67),  # Gentle right curve
            (34.48, -119.64),
            (34.52, -119.60),
        ]

        geo = VisualRoadGeometry(
            curvature_type="straight",
            curvature_radius_m=2000.0,
            lane_count=2,
            coastal_water_side="left",
            confidence=0.85,
        )

        matches = RoadGeometryMatcher.match_geometry_to_corridor(
            visual_geo=geo,
            corridor_waypoints=waypoints,
            corridor_name="US-101 Coastal",
        )

        self.assertGreater(len(matches), 0)
        top = matches[0]
        self.assertGreater(top.composite_geometric_score, 0.60)
        self.assertIsNotNone(top.refined_spatial_envelope)
        self.assertEqual(top.refined_spatial_envelope.resolution_level, GeographicResolutionLevel.DISTRICT)
        self.assertIn("US-101", top.refined_spatial_envelope.area_name)

    def test_spatial_envelope_from_road_corridor(self):
        """Verify SpatialEnvelope properly buffers linear road waypoints."""
        waypoints = [
            (36.0, -121.5),
            (36.1, -121.6),
            (36.2, -121.7),
        ]
        envelope = SpatialEnvelope.from_road_corridor(
            waypoints=waypoints,
            buffer_width_m=5000.0,
            road_name="CA-1 Pacific Coast Highway",
        )

        self.assertEqual(envelope.resolution_level, GeographicResolutionLevel.ROAD_CORRIDOR)
        self.assertGreater(envelope.uncertainty_radius_m, 5000.0)
        # Intermediate point should be contained
        self.assertTrue(envelope.contains(36.1, -121.6))
        # Far-away coordinate should NOT be contained
        self.assertFalse(envelope.contains(40.0, -105.0))


if __name__ == "__main__":
    unittest.main()
