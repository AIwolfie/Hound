"""Unit tests for HOUND Advanced Terrain & Mountain Skyline Intelligence.

Tests:
1. Automated photographic skyline boundary extraction and relief variance.
2. Multi-parametric cross-correlation (heading, camera roll, camera pitch).
3. Flat terrain relief safeguard (rejecting spurious matches when relief < 0.4°).
"""
from __future__ import annotations

import math
import unittest
import numpy as np
import cv2

from core.spatial_envelope import GeographicResolutionLevel
from integrations.elevation.advanced_skyline import (
    AdvancedSkylineExtractor,
    AdvancedSkylineMatcher,
    SkylineProfile,
)
from integrations.elevation.horizon_generator import HorizonProfile


class TestAdvancedSkyline(unittest.TestCase):
    def setUp(self):
        # 1. Create a synthetic image of a jagged mountain ridge
        self.mountain_img = np.zeros((300, 400, 3), dtype=np.uint8)
        # Blue sky on top
        self.mountain_img[:, :] = [210, 180, 120]  # Light sky

        # Draw a jagged mountain silhouette with high vertical relief
        points = []
        for x in range(400):
            # Two peaks: sin waves + high relief
            y = int(120 + 40 * math.sin(x * 0.02) + 25 * math.cos(x * 0.05))
            points.append((x, y))

        pts_arr = np.array(points, dtype=np.int32)
        poly = np.vstack([pts_arr, [399, 299], [0, 299]])
        cv2.fillPoly(self.mountain_img, [poly], [40, 50, 40])  # Dark rocky mountain fill

        # 2. Create a flat landscape image (horizontal horizon line)
        self.flat_img = np.zeros((300, 400, 3), dtype=np.uint8)
        self.flat_img[:150, :] = [220, 180, 120]
        self.flat_img[150:, :] = [40, 90, 40]

    def test_photographic_skyline_extraction(self):
        """Verify automated extractor captures jagged silhouette and computes relief variance."""
        profile = AdvancedSkylineExtractor.extract_skyline_from_image(self.mountain_img, assumed_fov_deg=60.0)

        self.assertEqual(len(profile.relative_bearings_deg), 128)
        self.assertEqual(len(profile.apparent_elevation_deg), 128)
        # Jagged peaks should produce significant relief variance (> 1.0°)
        self.assertGreater(profile.relief_variance_deg, 0.80)
        self.assertAlmostEqual(profile.field_of_view_deg, 60.0)

    def test_multi_parametric_skyline_cross_correlation(self):
        """Verify matcher solves for optimal heading and camera pitch against synthetic DEM profile."""
        profile = AdvancedSkylineExtractor.extract_skyline_from_image(self.mountain_img, assumed_fov_deg=60.0)

        # Construct matching 360° DEM horizon with a corresponding mountain peak around 90° azimuth
        bearings = [float(b) for b in range(360)]
        angles = []
        for b in bearings:
            if 60.0 <= b <= 120.0:
                # Corresponding peak
                angle = 8.0 + 5.0 * math.sin(math.radians(b * 3.0))
            else:
                angle = 1.0 + 0.5 * math.cos(math.radians(b))
            angles.append(angle)

        dem_horizon = HorizonProfile(
            center_coord=(46.0, 7.5),
            observer_elevation_m=1600.0,
            bearings_deg=bearings,
            elevation_angles_deg=angles,
            peak_azimuth_deg=90.0,
            peak_elevation_deg=13.0,
            max_distance_m=20000.0,
        )

        match_res = AdvancedSkylineMatcher.match_skyline(
            observed=profile,
            dem_horizon=dem_horizon,
            heading_search_step_deg=2.0,
        )

        self.assertIsNotNone(match_res)
        self.assertGreater(match_res.correlation_score, 0.15)
        self.assertIsNotNone(match_res.refined_spatial_envelope)
        self.assertIn(match_res.refined_spatial_envelope.resolution_level, (GeographicResolutionLevel.DISTRICT, GeographicResolutionLevel.LANDMARK))

    def test_flat_terrain_safeguard_prevents_false_matches(self):
        """Verify that flat terrain (std < 0.4°) triggers the relief safeguard and rejects false positive claims."""
        profile = AdvancedSkylineExtractor.extract_skyline_from_image(self.flat_img)

        # Flat DEM horizon (all elevation angles ~0.5°)
        bearings = [float(b) for b in range(360)]
        flat_angles = [0.5 + 0.05 * math.sin(b) for b in bearings]  # std < 0.1°

        flat_dem = HorizonProfile(
            center_coord=(40.0, -100.0),
            observer_elevation_m=500.0,
            bearings_deg=bearings,
            elevation_angles_deg=flat_angles,
            peak_azimuth_deg=0.0,
            peak_elevation_deg=0.55,
            max_distance_m=20000.0,
        )

        res = AdvancedSkylineMatcher.match_skyline(profile, flat_dem)

        # Safeguard must flag as NOT statistically significant
        self.assertFalse(res.is_statistically_significant)
        self.assertEqual(res.correlation_score, 0.10)
        self.assertIn("flat terrain", res.justification.lower())


if __name__ == "__main__":
    unittest.main()
