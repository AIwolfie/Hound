"""Unit tests for HOUND Geographic Visualizer."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from PIL import Image

from agent.candidate_generator import GeographicCandidate
from integrations.elevation.horizon_generator import HorizonProfile
from reports.geographic_visualizer import GeographicVisualizer


class TestGeographicVisualizer(unittest.TestCase):
    """Test suite for GeographicVisualizer map and horizon rendering."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_render_candidate_map_fallback(self):
        """Verify map rendering with empty candidates creates fallback map."""
        out_path = self.output_dir / "test_empty_map.jpg"
        result_path = GeographicVisualizer.render_candidate_map([], out_path)

        self.assertTrue(result_path.is_file())
        with Image.open(result_path) as img:
            self.assertEqual(img.format, "JPEG")
            self.assertEqual(img.size, (640, 480))

    def test_render_candidate_map_with_candidates(self):
        """Verify map rendering with candidates creates annotated image."""
        out_path = self.output_dir / "test_cands_map.jpg"
        c1 = GeographicCandidate(
            id="C1",
            investigation_id="inv1",
            level="point",
            name="Alpha Location",
            lat=37.7749,
            lon=-122.4194,
            radius_m=500.0,
            status="active",
        )
        c2 = GeographicCandidate(
            id="C2",
            investigation_id="inv1",
            level="area",
            name="Beta Location",
            lat=37.8000,
            lon=-122.4000,
            status="excluded",
        )

        result_path = GeographicVisualizer.render_candidate_map([c1, c2], out_path)
        self.assertTrue(result_path.is_file())
        with Image.open(result_path) as img:
            self.assertEqual(img.format, "JPEG")
            self.assertGreater(img.width, 200)
            self.assertGreater(img.height, 200)

    def test_render_horizon_profile_plot(self):
        """Verify 360-degree horizon profile plot rendering."""
        out_path = self.output_dir / "test_horizon.jpg"
        bearings = [float(b) for b in range(0, 360, 2)]
        import math

        elevations = [2.0 + 3.0 * math.sin(math.radians(b * 2)) for b in bearings]

        profile = HorizonProfile(
            center_coord=(45.0, 7.0),
            observer_elevation_m=1200.0,
            bearings_deg=bearings,
            elevation_angles_deg=elevations,
            peak_azimuth_deg=90.0,
            peak_elevation_deg=5.0,
            max_distance_m=10000.0,
        )

        observed_ridge = [(0.0, 4.8), (2.0, 4.9), (4.0, 5.0), (6.0, 4.8)]
        result_path = GeographicVisualizer.render_horizon_profile_plot(
            profile, out_path, observed_ridge=observed_ridge
        )

        self.assertTrue(result_path.is_file())
        with Image.open(result_path) as img:
            self.assertEqual(img.format, "JPEG")
            self.assertEqual(img.size, (900, 350))


if __name__ == "__main__":
    unittest.main()
