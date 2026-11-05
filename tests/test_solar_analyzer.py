"""Tests for HOUND Solar Ephemeris and Shadow Analysis Engine."""
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image, ImageDraw

from core.image_loader import load_and_assess_image
from investigators.shadows.solar_analyzer import SolarAnalyzer, SolarPosition


class TestSolarAnalyzer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)
        self.analyzer = SolarAnalyzer()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_solar_position_equinox_solar_noon(self):
        # March 21 (approx equinox), 12:00 UTC at 0° latitude, 0° longitude
        dt_equinox = datetime(2025, 3, 21, 12, 0, 0, tzinfo=timezone.utc)
        pos = self.analyzer.calculate_solar_position(
            lat_deg=0.0,
            lon_deg=0.0,
            dt_utc=dt_equinox,
        )

        self.assertIsInstance(pos, SolarPosition)
        # At equator on equinox at solar noon, sun should be nearly overhead (> 85°)
        self.assertGreater(pos.elevation_deg, 85.0)
        # Declination should be close to 0°
        self.assertAlmostEqual(pos.declination_deg, 0.0, delta=2.0)
        # Shadow ratio should be near 0 (minimal shadow)
        if pos.shadow_length_ratio is not None:
            self.assertLess(pos.shadow_length_ratio, 0.20)

    def test_solar_position_night(self):
        # Midnight UTC at 0° lat, 0° lon
        dt_night = datetime(2025, 6, 21, 0, 0, 0, tzinfo=timezone.utc)
        pos = self.analyzer.calculate_solar_position(0.0, 0.0, dt_night)
        # Sun elevation should be well below horizon (< 0°)
        self.assertLess(pos.elevation_deg, -50.0)
        self.assertIsNone(pos.shadow_length_ratio)

    def test_shadow_detection_on_synthetic_image(self):
        # Create an image with an elongated dark cast shadow on bright ground
        img_path = self.dir_path / "shadow_test.jpg"
        img = Image.new("RGB", (600, 400), color=(220, 220, 220))
        draw = ImageDraw.Draw(img)

        # Draw elongated diagonal dark shadow polygon
        draw.polygon([(100, 200), (350, 320), (345, 335), (95, 215)], fill=(30, 30, 35))
        img.save(img_path)

        candidates = self.analyzer.detect_candidate_shadows(img_path, min_area=300.0, min_aspect_ratio=2.0)
        self.assertGreaterEqual(len(candidates), 1)

        primary = candidates[0]
        self.assertGreater(primary.length_px, 150.0)
        self.assertGreater(primary.aspect_ratio, 3.0)
        self.assertGreater(primary.confidence, 0.50)


if __name__ == "__main__":
    unittest.main()
