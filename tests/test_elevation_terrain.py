"""Tests for HOUND Elevation, Terrarium DEM, and Ridge Matching Engine."""
import tempfile
import unittest
from pathlib import Path
import numpy as np

from integrations.elevation.horizon_generator import HorizonGenerator, HorizonProfile
from integrations.elevation.ridge_matcher import RidgeMatcher, RidgeMatchResult
from integrations.elevation.terrarium_client import TerrariumClient, decode_terrarium_elevation


class MockTerrariumClient(TerrariumClient):
    """Mock DEM client returning deterministic elevation surfaces for testing."""

    def __init__(self):
        super().__init__()

    def get_elevation(self, lat: float, lon: float, zoom: int = 12) -> float:
        # Simulated mountain peak situated to the North (higher latitude)
        dist_from_peak = np.sqrt((lat - 46.5)**2 + (lon - 10.0)**2)
        simulated_elev = max(200.0, 3000.0 - (dist_from_peak * 8000.0))
        return float(simulated_elev)


class TestElevationTerrain(unittest.TestCase):
    def test_terrarium_decoding(self):
        # Sea level (0m) corresponds to (128, 0, 0): 128 * 256 - 32768 = 0
        elev_0 = decode_terrarium_elevation(128, 0, 0)
        self.assertEqual(elev_0, 0.0)

        # 1000m elevation: (131, 232, 0): 131 * 256 + 232 - 32768 = 1000.0
        elev_1000 = decode_terrarium_elevation(131, 232, 0)
        self.assertEqual(elev_1000, 1000.0)

        # Below sea level (-50m): 32718 -> R=127, G=206, B=0
        elev_minus50 = decode_terrarium_elevation(127, 206, 0)
        self.assertEqual(elev_minus50, -50.0)

    def test_synthetic_horizon_generation(self):
        mock_dem = MockTerrariumClient()
        gen = HorizonGenerator(terrarium_client=mock_dem)

        # Place observer at (46.0, 10.0) with mountain peak to the North at (46.5, 10.0)
        profile = gen.generate_horizon(
            lat=46.0,
            lon=10.0,
            observer_height_m=2.0,
            max_distance_m=60000.0,
            radial_step_m=1000.0,
            bearing_step_deg=5.0,
        )

        self.assertIsInstance(profile, HorizonProfile)
        self.assertEqual(len(profile.bearings_deg), 72)  # 360 / 5
        self.assertEqual(profile.center_coord, (46.0, 10.0))

        # Peak should be situated towards Due North (0° or 360°)
        self.assertTrue(profile.peak_azimuth_deg <= 15.0 or profile.peak_azimuth_deg >= 345.0)
        self.assertGreater(profile.peak_elevation_deg, 1.0)

    def test_ridge_matcher_cross_correlation(self):
        # Create a synthetic 360° horizon profile with a distinct peak at 90° (East)
        bearings = [float(b) for b in range(360)]
        # Gaussian peak at 90° with height 6.0°
        angles = [float(6.0 * np.exp(-((b - 90.0) ** 2) / 40.0) - 0.5) for b in bearings]

        profile = HorizonProfile(
            center_coord=(45.0, 8.0),
            observer_elevation_m=500.0,
            bearings_deg=bearings,
            elevation_angles_deg=angles,
            peak_azimuth_deg=90.0,
            peak_elevation_deg=6.0,
            max_distance_m=20000.0,
        )

        # Create observed ridge in a photo centered around 0° (relative angle in frame)
        # Peak is at relative angle 0.0° with height 6.0°
        rel_offsets = np.linspace(-10.0, 10.0, 21)
        observed_ridge = [(float(dx), float(6.0 * np.exp(-(dx**2) / 40.0) - 0.5)) for dx in rel_offsets]

        # Match ridge: camera was facing 90° East!
        result = RidgeMatcher.match_ridge(
            observed_ridge=observed_ridge,
            horizon_profile=profile,
            heading_search_min_deg=80.0,
            heading_search_max_deg=100.0,
            heading_step_deg=1.0,
        )

        self.assertIsInstance(result, RidgeMatchResult)
        self.assertAlmostEqual(result.best_heading_deg, 90.0, delta=1.0)
        self.assertLess(result.rmse_deg, 0.20)
        self.assertGreater(result.correlation_score, 0.80)


if __name__ == "__main__":
    unittest.main()
