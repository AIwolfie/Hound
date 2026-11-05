"""Unit and regression tests for HOUND Public-Landmark Benchmark Suite."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from benchmarks.benchmark_suite import PublicLandmarkBenchmark
from core.geographic_search import BoundingBox
from integrations.geocoding.gazetteer import GeocodingResult
from storage.database import DatabaseManager


class TestBenchmarkSuite(unittest.TestCase):
    """Test suite executing the deterministic public landmark benchmark."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        self.db = DatabaseManager(self.base_path / "test_benchmark.db")
        self.benchmark = PublicLandmarkBenchmark(db_manager=self.db)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_benchmark_execution(self):
        """Execute benchmark cases and verify scale, precision, and abstention."""
        # Mock gazetteer forward geocode to keep test offline and deterministic
        def mock_forward_geocode(query: str, limit: int = 3):
            if "101" in query:
                return [
                    GeocodingResult(
                        display_name="US Highway 101, California, USA",
                        lat=34.1480,
                        lon=-118.7956,
                        bbox=BoundingBox(min_lat=34.0, min_lon=-119.0, max_lat=35.0, max_lon=-118.0),
                        place_type="road",
                        country="United States",
                    )
                ]
            elif "Grand Central" in query:
                return [
                    GeocodingResult(
                        display_name="Grand Central Terminal, 89 E 42nd St, New York, NY",
                        lat=40.7527,
                        lon=-73.9772,
                        bbox=BoundingBox(min_lat=40.75, min_lon=-73.98, max_lat=40.76, max_lon=-73.97),
                        place_type="poi",
                        country="United States",
                    )
                ]
            return []

        def mock_reverse_geocode(lat: float, lon: float):
            if abs(lat - 48.8584) < 0.1:
                return GeocodingResult(
                    display_name="Eiffel Tower, Champ de Mars, Paris, France",
                    lat=48.8584,
                    lon=2.2945,
                    bbox=BoundingBox(min_lat=48.85, min_lon=2.29, max_lat=48.86, max_lon=2.30),
                    place_type="poi",
                    country="France",
                )
            elif abs(lat - 27.9881) < 0.1:
                return GeocodingResult(
                    display_name="Mount Everest, Solukhumbu, Nepal",
                    lat=27.9881,
                    lon=86.9250,
                    bbox=BoundingBox(min_lat=27.98, min_lon=86.92, max_lat=28.00, max_lon=86.93),
                    place_type="peak",
                    country="Nepal",
                )
            return None

        mock_highways = MagicMock()
        mock_highways.status = "success"
        mock_highways.ways = []
        mock_highways.endpoint_used = "mock_osm"

        mock_horizon = MagicMock()
        mock_horizon.peak_elevation_deg = 2.0
        mock_horizon.peak_azimuth_deg = 180.0
        mock_horizon.elevation_angles_deg = [0.0] * 180
        mock_horizon.bearings_deg = [float(b) for b in range(180)]
        mock_horizon.center_coord = (0.0, 0.0)
        mock_horizon.observer_elevation_m = 100.0

        with patch("integrations.geocoding.gazetteer.GazetteerService.forward_geocode", side_effect=mock_forward_geocode), \
             patch("integrations.geocoding.gazetteer.GazetteerService.reverse_geocode", side_effect=mock_reverse_geocode), \
             patch("integrations.openstreetmap.overpass_client.OverpassClient.query_highways", return_value=mock_highways), \
             patch("integrations.elevation.horizon_generator.HorizonGenerator.generate_horizon", return_value=mock_horizon), \
             patch("reports.geographic_visualizer.SatelliteTileService.build_mosaic", return_value=None):

            results = self.benchmark.run_all()
            self.assertEqual(len(results), 5)

            # BM-01: Eiffel Tower GPS anchor
            bm01 = next(r for r in results if r.case_id == "BM-01")
            self.assertEqual(bm01.top1_candidate_level, "point")
            self.assertIsNotNone(bm01.distance_to_ground_truth_km)
            self.assertLess(bm01.distance_to_ground_truth_km, 1.0)
            self.assertTrue(bm01.scale_matches)

            # BM-02: US-101 Corridor
            bm02 = next(r for r in results if r.case_id == "BM-02")
            self.assertEqual(bm02.top1_candidate_level, "area")
            self.assertIn("US Highway 101", bm02.top1_candidate_name)

            # BM-03: Grand Central Terminal
            bm03 = next(r for r in results if r.case_id == "BM-03")
            self.assertIn("Grand Central", bm03.top1_candidate_name)

            # BM-04: Mount Everest
            bm04 = next(r for r in results if r.case_id == "BM-04")
            self.assertEqual(bm04.top1_candidate_level, "point")
            self.assertLess(bm04.distance_to_ground_truth_km, 1.0)

            # BM-05: Abstention Control (Low-Info Scene)
            bm05 = next(r for r in results if r.case_id == "BM-05")
            self.assertTrue(bm05.abstention_correct)
            self.assertEqual(bm05.top1_candidate_level, "region")
            self.assertIn("Unlocalized", bm05.top1_candidate_name)
            self.assertIsNone(bm05.top1_candidate_coords)


if __name__ == "__main__":
    unittest.main()
