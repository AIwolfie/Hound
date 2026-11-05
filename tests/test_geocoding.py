"""Tests for HOUND Geocoding & Gazetteer Engine."""
import json
import tempfile
import unittest
from pathlib import Path

from integrations.geocoding.gazetteer import GazetteerService, GeocodingResult
from storage.database import DatabaseManager


class TestGeocoding(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)
        self.db_path = self.dir_path / "test_geo.db"
        self.db_manager = DatabaseManager(self.db_path)
        self.gazetteer = GazetteerService(db_manager=self.db_manager)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parse_nominatim_item(self):
        raw_item = {
            "lat": "49.2827",
            "lon": "-123.1207",
            "display_name": "Vancouver, Greater Vancouver, British Columbia, Canada",
            "type": "city",
            "importance": 0.85,
            "osm_id": 1852574,
            "boundingbox": ["49.1989", "49.3161", "-123.2249", "-123.0232"],
            "address": {
                "city": "Vancouver",
                "state": "British Columbia",
                "country": "Canada",
            },
        }

        res = self.gazetteer._parse_item(raw_item)
        self.assertIsInstance(res, GeocodingResult)
        self.assertAlmostEqual(res.lat, 49.2827, places=3)
        self.assertAlmostEqual(res.lon, -123.1207, places=3)
        self.assertEqual(res.city, "Vancouver")
        self.assertEqual(res.state, "British Columbia")
        self.assertEqual(res.country, "Canada")
        self.assertTrue(res.bbox.contains(49.2827, -123.1207))

    def test_gazetteer_sqlite_caching(self):
        cache_key = "fwd:test_city::5"
        fake_data = [{"lat": "35.6895", "lon": "139.6917", "display_name": "Tokyo, Japan", "type": "city", "boundingbox": ["35.5", "35.8", "139.5", "139.8"]}]

        self.gazetteer._save_cached(cache_key, fake_data)
        cached = self.gazetteer._get_cached(cache_key)

        self.assertIsNotNone(cached)
        self.assertEqual(cached[0]["display_name"], "Tokyo, Japan")


if __name__ == "__main__":
    unittest.main()
