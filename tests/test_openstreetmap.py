"""Tests for HOUND OpenStreetMap Engine and Geometry Parser."""
import json
import tempfile
import unittest
from pathlib import Path

from core.geographic_search import BoundingBox
from integrations.openstreetmap.geometry_parser import parse_overpass_json
from integrations.openstreetmap.overpass_client import OverpassClient
from storage.database import DatabaseManager


class TestOpenStreetMap(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)
        self.db_path = self.dir_path / "test_osm.db"
        self.db_manager = DatabaseManager(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_overpass_geometry_parser(self):
        sample_json = {
            "elements": [
                {"type": "node", "id": 1, "lat": 48.8584, "lon": 2.2945, "tags": {"railway": "station", "name": "Eiffel Station"}},
                {"type": "node", "id": 2, "lat": 48.8590, "lon": 2.2950},
                {
                    "type": "way",
                    "id": 101,
                    "nodes": [1, 2],
                    "tags": {"highway": "primary", "name": "Pont d'Iéna"},
                    "geometry": [{"lat": 48.8584, "lon": 2.2945}, {"lat": 48.8590, "lon": 2.2950}],
                },
                {"type": "relation", "id": 201, "members": [{"type": "way", "ref": 101, "role": "outer"}], "tags": {"type": "multipolygon"}},
            ]
        }

        features = parse_overpass_json(sample_json)
        self.assertEqual(len(features.nodes), 1)
        self.assertEqual(features.nodes[0].tags["name"], "Eiffel Station")

        self.assertEqual(len(features.ways), 1)
        way = features.ways[0]
        self.assertEqual(way.id, 101)
        self.assertEqual(way.tags["name"], "Pont d'Iéna")
        self.assertGreater(way.length_meters, 50.0)
        self.assertIsNotNone(way.primary_bearing_deg)
        self.assertIsNotNone(way.bounding_box)

        self.assertEqual(len(features.relations), 1)

    def test_overpass_client_caching(self):
        client = OverpassClient(db_manager=self.db_manager)
        query = "[out:json]; node(1); out;"
        query_hash = "fake_query_hash_001"
        fake_response = {"elements": [{"type": "node", "id": 999, "lat": 10.0, "lon": 20.0, "tags": {"name": "Test"}}]}

        # Save to SQLite cache
        client._save_to_cache(query_hash, query, fake_response)

        # Retrieve back
        cached = client._get_cached_response(query_hash)
        self.assertIsNotNone(cached)
        self.assertEqual(cached["elements"][0]["id"], 999)


if __name__ == "__main__":
    unittest.main()
