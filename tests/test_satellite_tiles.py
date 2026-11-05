"""Unit tests for HOUND Satellite & Map Tile Service."""
from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image

from core.geographic_search import BoundingBox
from integrations.satellite.tile_service import SatelliteTileService


class TestSatelliteTileService(unittest.TestCase):
    """Test suite for SatelliteTileService caching, projection, and mosaic stitching."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.temp_dir.name)
        self.service = SatelliteTileService(cache_dir=self.cache_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_geo_to_pixel_projection(self):
        """Verify geodetic coordinate to image pixel projection."""
        bbox = BoundingBox(min_lat=10.0, min_lon=20.0, max_lat=11.0, max_lon=21.0)
        w, h = 500, 500

        # Top-left corner (max_lat, min_lon) should map to (0, 0)
        px, py = SatelliteTileService.geo_to_pixel(11.0, 20.0, bbox, w, h)
        self.assertEqual(px, 0)
        self.assertEqual(py, 0)

        # Bottom-right corner (min_lat, max_lon) should map to (w-1, h-1)
        px, py = SatelliteTileService.geo_to_pixel(10.0, 21.0, bbox, w, h)
        self.assertEqual(px, 499)
        self.assertEqual(py, 499)

        # Center point
        px, py = SatelliteTileService.geo_to_pixel(10.5, 20.5, bbox, w, h)
        self.assertAlmostEqual(px, 250, delta=2)
        self.assertAlmostEqual(py, 250, delta=2)

    def test_fetch_tile_caching(self):
        """Verify cached tiles are read from disk without network calls."""
        # Create a pre-cached tile
        tile_path = self.cache_dir / "osm" / "10" / "500" / "300.png"
        tile_path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (256, 256), color=(100, 150, 200))
        img.save(tile_path, format="PNG")

        # fetch_tile should load from disk
        loaded = self.service.fetch_tile(500, 300, 10, provider="osm")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.size, (256, 256))

    def test_mosaic_budget_limit(self):
        """Verify build_mosaic aborts if requested area exceeds max_tiles."""
        # Global bounding box at zoom 15 would require millions of tiles
        world_bbox = BoundingBox(min_lat=-50.0, min_lon=-100.0, max_lat=50.0, max_lon=100.0)
        result = self.service.build_mosaic(world_bbox, zoom=15, max_tiles=10)
        self.assertIsNone(result)

    def test_build_mosaic_assembly(self):
        """Verify mosaic assembly stitches multiple tiles correctly."""
        dummy_tile = Image.new("RGB", (256, 256), color=(50, 60, 70))

        # Local bbox in San Francisco spanning 2x2 tiles at zoom 14
        sf_bbox = BoundingBox(min_lat=37.75, min_lon=-122.45, max_lat=37.78, max_lon=-122.41)

        with patch.object(self.service, "fetch_tile", return_value=dummy_tile):
            res = self.service.build_mosaic(sf_bbox, zoom=14, max_tiles=25)
            self.assertIsNotNone(res)
            mosaic_img, cov_bbox = res

            self.assertIsInstance(mosaic_img, Image.Image)
            self.assertGreaterEqual(mosaic_img.width, 256)
            self.assertGreaterEqual(mosaic_img.height, 256)
            self.assertLessEqual(cov_bbox.min_lat, sf_bbox.min_lat)
            self.assertGreaterEqual(cov_bbox.max_lat, sf_bbox.max_lat)


if __name__ == "__main__":
    unittest.main()
