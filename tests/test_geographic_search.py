"""Tests for HOUND Geographic Geometry & Geodesy Engine."""
import math
import unittest

from core.geographic_search import (
    BoundingBox,
    destination_point,
    forward_azimuth,
    haversine_distance,
    lat_lon_to_tile,
    ray_intersection,
    tile_to_bounding_box,
    tile_to_lat_lon,
)


class TestGeographicSearch(unittest.TestCase):
    def test_haversine_distance(self):
        # Paris (48.8566, 2.3522) to London (51.5074, -0.1278) ~ 343 km
        paris = (48.8566, 2.3522)
        london = (51.5074, -0.1278)
        dist = haversine_distance(paris, london)
        self.assertAlmostEqual(dist, 343500.0, delta=2000.0)

    def test_forward_azimuth(self):
        # Due North
        p1 = (10.0, 20.0)
        p2 = (20.0, 20.0)
        az_north = forward_azimuth(p1, p2)
        self.assertAlmostEqual(az_north, 0.0, delta=0.5)

        # Due East along Equator
        p_eq1 = (0.0, 0.0)
        p_eq2 = (0.0, 10.0)
        az_east = forward_azimuth(p_eq1, p_eq2)
        self.assertAlmostEqual(az_east, 90.0, delta=0.5)

    def test_destination_point(self):
        start = (40.0, -74.0)
        dist_m = 10000.0  # 10 km
        bearing = 90.0  # East
        dest = destination_point(start, dist_m, bearing)
        self.assertAlmostEqual(dest[0], 40.0, delta=0.01)
        self.assertGreater(dest[1], -74.0)
        # Verify distance back
        calc_dist = haversine_distance(start, dest)
        self.assertAlmostEqual(calc_dist, dist_m, delta=50.0)

    def test_ray_intersection(self):
        # Point A at (0, 0) looking North (0°)
        # Point B at (1, 1) looking West (270°)
        # Target intersection should be at (1, 0)
        p1 = (0.0, 0.0)
        p2 = (1.0, 1.0)
        inter = ray_intersection(p1, 0.0, p2, 270.0)
        self.assertIsNotNone(inter)
        self.assertAlmostEqual(inter[0], 1.0, delta=0.05)
        self.assertAlmostEqual(inter[1], 0.0, delta=0.05)

    def test_bounding_box_operations(self):
        bbox = BoundingBox(min_lat=10.0, min_lon=20.0, max_lat=15.0, max_lon=25.0)
        self.assertTrue(bbox.contains(12.0, 22.0))
        self.assertFalse(bbox.contains(5.0, 22.0))
        self.assertEqual(bbox.center, (12.5, 22.5))

        # Test expand
        expanded = bbox.expand(50000.0)  # 50 km
        self.assertLess(expanded.min_lat, bbox.min_lat)
        self.assertGreater(expanded.max_lat, bbox.max_lat)

        # Test from point and radius
        pt_box = BoundingBox.from_point_and_radius(45.0, 10.0, 1000.0)
        self.assertTrue(pt_box.contains(45.0, 10.0))

    def test_web_mercator_tile_conversions(self):
        lat, lon = 48.8584, 2.2945  # Eiffel Tower, Paris
        zoom = 15
        tx, ty, px, py = lat_lon_to_tile(lat, lon, zoom)
        self.assertGreater(tx, 0)
        self.assertGreater(ty, 0)
        self.assertTrue(0 <= px <= 255)
        self.assertTrue(0 <= py <= 255)

        # Reverse check bounding box
        bbox = tile_to_bounding_box(tx, ty, zoom)
        self.assertTrue(bbox.contains(lat, lon))


if __name__ == "__main__":
    unittest.main()
