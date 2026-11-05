"""HOUND Geographic Geometry & Geodesy Engine.

Provides spherical geodetic calculations, bearing math, ray intersections,
Web Mercator tile projections, and spatial bounding box operations.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field

# WGS-84 Earth radius constants
EARTH_RADIUS_METERS = 6371008.8  # Mean spherical Earth radius in meters


class BoundingBox(BaseModel):
    """Represents a geographic bounding box in decimal degrees WGS-84."""

    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float

    @property
    def south(self) -> float:
        return self.min_lat

    @property
    def west(self) -> float:
        return self.min_lon

    @property
    def north(self) -> float:
        return self.max_lat

    @property
    def east(self) -> float:
        return self.max_lon

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.min_lat + self.max_lat) / 2.0, (self.min_lon + self.max_lon) / 2.0)

    @property
    def span_lat(self) -> float:
        return abs(self.max_lat - self.min_lat)

    @property
    def span_lon(self) -> float:
        return abs(self.max_lon - self.min_lon)

    def contains(self, lat: float, lon: float) -> bool:
        """Check if a coordinate lies within the bounding box."""
        return self.min_lat <= lat <= self.max_lat and self.min_lon <= lon <= self.max_lon

    def intersects(self, other: BoundingBox) -> bool:
        """Check if this bounding box overlaps with another."""
        return not (
            self.max_lat < other.min_lat
            or self.min_lat > other.max_lat
            or self.max_lon < other.min_lon
            or self.min_lon > other.max_lon
        )

    def expand(self, distance_m: float) -> BoundingBox:
        """Expand bounding box by distance in meters in all directions."""
        delta_lat = (distance_m / EARTH_RADIUS_METERS) * (180.0 / math.pi)
        mid_lat = (self.min_lat + self.max_lat) / 2.0
        cos_lat = max(0.01, math.cos(math.radians(mid_lat)))
        delta_lon = (distance_m / (EARTH_RADIUS_METERS * cos_lat)) * (180.0 / math.pi)

        return BoundingBox(
            min_lat=max(-90.0, self.min_lat - delta_lat),
            min_lon=max(-180.0, self.min_lon - delta_lon),
            max_lat=min(90.0, self.max_lat + delta_lat),
            max_lon=min(180.0, self.max_lon + delta_lon),
        )

    def to_overpass_str(self) -> str:
        """Format as Overpass QL bbox: south,west,north,east."""
        return f"{self.min_lat:.6f},{self.min_lon:.6f},{self.max_lat:.6f},{self.max_lon:.6f}"

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.min_lat, self.min_lon, self.max_lat, self.max_lon)

    @classmethod
    def from_point_and_radius(cls, lat: float, lon: float, radius_m: float) -> BoundingBox:
        """Create bounding box around a center coordinate with radius in meters."""
        delta_lat = (radius_m / EARTH_RADIUS_METERS) * (180.0 / math.pi)
        cos_lat = max(0.01, math.cos(math.radians(lat)))
        delta_lon = (radius_m / (EARTH_RADIUS_METERS * cos_lat)) * (180.0 / math.pi)

        return cls(
            min_lat=max(-90.0, lat - delta_lat),
            min_lon=max(-180.0, lon - delta_lon),
            max_lat=min(90.0, lat + delta_lat),
            max_lon=min(180.0, lon + delta_lon),
        )


def haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """Calculate great-circle distance in meters between two (lat, lon) coordinates."""
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return EARTH_RADIUS_METERS * c


def forward_azimuth(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """Calculate forward initial bearing (degrees from True North: 0° to 360°) from coord1 to coord2."""
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    delta_lon = lon2 - lon1

    y = math.sin(delta_lon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)

    bearing_rad = math.atan2(y, x)
    bearing_deg = (math.degrees(bearing_rad) + 360.0) % 360.0
    return round(bearing_deg, 3)


def destination_point(
    start_coord: Tuple[float, float], distance_m: float, bearing_deg: float
) -> Tuple[float, float]:
    """Calculate destination coordinate given starting point, distance in meters, and bearing in degrees."""
    lat1 = math.radians(start_coord[0])
    lon1 = math.radians(start_coord[1])
    bearing_rad = math.radians(bearing_deg)
    angular_dist = distance_m / EARTH_RADIUS_METERS

    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular_dist)
        + math.cos(lat1) * math.sin(angular_dist) * math.cos(bearing_rad)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing_rad) * math.sin(angular_dist) * math.cos(lat1),
        math.cos(angular_dist) - math.sin(lat1) * math.sin(lat2),
    )

    return (round(math.degrees(lat2), 7), round((math.degrees(lon2) + 540.0) % 360.0 - 180.0, 7))


def ray_intersection(
    p1: Tuple[float, float],
    bearing1_deg: float,
    p2: Tuple[float, float],
    bearing2_deg: float,
) -> Optional[Tuple[float, float]]:
    """Calculate intersection coordinate of two great-circle bearing rays.

    Returns:
        Intersection (lat, lon) in decimal degrees, or None if rays are parallel or non-intersecting.
    """
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    theta1 = math.radians(bearing1_deg)
    theta2 = math.radians(bearing2_deg)

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    # Angular distance between p1 and p2
    delta12 = 2.0 * math.asin(
        math.sqrt(
            math.sin(delta_lat / 2.0) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2.0) ** 2
        )
    )

    if delta12 < 1e-9:
        return p1

    # Initial/final bearings between p1 and p2
    cos_theta_a = (math.sin(lat2) - math.sin(lat1) * math.cos(delta12)) / (
        math.sin(delta12) * math.cos(lat1) + 1e-12
    )
    cos_theta_b = (math.sin(lat1) - math.sin(lat2) * math.cos(delta12)) / (
        math.sin(delta12) * math.cos(lat2) + 1e-12
    )

    theta_a = math.acos(max(-1.0, min(1.0, cos_theta_a)))
    theta_b = math.acos(max(-1.0, min(1.0, cos_theta_b)))

    if math.sin(delta_lon) > 0:
        theta12 = theta_a
        theta21 = 2.0 * math.pi - theta_b
    else:
        theta12 = 2.0 * math.pi - theta_a
        theta21 = theta_b

    alpha1 = (theta1 - theta12 + math.pi) % (2.0 * math.pi) - math.pi
    alpha2 = (theta21 - theta2 + math.pi) % (2.0 * math.pi) - math.pi

    if math.sin(alpha1) == 0.0 and math.sin(alpha2) == 0.0:
        return None  # Infinite intersections (collinear)
    if math.sin(alpha1) * math.sin(alpha2) < 0.0:
        return None  # Rays diverge or intersect behind camera

    cos_alpha3 = -math.cos(alpha1) * math.cos(alpha2) + math.sin(alpha1) * math.sin(alpha2) * math.cos(delta12)
    delta13 = math.atan2(
        math.sin(delta12) * math.sin(alpha1) * math.sin(alpha2),
        math.cos(alpha2) + math.cos(alpha1) * cos_alpha3,
    )

    lat3 = math.asin(
        math.sin(lat1) * math.cos(delta13) + math.cos(lat1) * math.sin(delta13) * math.cos(theta1)
    )
    delta_lon13 = math.atan2(
        math.sin(theta1) * math.sin(delta13) * math.cos(lat1),
        math.cos(delta13) - math.sin(lat1) * math.sin(lat3),
    )
    lon3 = lon1 + delta_lon13

    return (round(math.degrees(lat3), 7), round((math.degrees(lon3) + 540.0) % 360.0 - 180.0, 7))


def lat_lon_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int, int, int]:
    """Project (lat, lon) to Web Mercator Slippy Map tile numbers and in-tile pixel coordinates.

    Returns:
        (tile_x, tile_y, pixel_x, pixel_y) where pixel is [0, 255] inside a 256x256 tile.
    """
    lat_clamped = max(-85.05112878, min(85.05112878, lat))
    lat_rad = math.radians(lat_clamped)
    n = 2.0**zoom

    x_float = ((lon + 180.0) / 360.0) * n
    y_float = (1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * n

    tile_x = int(math.floor(x_float))
    tile_y = int(math.floor(y_float))

    px_x = int(math.floor((x_float - tile_x) * 256.0))
    px_y = int(math.floor((y_float - tile_y) * 256.0))

    return (tile_x, tile_y, px_x, px_y)


def lat_lon_to_subpixel_tile(
    lat: float, lon: float, zoom: int
) -> Tuple[int, int, int, int, float, float]:
    """Project (lat, lon) to Web Mercator tile, integer pixel (x0, y0), and sub-pixel fractions (fx, fy).

    Returns:
        (tile_x, tile_y, x0, y0, fx, fy) where x0, y0 in [0, 255] and fx, fy in [0.0, 1.0).
    """
    lat_clamped = max(-85.05112878, min(85.05112878, lat))
    lat_rad = math.radians(lat_clamped)
    n = 2.0**zoom

    x_float = ((lon + 180.0) / 360.0) * n
    y_float = (1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * n

    tile_x = int(math.floor(x_float))
    tile_y = int(math.floor(y_float))

    raw_px_x = (x_float - tile_x) * 256.0
    raw_px_y = (y_float - tile_y) * 256.0

    x0 = min(255, max(0, int(math.floor(raw_px_x))))
    y0 = min(255, max(0, int(math.floor(raw_px_y))))

    fx = max(0.0, min(1.0, raw_px_x - x0))
    fy = max(0.0, min(1.0, raw_px_y - y0))

    return (tile_x, tile_y, x0, y0, fx, fy)


def tile_to_bounding_box(tile_x: int, tile_y: int, zoom: int) -> BoundingBox:
    """Calculate geographic bounding box for a given Web Mercator tile."""
    n = 2.0**zoom
    lon_min = (tile_x / n) * 360.0 - 180.0
    lon_max = ((tile_x + 1) / n) * 360.0 - 180.0

    lat_max_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * (tile_y / n))))
    lat_min_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * ((tile_y + 1) / n))))

    return BoundingBox(
        min_lat=math.degrees(lat_min_rad),
        min_lon=lon_min,
        max_lat=math.degrees(lat_max_rad),
        max_lon=lon_max,
    )


def tile_to_lat_lon(tile_x: int, tile_y: int, zoom: int) -> Tuple[float, float]:
    """Return top-left (north-west) coordinate of a Slippy Map tile."""
    n = 2.0**zoom
    lon = (tile_x / n) * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * (tile_y / n))))
    return (math.degrees(lat_rad), lon)
