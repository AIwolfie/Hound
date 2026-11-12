"""HOUND Spatial Uncertainty Envelopes & Geographic Resolution Hierarchy.

Defines formal geographic resolution levels, bounded uncertainty envelopes,
and prevents false precision claims when evidence only justifies broad corridors or regions.
"""
from __future__ import annotations

import enum
import math
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from core.geographic_search import BoundingBox, EARTH_RADIUS_METERS, haversine_distance


class GeographicResolutionLevel(str, enum.Enum):
    """Hierarchical scale of geographic localization precision."""
    COUNTRY = "country"                     # ~500 - 2500 km
    STATE_OR_PROVINCE = "state_or_province" # ~100 - 500 km
    REGION = "region"                       # ~50 - 200 km
    CITY = "city"                           # ~10 - 40 km
    DISTRICT = "district"                   # ~3 - 10 km
    ROAD_CORRIDOR = "road_corridor"         # Linear buffer ~1 - 50 km along road axis
    NEIGHBORHOOD = "neighborhood"           # ~1 - 3 km
    LANDMARK = "landmark"                   # ~100 - 500 m
    SPECIFIC_COORDINATE = "specific_coordinate" # <= 100 m
    UNLOCALIZED = "unlocalized"             # Indeterminate / global envelope


# Maximum permissible uncertainty radius (meters) for claiming each resolution level
MAX_UNCERTAINTY_RADIUS_M: Dict[GeographicResolutionLevel, float] = {
    GeographicResolutionLevel.SPECIFIC_COORDINATE: 100.0,
    GeographicResolutionLevel.LANDMARK: 500.0,
    GeographicResolutionLevel.NEIGHBORHOOD: 3000.0,
    GeographicResolutionLevel.ROAD_CORRIDOR: 50000.0,
    GeographicResolutionLevel.DISTRICT: 10000.0,
    GeographicResolutionLevel.CITY: 40000.0,
    GeographicResolutionLevel.REGION: 200000.0,
    GeographicResolutionLevel.STATE_OR_PROVINCE: 500000.0,
    GeographicResolutionLevel.COUNTRY: 2500000.0,
    GeographicResolutionLevel.UNLOCALIZED: 20000000.0,
}


class SpatialEnvelope(BaseModel):
    """Represents a bounded geographic region with explicit uncertainty dimensions."""

    resolution_level: GeographicResolutionLevel
    center_lat: Optional[float] = None
    center_lon: Optional[float] = None
    uncertainty_radius_m: float = 1000000.0
    bbox: Optional[BoundingBox] = None
    corridor_waypoints: List[Tuple[float, float]] = Field(default_factory=list)
    area_name: str = "Unspecified Area"
    supporting_modalities: List[str] = Field(default_factory=list)
    justification: str = "Initial unlocalized envelope."

    @classmethod
    def from_point_and_radius(
        cls,
        lat: float,
        lon: float,
        radius_m: float,
        level: Optional[GeographicResolutionLevel] = None,
        area_name: str = "",
        modalities: Optional[List[str]] = None,
        justification: str = "",
    ) -> SpatialEnvelope:
        """Construct a calibrated spatial envelope around a point."""
        # Derive appropriate resolution level if not explicitly provided
        if level is None:
            if radius_m <= 100.0:
                level = GeographicResolutionLevel.SPECIFIC_COORDINATE
            elif radius_m <= 500.0:
                level = GeographicResolutionLevel.LANDMARK
            elif radius_m <= 3000.0:
                level = GeographicResolutionLevel.NEIGHBORHOOD
            elif radius_m <= 10000.0:
                level = GeographicResolutionLevel.DISTRICT
            elif radius_m <= 40000.0:
                level = GeographicResolutionLevel.CITY
            elif radius_m <= 200000.0:
                level = GeographicResolutionLevel.REGION
            elif radius_m <= 500000.0:
                level = GeographicResolutionLevel.STATE_OR_PROVINCE
            elif radius_m <= 2500000.0:
                level = GeographicResolutionLevel.COUNTRY
            else:
                level = GeographicResolutionLevel.UNLOCALIZED

        bbox = BoundingBox.from_point_and_radius(lat, lon, radius_m)
        return cls(
            resolution_level=level,
            center_lat=lat,
            center_lon=lon,
            uncertainty_radius_m=radius_m,
            bbox=bbox,
            area_name=area_name or f"{level.value.replace('_', ' ').title()} Envelope",
            supporting_modalities=modalities or [],
            justification=justification or f"Localized to {level.value} with +/- {radius_m/1000.0:.1f} km uncertainty.",
        )

    @classmethod
    def point(
        cls,
        lat: float,
        lon: float,
        radius_m: float,
        resolution: Optional[GeographicResolutionLevel] = None,
        reasoning: str = "",
        area_name: str = "",
        modalities: Optional[List[str]] = None,
    ) -> SpatialEnvelope:
        """Convenience factory for point-based spatial envelopes."""
        return cls.from_point_and_radius(
            lat=lat,
            lon=lon,
            radius_m=radius_m,
            level=resolution,
            area_name=area_name,
            modalities=modalities,
            justification=reasoning,
        )

    @classmethod
    def from_road_corridor(
        cls,
        waypoints: List[Tuple[float, float]],
        buffer_width_m: float = 5000.0,
        road_name: str = "",
        modalities: Optional[List[str]] = None,
        justification: str = "",
    ) -> SpatialEnvelope:
        """Construct a corridor envelope along a linear roadway or railway axis."""
        if not waypoints:
            return cls.unlocalized("Empty corridor waypoints.")

        lats = [pt[0] for pt in waypoints]
        lons = [pt[1] for pt in waypoints]
        mid_idx = len(waypoints) // 2
        center_lat, center_lon = waypoints[mid_idx]

        # Calculate bounding box with buffer
        min_lat = min(lats)
        max_lat = max(lats)
        min_lon = min(lons)
        max_lon = max(lons)

        delta_lat = (buffer_width_m / EARTH_RADIUS_METERS) * (180.0 / math.pi)
        mid_lat = (min_lat + max_lat) / 2.0
        cos_lat = max(0.01, math.cos(math.radians(mid_lat)))
        delta_lon = (buffer_width_m / (EARTH_RADIUS_METERS * cos_lat)) * (180.0 / math.pi)

        bbox = BoundingBox(
            min_lat=max(-90.0, min_lat - delta_lat),
            min_lon=max(-180.0, min_lon - delta_lon),
            max_lat=min(90.0, max_lat + delta_lat),
            max_lon=min(180.0, max_lon + delta_lon),
        )

        # Length of corridor
        total_len_m = 0.0
        for i in range(len(waypoints) - 1):
            total_len_m += haversine_distance(waypoints[i], waypoints[i + 1])

        uncertainty_radius_m = max(buffer_width_m, total_len_m / 2.0)

        return cls(
            resolution_level=GeographicResolutionLevel.ROAD_CORRIDOR,
            center_lat=center_lat,
            center_lon=center_lon,
            uncertainty_radius_m=uncertainty_radius_m,
            bbox=bbox,
            corridor_waypoints=waypoints,
            area_name=road_name or "Road Corridor",
            supporting_modalities=modalities or ["road_geometry"],
            justification=justification or f"Linear road corridor of length {total_len_m/1000.0:.1f} km with +/- {buffer_width_m/1000.0:.1f} km lateral buffer.",
        )

    @classmethod
    def unlocalized(cls, reason: str = "Insufficient visual evidence.") -> SpatialEnvelope:
        """Default unlocalized global envelope when evidence is absent or inconclusive."""
        return cls(
            resolution_level=GeographicResolutionLevel.UNLOCALIZED,
            center_lat=None,
            center_lon=None,
            uncertainty_radius_m=20000000.0,
            bbox=None,
            area_name="Unlocalized Global",
            supporting_modalities=[],
            justification=f"Abstaining from localization: {reason}",
        )

    def contains(self, lat: float, lon: float) -> bool:
        """Evaluate whether a coordinate falls within this spatial envelope."""
        if self.center_lat is None or self.center_lon is None:
            return False

        if self.corridor_waypoints:
            # Check distance to closest corridor segment
            min_dist = float("inf")
            target = (lat, lon)
            for i in range(len(self.corridor_waypoints) - 1):
                p1 = self.corridor_waypoints[i]
                p2 = self.corridor_waypoints[i + 1]
                # Distance to segment vertices as robust approximation
                d = min(haversine_distance(target, p1), haversine_distance(target, p2))
                if d < min_dist:
                    min_dist = d
            return min_dist <= self.uncertainty_radius_m

        dist = haversine_distance((self.center_lat, self.center_lon), (lat, lon))
        return dist <= self.uncertainty_radius_m

    @property
    def formatted_display(self) -> str:
        """Human-readable display string reflecting precision level and bounds."""
        if self.resolution_level == GeographicResolutionLevel.UNLOCALIZED:
            return "Unlocalized (Global Scope)"

        level_str = self.resolution_level.value.replace("_", " ").title()
        if self.center_lat is not None and self.center_lon is not None:
            if self.resolution_level in (GeographicResolutionLevel.SPECIFIC_COORDINATE, GeographicResolutionLevel.LANDMARK):
                return f"{self.area_name} [{level_str}] ({self.center_lat:.5f}, {self.center_lon:.5f}) +/- {self.uncertainty_radius_m:.0f}m"
            else:
                radius_km = self.uncertainty_radius_m / 1000.0
                return f"{self.area_name} [{level_str}] Center: ({self.center_lat:.3f}, {self.center_lon:.3f}) +/- {radius_km:.1f} km"

        return f"{self.area_name} [{level_str}]"
