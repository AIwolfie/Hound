"""HOUND 360-Degree Synthetic Terrain Horizon Generator.

Rays outward from a candidate viewpoint across digital elevation models, computing
apparent vertical elevation angles incorporating Earth curvature and atmospheric refraction.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field

from core.geographic_search import EARTH_RADIUS_METERS, destination_point
from core.logger import get_logger
from integrations.elevation.terrarium_client import TerrariumClient

logger = get_logger("hound.elevation.horizon")

# Standard atmospheric refraction coefficient (approx 0.13)
ATMOSPHERIC_REFRACTION_K = 0.13


class HorizonProfile(BaseModel):
    center_coord: Tuple[float, float]
    observer_elevation_m: float
    bearings_deg: List[float]
    elevation_angles_deg: List[float]
    peak_azimuth_deg: float
    peak_elevation_deg: float
    max_distance_m: float
    data_source: str = "AWS Terrarium DEM"


class HorizonGenerator:
    """Generates 360-degree apparent terrain horizon profiles from digital elevation models."""

    def __init__(self, terrarium_client: Optional[TerrariumClient] = None):
        self.dem = terrarium_client or TerrariumClient()

    def generate_horizon(
        self,
        lat: float,
        lon: float,
        observer_height_m: float = 2.0,
        max_distance_m: float = 25000.0,
        radial_step_m: float = 300.0,
        bearing_step_deg: float = 1.0,
        zoom: int = 12,
    ) -> HorizonProfile:
        """Compute synthetic 360-degree horizon profile for a candidate observer position."""
        obs_ground_elev = self.dem.get_elevation(lat, lon, zoom=zoom)
        eye_elev = obs_ground_elev + observer_height_m

        bearings: List[float] = []
        horizon_angles: List[float] = []

        # Number of radial steps along each ray
        num_steps = max(5, int(max_distance_m / radial_step_m))
        distances = [radial_step_m * (i + 1) for i in range(num_steps)]

        # Precompute Earth curvature + refraction corrections per distance
        # curvature_correction = d / (2 * R * (1 - k)) in radians
        effective_radius = EARTH_RADIUS_METERS / (1.0 - ATMOSPHERIC_REFRACTION_K)
        curvature_corrections = [(d / (2.0 * effective_radius)) for d in distances]

        current_bearing = 0.0
        while current_bearing < 360.0:
            max_apparent_angle_deg = -90.0

            for d, curv_corr in zip(distances, curvature_corrections):
                # Calculate target geographic coordinate
                target_coord = destination_point((lat, lon), d, current_bearing)
                target_elev = self.dem.get_elevation(target_coord[0], target_coord[1], zoom=zoom)

                # Apparent vertical elevation angle in radians
                delta_h = target_elev - eye_elev
                raw_angle_rad = math.atan2(delta_h, d)
                apparent_angle_rad = raw_angle_rad - curv_corr
                apparent_angle_deg = math.degrees(apparent_angle_rad)

                if apparent_angle_deg > max_apparent_angle_deg:
                    max_apparent_angle_deg = apparent_angle_deg

            bearings.append(round(current_bearing, 1))
            horizon_angles.append(round(max_apparent_angle_deg, 3))
            current_bearing += bearing_step_deg

        # Identify prominent peak
        max_idx = int(np.argmax(horizon_angles)) if "np" in globals() else horizon_angles.index(max(horizon_angles))
        peak_azimuth = bearings[max_idx]
        peak_angle = horizon_angles[max_idx]

        return HorizonProfile(
            center_coord=(round(lat, 6), round(lon, 6)),
            observer_elevation_m=round(obs_ground_elev, 1),
            bearings_deg=bearings,
            elevation_angles_deg=horizon_angles,
            peak_azimuth_deg=peak_azimuth,
            peak_elevation_deg=peak_angle,
            max_distance_m=max_distance_m,
        )
