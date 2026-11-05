"""HOUND Mountain Ridge Cross-Correlation & Skyline Matcher.

Correlates observed mountain ridge silhouettes against synthetic DEM horizons,
optimizing camera heading, pitch offset, and computing root-mean-squared error (RMSE).
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field

from core.logger import get_logger
from integrations.elevation.horizon_generator import HorizonProfile

logger = get_logger("hound.elevation.matcher")


class RidgeMatchResult(BaseModel):
    candidate_coord: Tuple[float, float]
    best_heading_deg: float
    best_pitch_offset_deg: float
    rmse_deg: float
    correlation_score: float  # Normalized [0.0, 1.0] where 1.0 is exact fit
    angular_uncertainty_deg: float
    matched_points_count: int
    data_source: str = "AWS Terrarium DEM"


class RidgeMatcher:
    """Matches observed mountain ridge profiles against predicted 360° DEM horizons."""

    @staticmethod
    def match_ridge(
        observed_ridge: List[Tuple[float, float]],  # List of (relative_azimuth_deg, elevation_angle_deg)
        horizon_profile: HorizonProfile,
        heading_search_min_deg: float = 0.0,
        heading_search_max_deg: float = 360.0,
        heading_step_deg: float = 1.0,
        max_pitch_offset_deg: float = 5.0,
    ) -> RidgeMatchResult:
        """Cross-correlate observed skyline against 360° horizon profile.

        Args:
            observed_ridge: Array of (delta_azimuth_from_center, apparent_elevation_deg).
            horizon_profile: 360° synthetic horizon profile from candidate viewpoint.
            heading_search_min_deg: Start of candidate heading search window.
            heading_search_max_deg: End of candidate heading search window.
            heading_step_deg: Search resolution in degrees.
            max_pitch_offset_deg: Allowable camera pitch / horizon vertical offset.
        """
        if not observed_ridge:
            raise ValueError("Observed ridge cannot be empty.")

        obs_rel_azimuths = np.array([pt[0] for pt in observed_ridge], dtype=np.float32)
        obs_elevations = np.array([pt[1] for pt in observed_ridge], dtype=np.float32)

        # Build interpolator for the 360° horizon profile
        horizon_bearings = np.array(horizon_profile.bearings_deg, dtype=np.float32)
        horizon_angles = np.array(horizon_profile.elevation_angles_deg, dtype=np.float32)

        # Extend horizon by 360° on both sides to handle wrap-around seamlessly
        extended_bearings = np.concatenate([horizon_bearings - 360.0, horizon_bearings, horizon_bearings + 360.0])
        extended_angles = np.concatenate([horizon_angles, horizon_angles, horizon_angles])

        best_rmse = float("inf")
        best_heading = 0.0
        best_pitch = 0.0

        current_heading = heading_search_min_deg
        while current_heading < heading_search_max_deg:
            # Absolute target bearings for each point in observed ridge
            target_bearings = (current_heading + obs_rel_azimuths) % 360.0

            # Interpolate synthetic DEM horizon at target bearings
            dem_sampled_angles = np.interp(target_bearings, extended_bearings, extended_angles)

            # Solve optimal vertical pitch offset (mean difference)
            delta = dem_sampled_angles - obs_elevations
            pitch_offset = float(np.mean(delta))
            pitch_offset = max(-max_pitch_offset_deg, min(max_pitch_offset_deg, pitch_offset))

            # Compute Root Mean Squared Error (RMSE)
            residuals = dem_sampled_angles - (obs_elevations + pitch_offset)
            rmse = float(np.sqrt(np.mean(residuals**2)))

            if rmse < best_rmse:
                best_rmse = rmse
                best_heading = current_heading
                best_pitch = pitch_offset

            current_heading += heading_step_deg

        # Correlation score: 1.0 / (1.0 + rmse)
        corr_score = 1.0 / (1.0 + best_rmse)

        # Angular uncertainty: increases as RMSE increases
        uncertainty_deg = max(1.0, round(best_rmse * 2.0, 1))

        return RidgeMatchResult(
            candidate_coord=horizon_profile.center_coord,
            best_heading_deg=round(best_heading, 1),
            best_pitch_offset_deg=round(best_pitch, 2),
            rmse_deg=round(best_rmse, 3),
            correlation_score=round(corr_score, 4),
            angular_uncertainty_deg=uncertainty_deg,
            matched_points_count=len(observed_ridge),
        )
