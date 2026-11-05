"""HOUND Geographic Visualizer.

Renders candidate location maps on georeferenced satellite mosaics, 360° synthetic
horizon elevation profiles, and candidate ranking comparison cards.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from agent.candidate_comparator import CandidateEvaluation
from agent.candidate_generator import GeographicCandidate
from core.geographic_search import BoundingBox
from core.logger import get_logger
from integrations.elevation.horizon_generator import HorizonProfile
from integrations.satellite.tile_service import SatelliteTileService

logger = get_logger("hound.reports.geo_visualizer")


class GeographicVisualizer:
    """Renders geographic maps, horizon plots, and candidate comparison cards."""

    @classmethod
    def render_candidate_map(
        cls,
        candidates: List[GeographicCandidate],
        output_path: Path,
        tile_service: Optional[SatelliteTileService] = None,
        zoom: int = 13,
    ) -> Path:
        """Render candidate points and bounding envelopes on a real map mosaic."""
        tiles = tile_service or SatelliteTileService()
        valid_cands = [c for c in candidates if c.lat is not None and c.lon is not None]

        if not valid_cands:
            # Fallback blank map if no coordinates
            img = Image.new("RGB", (640, 480), color=(40, 45, 50))
            draw = ImageDraw.Draw(img)
            draw.text((20, 20), "No discrete geographic coordinates to map.", fill=(200, 200, 200))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path, format="JPEG")
            return output_path

        # Compute bounding envelope covering all candidates
        lats = [c.lat for c in valid_cands]
        lons = [c.lon for c in valid_cands]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)

        envelope = BoundingBox(
            min_lat=min(lats) - 0.04,
            min_lon=min(lons) - 0.05,
            max_lat=max(lats) + 0.04,
            max_lon=max(lons) + 0.05,
        )

        mosaic_res = None
        for z in [zoom, max(2, zoom - 4), 5, 3]:
            mosaic_res = tiles.build_mosaic(envelope, zoom=z, provider="osm", max_tiles=16)
            if mosaic_res is not None:
                break

        if mosaic_res is not None:
            base_img, cov_bbox = mosaic_res
        else:
            base_img = Image.new("RGB", (800, 600), color=(220, 225, 230))
            cov_bbox = envelope

        draw = ImageDraw.Draw(base_img)
        w, h = base_img.size

        # Plot each candidate
        for idx, cand in enumerate(valid_cands):
            px, py = SatelliteTileService.geo_to_pixel(cand.lat, cand.lon, cov_bbox, w, h)
            color = (0, 200, 0) if cand.status == "active" else (220, 50, 50)

            # Draw radius ring if radius is specified
            if cand.radius_m:
                r_px = max(6, int((cand.radius_m / 1000.0) * 12.0))
                draw.ellipse([px - r_px, py - r_px, px + r_px, py + r_px], outline=color + (180,), width=2)

            # Draw center point
            draw.ellipse([px - 6, py - 6, px + 6, py + 6], fill=(255, 255, 255), outline=color, width=2)

            # Draw candidate label
            label = f"#{idx + 1} {cand.name[:25]}"
            draw.rectangle([px + 10, py - 10, px + 10 + len(label) * 7, py + 10], fill=(30, 30, 35))
            draw.text((px + 14, py - 6), label, fill=(255, 255, 255))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        base_img.save(output_path, format="JPEG", quality=90)
        logger.debug(f"Saved candidate map to {output_path}")
        return output_path

    @classmethod
    def render_horizon_profile_plot(
        cls,
        profile: HorizonProfile,
        output_path: Path,
        observed_ridge: Optional[List[Tuple[float, float]]] = None,
        plot_size: Tuple[int, int] = (900, 350),
    ) -> Path:
        """Render apparent elevation angle vs. azimuth bearing plot."""
        w, h = plot_size
        img = Image.new("RGB", (w, h), color=(25, 30, 35))
        draw = ImageDraw.Draw(img)

        # Plot boundaries
        pad_l, pad_r, pad_t, pad_b = 60, 40, 40, 50
        plot_w = w - pad_l - pad_r
        plot_h = h - pad_t - pad_b

        # Min / max angles on Y axis
        all_angles = profile.elevation_angles_deg
        min_deg = min(-2.0, min(all_angles) - 1.0)
        max_deg = max(5.0, max(all_angles) + 2.0)
        deg_span = max_deg - min_deg

        # Title
        title = (
            f"360° Terrain Horizon Profile: ({profile.center_coord[0]:.4f}, {profile.center_coord[1]:.4f}) "
            f"— Elev: {profile.observer_elevation_m:.0f}m"
        )
        draw.text((pad_l, 12), title, fill=(0, 220, 255))

        # Draw grid lines and cardinal directions
        cardinals = [(0.0, "N"), (90.0, "E"), (180.0, "S"), (270.0, "W"), (360.0, "N")]
        for bearing, label in cardinals:
            bx = pad_l + int((bearing / 360.0) * plot_w)
            draw.line([(bx, pad_t), (bx, pad_t + plot_h)], fill=(60, 70, 80), width=1)
            draw.text((bx - 4, pad_t + plot_h + 8), label, fill=(180, 190, 200))

        # Draw horizontal zero-elevation line
        zero_y = pad_t + int(((max_deg - 0.0) / deg_span) * plot_h)
        draw.line([(pad_l, zero_y), (pad_l + plot_w, zero_y)], fill=(120, 130, 140), width=1)
        draw.text((pad_l - 40, zero_y - 6), "0.0°", fill=(140, 150, 160))

        # Convert profile (bearing, angle) to pixel coordinates
        pts: List[Tuple[int, int]] = []
        for b, ang in zip(profile.bearings_deg, profile.elevation_angles_deg):
            px = pad_l + int((b / 360.0) * plot_w)
            norm_y = (max_deg - ang) / deg_span
            py = pad_t + int(norm_y * plot_h)
            pts.append((px, py))

        # Fill terrain polygon
        poly_pts = [(pad_l, pad_t + plot_h)] + pts + [(pad_l + plot_w, pad_t + plot_h)]
        draw.polygon(poly_pts, fill=(45, 60, 50))
        draw.line(pts, fill=(50, 205, 50), width=2)

        # Plot observed ridge overlay if provided
        if observed_ridge:
            obs_pts = []
            for b_rel, ang in observed_ridge:
                b_abs = (profile.peak_azimuth_deg + b_rel) % 360.0
                px = pad_l + int((b_abs / 360.0) * plot_w)
                py = pad_t + int(((max_deg - ang) / deg_span) * plot_h)
                obs_pts.append((px, py))
            draw.line(obs_pts, fill=(255, 69, 0), width=3)
            draw.text((pad_l + 20, pad_t + 20), "Red = Observed Ridge | Green = Synthetic DEM", fill=(255, 120, 50))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, format="JPEG", quality=90)
        logger.debug(f"Saved horizon plot to {output_path}")
        return output_path
