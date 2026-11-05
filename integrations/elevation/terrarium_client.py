"""HOUND Terrarium Digital Elevation Model (DEM) Client.

Retrieves, decodes, and caches AWS Terrarium elevation tiles, computing precise
ground elevations via bilinear sub-pixel interpolation.
"""
from __future__ import annotations

import io
import math
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
from PIL import Image

from core.configuration import get_config
from core.geographic_search import lat_lon_to_subpixel_tile, lat_lon_to_tile
from core.logger import get_logger

logger = get_logger("hound.elevation.terrarium")


def decode_terrarium_elevation(r: int, g: int, b: int) -> float:
    """Decode 24-bit RGB Terrarium pixel value to elevation in meters.

    Formula: (R * 256.0 + G + B / 256.0) - 32768.0
    """
    return (float(r) * 256.0 + float(g) + float(b) / 256.0) - 32768.0


class TerrariumClient:
    """Fetches and decodes AWS Terrarium elevation tiles with persistent disk caching."""

    def __init__(
        self,
        cache_dir: Optional[Path | str] = None,
        base_url: Optional[str] = None,
        timeout: int = 15,
    ):
        cfg = get_config()
        self.cache_dir = Path(cache_dir or cfg.paths.cache_dir / "elevation")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url or cfg.integrations.elevation.terrarium_tile_url
        self.user_agent = cfg.network.user_agent
        self.timeout = timeout
        self._tile_cache: Dict[Tuple[int, int, int], np.ndarray] = {}

    def get_elevation(self, lat: float, lon: float, zoom: int = 12) -> float:
        """Get ground elevation in meters at (lat, lon) with bilinear interpolation."""
        tile_x, tile_y, x0, y0, fx, fy = lat_lon_to_subpixel_tile(lat, lon, zoom)
        tile_array = self.get_tile(tile_x, tile_y, zoom)
        if tile_array is None:
            return 0.0

        # Sub-pixel bilinear interpolation inside the 256x256 tile
        # tile_array shape: (256, 256) float elevations
        x1 = min(255, x0 + 1)
        y1 = min(255, y0 + 1)

        e00 = float(tile_array[y0, x0])
        e10 = float(tile_array[y0, x1])
        e01 = float(tile_array[y1, x0])
        e11 = float(tile_array[y1, x1])

        elev = (
            e00 * (1.0 - fx) * (1.0 - fy)
            + e10 * fx * (1.0 - fy)
            + e01 * (1.0 - fx) * fy
            + e11 * fx * fy
        )
        return float(round(elev, 2))

    def get_tile(self, tile_x: int, tile_y: int, zoom: int) -> Optional[np.ndarray]:
        """Fetch and decode a Terrarium elevation tile, returning (256, 256) float32 array in meters."""
        key = (zoom, tile_x, tile_y)
        if key in self._tile_cache:
            return self._tile_cache[key]

        cached_file = self.cache_dir / str(zoom) / str(tile_x) / f"{tile_y}.png"
        raw_img = None

        if cached_file.is_file():
            try:
                raw_img = Image.open(cached_file).convert("RGB")
            except Exception as e:
                logger.debug(f"Failed to read cached elevation tile {cached_file}: {e}")

        if raw_img is None:
            url = self.base_url.format(z=zoom, x=tile_x, y=tile_y)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = resp.read()
                    raw_img = Image.open(io.BytesIO(data)).convert("RGB")
                    # Save to disk cache
                    cached_file.parent.mkdir(parents=True, exist_ok=True)
                    raw_img.save(cached_file, format="PNG")
            except Exception as e:
                logger.debug(f"Terrarium tile fetch failed ({url}): {e}")
                return None

        # Decode RGB values into elevation meters
        arr = np.array(raw_img, dtype=np.float32)
        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]
        elevations = (r * 256.0 + g + b / 256.0) - 32768.0

        # Cache in memory (cap at 200 tiles)
        if len(self._tile_cache) > 200:
            self._tile_cache.clear()
        self._tile_cache[key] = elevations
        return elevations
