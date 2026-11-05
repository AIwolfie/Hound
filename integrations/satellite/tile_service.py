"""HOUND Provider-Independent Satellite & Map Tile Service.

Retrieves, caches, and stitches multi-tile satellite mosaics, providing
geo-referencing between pixel coordinates and WGS-84 geographic coordinates.
"""
from __future__ import annotations

import io
import math
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image

from core.configuration import get_config
from core.geographic_search import BoundingBox, lat_lon_to_tile, tile_to_bounding_box
from core.logger import get_logger

logger = get_logger("hound.satellite")

DEFAULT_PROVIDERS = {
    "osm": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    "carto_light": "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
}


class SatelliteTileService:
    """Manages satellite and map tile retrieval, persistent caching, and mosaic assembly."""

    def __init__(
        self,
        cache_dir: Optional[Path | str] = None,
        timeout: int = 15,
    ):
        cfg = get_config()
        self.cache_dir = Path(cache_dir or cfg.paths.cache_dir / "satellite")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.user_agent = cfg.network.user_agent
        self.timeout = timeout
        self.providers: Dict[str, str] = dict(DEFAULT_PROVIDERS)
        if cfg.integrations.satellite.google_satellite_url:
            self.providers["google_sat"] = cfg.integrations.satellite.google_satellite_url

    def fetch_tile(self, tile_x: int, tile_y: int, zoom: int, provider: str = "osm") -> Optional[Image.Image]:
        """Fetch and cache a single map/satellite tile."""
        url_template = self.providers.get(provider, self.providers["osm"])
        cached_path = self.cache_dir / provider / str(zoom) / str(tile_x) / f"{tile_y}.png"

        if cached_path.is_file():
            try:
                return Image.open(cached_path).convert("RGB")
            except Exception as e:
                logger.debug(f"Failed to read cached satellite tile {cached_path}: {e}")

        url = url_template.format(z=zoom, x=tile_x, y=tile_y)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = resp.read()
                img = Image.open(io.BytesIO(data)).convert("RGB")
                cached_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(cached_path, format="PNG")
                return img
        except Exception as e:
            logger.debug(f"Satellite tile fetch failed ({url}): {e}")
            return None

    def build_mosaic(
        self,
        bbox: BoundingBox,
        zoom: int = 15,
        provider: str = "osm",
        max_tiles: int = 25,
    ) -> Optional[Tuple[Image.Image, BoundingBox]]:
        """Fetch and stitch bounding box tiles into a georeferenced mosaic image.

        Returns:
            Tuple of (Mosaic Image, Exact Covering BoundingBox) or None if exceeding tile budget.
        """
        # Compute tile coordinates covering top-left and bottom-right corners
        nw_tx, nw_ty, _, _ = lat_lon_to_tile(bbox.max_lat, bbox.min_lon, zoom)
        se_tx, se_ty, _, _ = lat_lon_to_tile(bbox.min_lat, bbox.max_lon, zoom)

        min_tx, max_tx = min(nw_tx, se_tx), max(nw_tx, se_tx)
        min_ty, max_ty = min(nw_ty, se_ty), max(nw_ty, se_ty)

        num_tiles_x = (max_tx - min_tx) + 1
        num_tiles_y = (max_ty - min_ty) + 1
        total_tiles = num_tiles_x * num_tiles_y

        if total_tiles > max_tiles:
            logger.warning(
                f"Requested mosaic requires {total_tiles} tiles, exceeding safe budget of {max_tiles}. Aborting."
            )
            return None

        mosaic_w = num_tiles_x * 256
        mosaic_h = num_tiles_y * 256
        mosaic = Image.new("RGB", (mosaic_w, mosaic_h), color=(230, 230, 230))

        for ty_idx, ty in enumerate(range(min_ty, max_ty + 1)):
            for tx_idx, tx in enumerate(range(min_tx, max_tx + 1)):
                tile_img = self.fetch_tile(tx, ty, zoom, provider=provider)
                if tile_img is not None:
                    mosaic.paste(tile_img, (tx_idx * 256, ty_idx * 256))

        # Calculate exact bounding box covering all stitched tiles
        nw_box = tile_to_bounding_box(min_tx, min_ty, zoom)
        se_box = tile_to_bounding_box(max_tx, max_ty, zoom)

        covering_bbox = BoundingBox(
            min_lat=se_box.min_lat,
            min_lon=nw_box.min_lon,
            max_lat=nw_box.max_lat,
            max_lon=se_box.max_lon,
        )

        return (mosaic, covering_bbox)

    @staticmethod
    def geo_to_pixel(
        lat: float, lon: float, bbox: BoundingBox, img_width: int, img_height: int
    ) -> Tuple[int, int]:
        """Convert (lat, lon) to image pixel coordinate (x, y) on a mosaic."""
        span_lon = bbox.max_lon - bbox.min_lon
        span_lat = bbox.max_lat - bbox.min_lat
        if span_lon <= 0 or span_lat <= 0:
            return (0, 0)

        # Invert latitude since image Y increases downwards
        norm_x = (lon - bbox.min_lon) / span_lon
        norm_y = (bbox.max_lat - lat) / span_lat

        px = int(round(norm_x * img_width))
        py = int(round(norm_y * img_height))
        return (max(0, min(img_width - 1, px)), max(0, min(img_height - 1, py)))
