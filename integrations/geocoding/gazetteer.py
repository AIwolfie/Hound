"""HOUND Geocoding & Gazetteer Engine.

Provides forward/reverse geocoding, administrative division hierarchy resolution,
bounding-box scoping, and local SQLite persistent caching.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from core.configuration import get_config
from core.geographic_search import BoundingBox
from core.logger import get_logger
from storage.database import DatabaseManager

logger = get_logger("hound.geocoding")


class GeocodingResult(BaseModel):
    display_name: str
    lat: float
    lon: float
    bbox: BoundingBox
    place_type: str  # "city", "administrative", "country", "road", "poi"
    country: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = None
    osm_id: Optional[int] = None
    importance: float = 0.5


class GazetteerService:
    """Manages forward/reverse geocoding with local persistent SQLite caching."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        nominatim_url: Optional[str] = None,
        rate_limit_delay: float = 1.0,
    ):
        cfg = get_config()
        self.db = db_manager or DatabaseManager(cfg.paths.database_path)
        self.nominatim_url = nominatim_url or cfg.integrations.openstreetmap.nominatim_url
        self.user_agent = cfg.network.user_agent
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time = 0.0

    def forward_geocode(
        self,
        query: str,
        country_code: Optional[str] = None,
        limit: int = 5,
        use_cache: bool = True,
    ) -> List[GeocodingResult]:
        """Resolve a place name or address query into geocoded candidates with bounding boxes."""
        cleaned_query = query.strip()
        cache_key = f"fwd:{cleaned_query.lower()}:{country_code or ''}:{limit}"

        if use_cache:
            cached = self._get_cached(cache_key)
            if cached is not None:
                return [self._parse_item(item) for item in cached]

        params: Dict[str, Any] = {
            "q": cleaned_query,
            "format": "jsonv2",
            "limit": limit,
            "addressdetails": 1,
        }
        if country_code:
            params["countrycodes"] = country_code.lower()

        url = f"{self.nominatim_url}?{urllib.parse.urlencode(params)}"
        raw_items = self._fetch_json(url)

        results = [self._parse_item(item) for item in raw_items]

        if use_cache and raw_items:
            self._save_cached(cache_key, raw_items)

        return results

    def reverse_geocode(
        self,
        lat: float,
        lon: float,
        zoom: int = 14,
        use_cache: bool = True,
    ) -> Optional[GeocodingResult]:
        """Resolve a coordinate point to its administrative hierarchy."""
        cache_key = f"rev:{round(lat, 4)}:{round(lon, 4)}:{zoom}"

        if use_cache:
            cached = self._get_cached(cache_key)
            if cached and isinstance(cached, list) and len(cached) > 0:
                return self._parse_item(cached[0])

        params = {
            "lat": lat,
            "lon": lon,
            "format": "jsonv2",
            "zoom": zoom,
            "addressdetails": 1,
        }
        url = f"https://nominatim.openstreetmap.org/reverse?{urllib.parse.urlencode(params)}"
        raw = self._fetch_json(url)

        if isinstance(raw, dict) and "error" not in raw:
            res = self._parse_item(raw)
            if use_cache:
                self._save_cached(cache_key, [raw])
            return res

        return None

    def _fetch_json(self, url: str) -> Any:
        """Fetch JSON with rate-limiting and user-agent compliance."""
        # Enforce rate limit
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)

        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                self._last_request_time = time.time()
                raw_bytes = resp.read()
                return json.loads(raw_bytes.decode("utf-8", errors="replace"))
        except Exception as e:
            logger.warning(f"Geocoding request failed: {e}")
            return []

    def _parse_item(self, item: Dict[str, Any]) -> GeocodingResult:
        """Parse raw Nominatim result dictionary into GeocodingResult."""
        lat = float(item["lat"])
        lon = float(item["lon"])

        # Nominatim boundingbox: [south, north, west, east]
        raw_box = item.get("boundingbox", [lat - 0.05, lat + 0.05, lon - 0.05, lon + 0.05])
        south, north = float(raw_box[0]), float(raw_box[1])
        west, east = float(raw_box[2]), float(raw_box[3])
        bbox = BoundingBox(min_lat=south, min_lon=west, max_lat=north, max_lon=east)

        addr = item.get("address", {})
        country = addr.get("country")
        state = addr.get("state") or addr.get("province") or addr.get("region")
        city = (
            addr.get("city")
            or addr.get("town")
            or addr.get("municipality")
            or addr.get("village")
            or addr.get("county")
        )

        return GeocodingResult(
            display_name=item.get("display_name", f"{lat:.4f}, {lon:.4f}"),
            lat=lat,
            lon=lon,
            bbox=bbox,
            place_type=item.get("type", "unknown"),
            country=country,
            state=state,
            city=city,
            osm_id=item.get("osm_id"),
            importance=float(item.get("importance", 0.5)),
        )

    def _get_cached(self, key: str) -> Optional[Any]:
        """Fetch cached geocoding result from SQLite."""
        try:
            with self.db.get_connection() as conn:
                cur = conn.execute("SELECT result_json FROM gazetteer_cache WHERE query_key = ?", (key,))
                row = cur.fetchone()
                if row:
                    return json.loads(row["result_json"])
        except Exception as e:
            logger.debug(f"Gazetteer cache read error: {e}")
        return None

    def _save_cached(self, key: str, data: Any) -> None:
        """Save geocoding result to SQLite cache."""
        try:
            json_str = json.dumps(data)
            now_str = datetime.now(timezone.utc).isoformat()
            with self.db.get_connection() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO gazetteer_cache (query_key, result_json, created_at) VALUES (?, ?, ?)",
                    (key, json_str, now_str),
                )
                conn.commit()
        except Exception as e:
            logger.debug(f"Gazetteer cache write error: {e}")
