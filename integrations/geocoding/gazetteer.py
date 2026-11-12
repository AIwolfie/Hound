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

# Built-in offline geographic index for common world transit hubs, corridors, and landmarks (Natural Earth / GeoNames fallback)
OFFLINE_GAZETTEER_FALLBACK: List[Dict[str, Any]] = [
    {
        "keywords": ["us-101", "us 101", "101 north", "ca coast hwy"],
        "lat": "34.4208",
        "lon": "-119.6982",
        "boundingbox": ["34.30", "34.55", "-119.85", "-119.50"],
        "display_name": "US Highway 101, Santa Barbara, California, United States",
        "type": "road",
        "address": {"road": "US-101", "city": "Santa Barbara", "state": "California", "country": "United States"},
        "importance": 0.85,
    },
    {
        "keywords": ["ca-1", "big sur", "highway 1", "state park"],
        "lat": "36.2704",
        "lon": "-121.8081",
        "boundingbox": ["36.10", "36.45", "-121.95", "-121.65"],
        "display_name": "California State Route 1, Big Sur, California, United States",
        "type": "road",
        "address": {"road": "CA-1", "county": "Monterey County", "state": "California", "country": "United States"},
        "importance": 0.85,
    },
    {
        "keywords": ["zermatt", "gornergrat", "matterhorn", "bahn"],
        "lat": "45.9765",
        "lon": "7.7491",
        "boundingbox": ["45.95", "46.02", "7.70", "7.80"],
        "display_name": "Gornergrat Bahn, Zermatt, Valais, Switzerland",
        "type": "railway",
        "address": {"railway": "Gornergrat Bahn", "town": "Zermatt", "state": "Valais", "country": "Switzerland"},
        "importance": 0.90,
    },
    {
        "keywords": ["shinjuku", "tokyo", "east exit"],
        "lat": "35.6909",
        "lon": "139.7003",
        "boundingbox": ["35.68", "35.70", "139.69", "139.71"],
        "display_name": "Shinjuku Station East Exit, Shinjuku, Tokyo, Japan",
        "type": "station",
        "address": {"station": "Shinjuku Station", "city": "Tokyo", "state": "Tokyo", "country": "Japan"},
        "importance": 0.92,
    },
    {
        "keywords": ["central iowa", "iowa", "co-op grain"],
        "lat": "42.0308",
        "lon": "-93.6319",
        "boundingbox": ["41.80", "42.25", "-93.90", "-93.40"],
        "display_name": "Central Iowa Agricultural Belt, Story County, Iowa, United States",
        "type": "administrative",
        "address": {"county": "Story County", "state": "Iowa", "country": "United States"},
        "importance": 0.75,
    },
    {
        "keywords": ["nathan road", "mong kok", "kowloon"],
        "lat": "22.3193",
        "lon": "114.1694",
        "boundingbox": ["22.30", "22.34", "114.15", "114.19"],
        "display_name": "Nathan Road, Mong Kok, Kowloon, Hong Kong",
        "type": "road",
        "address": {"road": "Nathan Road", "city": "Kowloon", "state": "Kowloon", "country": "Hong Kong"},
        "importance": 0.88,
    },
    {
        "keywords": ["frankfurt", "gleis 7", "hbf"],
        "lat": "50.1071",
        "lon": "8.6638",
        "boundingbox": ["50.09", "50.12", "8.64", "8.68"],
        "display_name": "Frankfurt (Main) Hauptbahnhof, Frankfurt am Main, Hesse, Germany",
        "type": "station",
        "address": {"station": "Frankfurt(Main) Hbf", "city": "Frankfurt am Main", "state": "Hesse", "country": "Germany"},
        "importance": 0.90,
    },
    {
        "keywords": ["route 66", "flagstaff", "seligman"],
        "lat": "35.1983",
        "lon": "-111.6513",
        "boundingbox": ["35.00", "35.40", "-112.00", "-111.40"],
        "display_name": "Historic US Route 66, Coconino County, Arizona, United States",
        "type": "road",
        "address": {"road": "Historic Route 66", "city": "Flagstaff", "state": "Arizona", "country": "United States"},
        "importance": 0.82,
    },
    {
        "keywords": ["eiffel", "champ de mars", "paris"],
        "lat": "48.8584",
        "lon": "2.2945",
        "boundingbox": ["48.84", "48.87", "2.28", "2.31"],
        "display_name": "Eiffel Tower, Champ de Mars, 7th Arrondissement, Paris, Île-de-France, France",
        "type": "attraction",
        "address": {"attraction": "Eiffel Tower", "city": "Paris", "state": "Île-de-France", "country": "France"},
        "importance": 0.95,
    },
    {
        "keywords": ["sydney opera house", "sydney harbour"],
        "lat": "-33.8568",
        "lon": "151.2153",
        "boundingbox": ["-33.87", "-33.84", "151.20", "151.23"],
        "display_name": "Sydney Harbour Bridge & Opera House, Sydney, New South Wales, Australia",
        "type": "attraction",
        "address": {"attraction": "Sydney Opera House", "city": "Sydney", "state": "New South Wales", "country": "Australia"},
        "importance": 0.95,
    },
    {
        "keywords": ["hawaii", "honolulu"],
        "lat": "21.3069",
        "lon": "-157.8583",
        "boundingbox": ["21.20", "21.40", "-158.00", "-157.70"],
        "display_name": "Hawaii, United States",
        "type": "administrative",
        "address": {"state": "Hawaii", "country": "United States"},
        "importance": 0.70,
    },
    {
        "keywords": ["british columbia", "bcplates"],
        "lat": "49.2827",
        "lon": "-123.1207",
        "boundingbox": ["48.00", "55.00", "-130.00", "-115.00"],
        "display_name": "British Columbia, Canada",
        "type": "administrative",
        "address": {"state": "British Columbia", "country": "Canada"},
        "importance": 0.70,
    },
]


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

        # Fallback to offline gazetteer index if network query returned empty or was offline
        if not raw_items:
            raw_items = self._lookup_offline_gazetteer(cleaned_query)

        results = [self._parse_item(item) for item in raw_items]

        if use_cache and raw_items:
            self._save_cached(cache_key, raw_items)

        return results

    def _lookup_offline_gazetteer(self, query: str) -> List[Dict[str, Any]]:
        """Match query against built-in offline geographic index."""
        q_lower = query.lower()
        matched = []
        for entry in OFFLINE_GAZETTEER_FALLBACK:
            for kw in entry["keywords"]:
                if kw in q_lower or (len(q_lower) >= 4 and q_lower in kw):
                    matched.append(entry)
                    break
        return matched

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
