"""HOUND OpenStreetMap Overpass Client.

Features rotating mirror pools, automatic failover, exponential backoff,
persistent SQLite caching, and specialized infrastructure query builders.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from core.configuration import get_config
from core.geographic_search import BoundingBox
from core.logger import get_logger
from integrations.openstreetmap.geometry_parser import OsmFeatureCollection, parse_overpass_json
from storage.database import DatabaseManager

logger = get_logger("hound.osm.client")

# Default public Overpass API mirrors
DEFAULT_OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]


class EndpointHealth(BaseModel):
    url: str
    is_healthy: bool = True
    consecutive_failures: int = 0
    total_requests: int = 0
    last_response_time_ms: float = 0.0
    last_error: Optional[str] = None


class OverpassClient:
    """Resilient OpenStreetMap Overpass API client with failover and SQLite caching."""

    def __init__(
        self,
        mirrors: Optional[List[str]] = None,
        db_manager: Optional[DatabaseManager] = None,
        user_agent: Optional[str] = None,
        timeout_seconds: int = 8,
        max_retries_per_mirror: int = 1,
    ):
        cfg = get_config()
        self.mirrors = mirrors or cfg.integrations.openstreetmap.overpass_mirrors or DEFAULT_OVERPASS_MIRRORS
        self.db = db_manager or DatabaseManager(cfg.paths.database_path)
        self.user_agent = user_agent or cfg.network.user_agent
        self.timeout = timeout_seconds
        self.max_retries = max_retries_per_mirror

        # Endpoint health tracker
        self.health_map: Dict[str, EndpointHealth] = {
            m: EndpointHealth(url=m) for m in self.mirrors
        }

    def execute_query(self, query_ql: str, use_cache: bool = True) -> OsmFeatureCollection:
        """Execute an Overpass QL query with caching, failover, and retry loops."""
        query_clean = query_ql.strip()
        query_hash = hashlib.sha256(query_clean.encode("utf-8")).hexdigest()

        # 1. Check local persistent SQLite cache
        if use_cache:
            cached_data = self._get_cached_response(query_hash)
            if cached_data is not None:
                logger.debug(f"Overpass cache hit for query hash {query_hash[:12]}")
                collection = parse_overpass_json(cached_data)
                collection.status = "cached"
                collection.is_cached = True
                collection.endpoint_used = "sqlite_cache"
                return collection

        # 2. Iterate through mirror pool with failover
        payload = urllib.parse.urlencode({"data": query_clean}).encode("utf-8")
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        }

        # Sort mirrors: prefer healthy mirrors with lowest failure count
        sorted_mirrors = sorted(
            self.mirrors,
            key=lambda m: (self.health_map[m].consecutive_failures, self.health_map[m].last_response_time_ms),
        )

        last_exception = None
        for mirror_url in sorted_mirrors:
            health = self.health_map[mirror_url]
            health.total_requests += 1

            for attempt in range(self.max_retries):
                t0 = time.perf_counter()
                try:
                    logger.debug(f"Querying Overpass mirror {mirror_url} (attempt {attempt + 1})...")
                    req = urllib.request.Request(mirror_url, data=payload, headers=headers, method="POST")
                    with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                        duration_ms = (time.perf_counter() - t0) * 1000.0
                        raw_bytes = resp.read()
                        data = json.loads(raw_bytes.decode("utf-8", errors="replace"))

                        # Update health
                        health.is_healthy = True
                        health.consecutive_failures = 0
                        health.last_response_time_ms = round(duration_ms, 1)
                        health.last_error = None

                        # Save to cache
                        if use_cache:
                            self._save_to_cache(query_hash, query_clean, data)

                        collection = parse_overpass_json(data)
                        collection.status = "empty" if (len(collection.ways) == 0 and len(collection.nodes) == 0) else "success"
                        collection.is_cached = False
                        collection.endpoint_used = mirror_url
                        return collection

                except Exception as e:
                    duration_ms = (time.perf_counter() - t0) * 1000.0
                    err_msg = str(e)
                    health.consecutive_failures += 1
                    health.last_response_time_ms = round(duration_ms, 1)
                    health.last_error = err_msg
                    if health.consecutive_failures >= 2:
                        health.is_healthy = False

                    last_exception = e
                    logger.warning(
                        f"Overpass mirror {mirror_url} failed ({err_msg}). Retrying in {0.5 * (attempt + 1)}s..."
                    )
                    time.sleep(0.5 * (attempt + 1))

        # If all live mirrors fail or network is unavailable, return structured error status
        logger.error(f"All Overpass mirrors failed. Last error: {last_exception}")
        return OsmFeatureCollection(
            status="service_unavailable",
            is_cached=False,
            error_message=str(last_exception) if last_exception else "Service unavailable",
        )

    def query_highways(
        self,
        bbox: BoundingBox,
        highway_types: Optional[List[str]] = None,
    ) -> OsmFeatureCollection:
        """Query highways/roads within bounding box."""
        types_regex = "|".join(highway_types or ["motorway", "trunk", "primary", "secondary"])
        ql = f"""
        [out:json][timeout:{self.timeout}];
        (
          way["highway"~"{types_regex}"]({bbox.to_overpass_str()});
        );
        out body geom qt;
        """
        return self.execute_query(ql)

    def query_bridges(self, bbox: BoundingBox) -> OsmFeatureCollection:
        """Query road and railway bridges within bounding box."""
        ql = f"""
        [out:json][timeout:{self.timeout}];
        (
          way["bridge"="yes"]({bbox.to_overpass_str()});
          way["man_made"="bridge"]({bbox.to_overpass_str()});
        );
        out body geom qt;
        """
        return self.execute_query(ql)

    def query_railways(self, bbox: BoundingBox, electrified_only: bool = False) -> OsmFeatureCollection:
        """Query railway corridors within bounding box."""
        elec_filter = '["electrified"="contact_line"]' if electrified_only else ""
        ql = f"""
        [out:json][timeout:{self.timeout}];
        (
          way["railway"~"rail|narrow_gauge"]{elec_filter}({bbox.to_overpass_str()});
        );
        out body geom qt;
        """
        return self.execute_query(ql)

    def query_railway_stations(self, bbox: BoundingBox) -> OsmFeatureCollection:
        """Query railway stations and halts within bounding box."""
        ql = f"""
        [out:json][timeout:{self.timeout}];
        (
          node["railway"="station"]({bbox.to_overpass_str()});
          node["railway"="halt"]({bbox.to_overpass_str()});
        );
        out body geom qt;
        """
        return self.execute_query(ql)

    def query_power_lines(self, bbox: BoundingBox) -> OsmFeatureCollection:
        """Query electrical transmission power lines within bounding box."""
        ql = f"""
        [out:json][timeout:{self.timeout}];
        (
          way["power"="line"]({bbox.to_overpass_str()});
          way["power"="minor_line"]({bbox.to_overpass_str()});
        );
        out body geom qt;
        """
        return self.execute_query(ql)

    def query_waterways(self, bbox: BoundingBox) -> OsmFeatureCollection:
        """Query rivers, canals, and streams within bounding box."""
        ql = f"""
        [out:json][timeout:{self.timeout}];
        (
          way["waterway"~"river|canal|stream"]({bbox.to_overpass_str()});
        );
        out body geom qt;
        """
        return self.execute_query(ql)

    def query_named_street(self, bbox: BoundingBox, street_name: str) -> OsmFeatureCollection:
        """Search for a specific named street within bounding box."""
        escaped_name = street_name.replace('"', '\\"')
        ql = f"""
        [out:json][timeout:{self.timeout}];
        (
          way["highway"]["name"~"{escaped_name}",i]({bbox.to_overpass_str()});
        );
        out body geom qt;
        """
        return self.execute_query(ql)

    def _get_cached_response(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached JSON from SQLite database."""
        try:
            with self.db.get_connection() as conn:
                cur = conn.execute("SELECT response_json FROM osm_cache WHERE query_hash = ?", (query_hash,))
                row = cur.fetchone()
                if row:
                    return json.loads(row["response_json"])
        except Exception as e:
            logger.debug(f"Failed to read OSM cache: {e}")
        return None

    def _save_to_cache(self, query_hash: str, query_text: str, data: Dict[str, Any]) -> None:
        """Save Overpass JSON response to SQLite cache."""
        try:
            element_count = len(data.get("elements", []))
            json_str = json.dumps(data)
            now_str = datetime.now(timezone.utc).isoformat()
            with self.db.get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO osm_cache (query_hash, query_text, response_json, element_count, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (query_hash, query_text, json_str, element_count, now_str),
                )
                conn.commit()
        except Exception as e:
            logger.debug(f"Failed to write OSM cache: {e}")
