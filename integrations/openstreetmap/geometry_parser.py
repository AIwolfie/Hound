"""HOUND OpenStreetMap Geometry & Feature Parser.

Parses Overpass QL JSON responses into structured nodes, ways, and relations,
computing geodetic lengths, bearings, and bounding envelopes.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from core.geographic_search import BoundingBox, forward_azimuth, haversine_distance
from core.logger import get_logger

logger = get_logger("hound.osm.parser")


class OsmNode(BaseModel):
    id: int
    lat: float
    lon: float
    tags: Dict[str, str] = Field(default_factory=dict)


class OsmWay(BaseModel):
    id: int
    node_ids: List[int] = Field(default_factory=list)
    coordinates: List[Tuple[float, float]] = Field(default_factory=list)  # [(lat, lon), ...]
    tags: Dict[str, str] = Field(default_factory=dict)
    length_meters: float = 0.0
    primary_bearing_deg: Optional[float] = None
    bounding_box: Optional[BoundingBox] = None


class OsmRelation(BaseModel):
    id: int
    members: List[Dict[str, Any]] = Field(default_factory=list)
    tags: Dict[str, str] = Field(default_factory=dict)


class OsmFeatureCollection(BaseModel):
    nodes: List[OsmNode] = Field(default_factory=list)
    ways: List[OsmWay] = Field(default_factory=list)
    relations: List[OsmRelation] = Field(default_factory=list)
    raw_element_count: int = 0
    query_timestamp: Optional[str] = None
    status: str = "success"  # "success", "empty", "cached", "service_unavailable", "timeout", "error"
    is_cached: bool = False
    endpoint_used: Optional[str] = None
    error_message: Optional[str] = None


def parse_overpass_json(data: Dict[str, Any]) -> OsmFeatureCollection:
    """Parse raw Overpass JSON output into structured OsmFeatureCollection."""
    elements = data.get("elements", [])
    node_map: Dict[int, OsmNode] = {}
    nodes: List[OsmNode] = []
    ways: List[OsmWay] = []
    relations: List[OsmRelation] = []

    # First pass: Index all nodes
    for el in elements:
        el_type = el.get("type")
        if el_type == "node":
            node_id = el["id"]
            node = OsmNode(
                id=node_id,
                lat=float(el["lat"]),
                lon=float(el["lon"]),
                tags=el.get("tags", {}),
            )
            node_map[node_id] = node
            if node.tags:  # Keep tagged nodes (e.g. railway stations, POIs)
                nodes.append(node)

    # Second pass: Process ways and resolve coordinate strings
    for el in elements:
        el_type = el.get("type")
        if el_type == "way":
            way_id = el["id"]
            tags = el.get("tags", {})
            node_refs = el.get("nodes", [])

            # Resolve coordinates from geometry list if present, else lookup in node_map
            coords: List[Tuple[float, float]] = []
            if "geometry" in el:
                for pt in el["geometry"]:
                    coords.append((float(pt["lat"]), float(pt["lon"])))
            elif node_refs:
                for ref in node_refs:
                    if ref in node_map:
                        coords.append((node_map[ref].lat, node_map[ref].lon))

            # Calculate total length, bearing, and bounding envelope
            length_m = 0.0
            bearing: Optional[float] = None
            bbox: Optional[BoundingBox] = None

            if len(coords) >= 2:
                for i in range(len(coords) - 1):
                    length_m += haversine_distance(coords[i], coords[i + 1])
                bearing = forward_azimuth(coords[0], coords[-1])

                lats = [c[0] for c in coords]
                lons = [c[1] for c in coords]
                bbox = BoundingBox(
                    min_lat=min(lats),
                    min_lon=min(lons),
                    max_lat=max(lats),
                    max_lon=max(lons),
                )

            ways.append(
                OsmWay(
                    id=way_id,
                    node_ids=node_refs,
                    coordinates=coords,
                    tags=tags,
                    length_meters=round(length_m, 2),
                    primary_bearing_deg=bearing,
                    bounding_box=bbox,
                )
            )

        elif el_type == "relation":
            relations.append(
                OsmRelation(
                    id=el["id"],
                    members=el.get("members", []),
                    tags=el.get("tags", {}),
                )
            )

    return OsmFeatureCollection(
        nodes=nodes,
        ways=ways,
        relations=relations,
        raw_element_count=len(elements),
        query_timestamp=data.get("osm3s", {}).get("timestamp_osm_base"),
    )
