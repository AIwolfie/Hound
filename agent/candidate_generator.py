"""HOUND Geographic Candidate Generator.

Synthesizes visual intelligence clues (OCR road designations, street names,
solar latitude bands, and infrastructure) into grounded geographic candidate hypotheses.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from core.geographic_search import BoundingBox
from core.image_loader import ImageRecord
from core.logger import get_logger
from integrations.geocoding.gazetteer import GazetteerService
from storage.database import DatabaseManager
from storage.evidence_store import CandidateItem, EvidenceItem, EvidenceStore

logger = get_logger("hound.agent.candidate_gen")


class GeographicCandidate(BaseModel):
    id: str = Field(default_factory=lambda: f"C{uuid.uuid4().hex[:6].upper()}")
    investigation_id: str
    parent_id: Optional[str] = None
    level: str  # "country", "admin", "city", "region", "area", "point"
    name: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    bbox: Optional[BoundingBox] = None
    radius_m: Optional[float] = None
    score: float = 1.0
    status: str = "active"  # "active", "excluded", "confirmed", "downweighted"
    seed_evidence_id: Optional[str] = None
    seed_evidence_kind: Optional[str] = None
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    contradictory_evidence_ids: List[str] = Field(default_factory=list)
    untested_checks: List[str] = Field(default_factory=list)
    falsification_condition: Optional[str] = None
    spatial_uncertainty_m: float = 50000.0
    ranking_explanation: str = ""


class GeographicCandidateGenerator:
    """Generates geographically grounded hypotheses from visual evidence."""

    def __init__(
        self,
        gazetteer: Optional[GazetteerService] = None,
        db_manager: Optional[DatabaseManager] = None,
    ):
        self.gazetteer = gazetteer or GazetteerService()
        self.db = db_manager

    def generate_candidates(
        self,
        investigation_id: str,
        image_record: ImageRecord,
        evidence_items: List[EvidenceItem],
    ) -> List[GeographicCandidate]:
        """Synthesize visual evidence into candidate hypotheses across geographic scales."""
        candidates: List[GeographicCandidate] = []

        # -------------------------------------------------------------
        # 1. Direct Ground Truth Anchor: Embedded EXIF GPS
        # -------------------------------------------------------------
        if image_record.exif.gps:
            gps = image_record.exif.gps
            gps_ev = next((e for e in evidence_items if e.kind == "exif" and "Embedded GPS" in e.text), None)
            ev_ids = [gps_ev.id] if gps_ev else []

            # Reverse geocode to resolve place name
            geo_res = self.gazetteer.reverse_geocode(gps.latitude, gps.longitude)
            place_name = geo_res.display_name if geo_res else f"GPS Coordinate ({gps.latitude:.4f}, {gps.longitude:.4f})"

            cand_gps = GeographicCandidate(
                investigation_id=investigation_id,
                level="point",
                name=f"Anchor: {place_name}",
                lat=gps.latitude,
                lon=gps.longitude,
                bbox=BoundingBox.from_point_and_radius(gps.latitude, gps.longitude, 100.0),
                radius_m=50.0,
                score=10.0,
                status="active",
                seed_evidence_id=gps_ev.id if gps_ev else None,
                seed_evidence_kind="gps",
                supporting_evidence_ids=ev_ids,
                spatial_uncertainty_m=50.0,
                ranking_explanation="Direct embedded EXIF GPS telemetry verified in image header.",
            )
            candidates.append(cand_gps)

        # -------------------------------------------------------------
        # 2. Text Clues: Road Identifiers & Named Routes
        # -------------------------------------------------------------
        road_evs = [
            e for e in evidence_items
            if e.kind == "ocr" and ("Category: road_number" in e.text or "road sign" in e.text.lower() or "highway" in e.text.lower())
        ]
        for rev in road_evs:
            # Extract road string e.g. "Recognized text: "US-101 North""
            match = re.search(r'Recognized text:\s*"([^"]+)"', rev.text)
            if match:
                road_name = match.group(1).strip()
            else:
                road_m = re.search(r'(?:US|I|Route|Highway|State Route|CA)[-\s]?\d+[A-Z]?(?:\s+[A-Za-z]+)?', rev.text, re.IGNORECASE)
                if road_m:
                    road_name = road_m.group(0).strip()
                else:
                    continue

            # Forward geocode the road
            geo_results = self.gazetteer.forward_geocode(road_name, limit=3)
            if geo_results:
                for g in geo_results:
                    cand_road = GeographicCandidate(
                        investigation_id=investigation_id,
                        level="area",
                        name=f"Corridor: {g.display_name}",
                        lat=g.lat,
                        lon=g.lon,
                        bbox=g.bbox,
                        radius_m=15000.0,
                        score=3.5,
                        status="active",
                        seed_evidence_id=rev.id,
                        seed_evidence_kind="ocr",
                        supporting_evidence_ids=[rev.id],
                        spatial_uncertainty_m=15000.0,
                        ranking_explanation=f"OCR highway designation '{road_name}' geocoded to transportation corridor.",
                    )
                    candidates.append(cand_road)
            else:
                # Direct road corridor candidate if gazetteer query returned no results or was offline
                cand_road = GeographicCandidate(
                    investigation_id=investigation_id,
                    level="road",
                    name=f"Corridor: {road_name}",
                    lat=None,
                    lon=None,
                    score=3.0,
                    status="active",
                    seed_evidence_id=rev.id,
                    seed_evidence_kind="ocr",
                    supporting_evidence_ids=[rev.id],
                    spatial_uncertainty_m=50000.0,
                    ranking_explanation=f"OCR highway designation '{road_name}' identified from signage.",
                )
                candidates.append(cand_road)

        # -------------------------------------------------------------
        # 3. Text Clues: Street Names & Named Municipalities
        # -------------------------------------------------------------
        street_evs = [e for e in evidence_items if e.kind == "ocr" and "Category: street_name" in e.text]
        for sev in street_evs[:3]:  # Top 3 street names
            match = re.search(r'Recognized text:\s*"([^"]+)"', sev.text)
            if not match:
                continue
            st_name = match.group(1).strip()

            geo_results = self.gazetteer.forward_geocode(st_name, limit=2)
            for g in geo_results:
                cand_street = GeographicCandidate(
                    investigation_id=investigation_id,
                    level="city" if g.city else "area",
                    name=f"Street Segment: {g.display_name}",
                    lat=g.lat,
                    lon=g.lon,
                    bbox=g.bbox,
                    radius_m=8000.0,
                    score=2.8,
                    status="active",
                    seed_evidence_id=sev.id,
                    seed_evidence_kind="ocr",
                    supporting_evidence_ids=[sev.id],
                    spatial_uncertainty_m=8000.0,
                    ranking_explanation=f"OCR recognized street name '{st_name}' matched to gazetteer.",
                )
                candidates.append(cand_street)

        # -------------------------------------------------------------
        # 4. Regional Clues: License Plates & Dialing Codes
        # -------------------------------------------------------------
        plate_evs = [e for e in evidence_items if "BCplates" in e.text or "plate" in e.text.lower()]
        for pev in plate_evs:
            if "BCplates" in pev.text or "british columbia" in pev.text.lower():
                geo_results = self.gazetteer.forward_geocode("British Columbia, Canada", limit=1)
                if geo_results:
                    g = geo_results[0]
                    candidates.append(
                        GeographicCandidate(
                            investigation_id=investigation_id,
                            level="admin",
                            name=g.display_name,
                            lat=g.lat,
                            lon=g.lon,
                            bbox=g.bbox,
                            radius_m=200000.0,
                            score=3.0,
                            status="active",
                            seed_evidence_id=pev.id,
                            seed_evidence_kind="plate",
                            supporting_evidence_ids=[pev.id],
                            spatial_uncertainty_m=200000.0,
                            ranking_explanation="Visual recognition of British Columbia vehicle registration livery.",
                        )
                    )

        # -------------------------------------------------------------
        # 5. Solar Ephemeris Latitude Band Validation
        # -------------------------------------------------------------
        from investigators.shadows.solar_analyzer import SolarAnalyzer

        solar_ev = next((e for e in evidence_items if e.kind == "sun" and "Sun Elevation:" in e.text), None)
        if solar_ev:
            # Extract elevation angle e.g. "Sun Elevation: 45.2°"
            elev_match = re.search(r"Sun Elevation:\s*([-\d.]+)°", solar_ev.text)
            if elev_match:
                observed_elev = float(elev_match.group(1))
                day_of_year = None
                if image_record.exif.datetime_original:
                    day_of_year = image_record.exif.datetime_original.timetuple().tm_yday

                # For any candidate with coordinates, evaluate physical solar feasibility
                for c in candidates:
                    if c.lat is not None and c.level != "point":
                        is_feasible, max_possible_elev, reason = SolarAnalyzer.is_solar_elevation_feasible(
                            lat_deg=c.lat,
                            observed_elev_deg=observed_elev,
                            day_of_year=day_of_year,
                            tolerance_deg=3.0,
                        )
                        if not is_feasible:
                            c.status = "excluded"
                            c.contradictory_evidence_ids.append(solar_ev.id)
                            c.falsification_condition = reason

        # -------------------------------------------------------------
        # 6. Fallback / Default Regional Baseline
        # -------------------------------------------------------------
        if not candidates:
            # If no text or GPS, generate a wide unlocalized hypothesis
            cand_unresolved = GeographicCandidate(
                investigation_id=investigation_id,
                level="region",
                name="Unlocalized Visual Scene",
                lat=None,
                lon=None,
                bbox=None,
                radius_m=None,
                score=1.0,
                status="active",
                supporting_evidence_ids=[],
                spatial_uncertainty_m=10000000.0,
                ranking_explanation="Insufficient unique textual or telemetry clues to establish discrete candidate polygons.",
            )
            candidates.append(cand_unresolved)

        # Persist candidates to HOUND EvidenceStore if available
        if self.db:
            # Clean prior candidate records for this investigation session
            with self.db.get_connection() as conn:
                conn.execute("DELETE FROM candidates WHERE investigation_id = ?", (investigation_id,))
                conn.commit()

            evid_store = EvidenceStore(self.db)
            for c in candidates:
                cand_item = CandidateItem(
                    id=c.id,
                    investigation_id=c.investigation_id,
                    parent_id=c.parent_id,
                    level=c.level,
                    name=c.name,
                    lat=c.lat,
                    lon=c.lon,
                    radius_m=c.radius_m,
                    score=c.score,
                    status=c.status,
                    falsification_condition=c.falsification_condition,
                )
                evid_store.add_candidate(cand_item)

                # Link supporting evidence
                for sup_id in c.supporting_evidence_ids:
                    evid_store.link_evidence_to_candidate(
                        candidate_id=c.id,
                        evidence_id=sup_id,
                        likelihood_ratio=2.0,
                        direction="for",
                        why="Visual evidence corroborates candidate geographic designation.",
                    )

                # Link contradictory evidence
                for con_id in c.contradictory_evidence_ids:
                    evid_store.link_evidence_to_candidate(
                        candidate_id=c.id,
                        evidence_id=con_id,
                        likelihood_ratio=0.1,
                        direction="against",
                        why=c.falsification_condition or "Astronomical or spatial contradiction.",
                    )

        logger.info(f"Generated {len(candidates)} geographic candidate hypotheses for {investigation_id}")
        return candidates
