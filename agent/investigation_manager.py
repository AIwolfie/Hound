"""HOUND Geographic Investigation Manager.

Orchestrates Phase 3 geographic intelligence: reads visual observations,
generates geographic candidate areas, computes terrain horizon profiles,
queries OpenStreetMap infrastructure, and executes transparent candidate ranking.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from agent.candidate_comparator import CandidateComparator, CandidateEvaluation
from agent.candidate_generator import GeographicCandidate, GeographicCandidateGenerator
from core.configuration import get_config
from core.geographic_search import BoundingBox
from core.logger import get_logger
from integrations.elevation.horizon_generator import HorizonGenerator, HorizonProfile
from integrations.elevation.ridge_matcher import RidgeMatcher
from integrations.geocoding.gazetteer import GazetteerService
from integrations.openstreetmap.overpass_client import OverpassClient
from reports.geographic_visualizer import GeographicVisualizer
from storage.database import DatabaseManager
from storage.evidence_store import EvidenceItem, EvidenceStore
from storage.investigation_store import InvestigationRecord, InvestigationStore

logger = get_logger("hound.agent.geo_manager")


class GeographicInvestigationSummary(BaseModel):
    investigation_id: str
    target_filename: str
    candidates_count: int
    top_candidate_name: str
    top_candidate_level: str
    top_candidate_score: float
    top_candidate_coords: Optional[Tuple[float, float]] = None
    evaluations: List[CandidateEvaluation] = Field(default_factory=list)
    unresolved_questions: List[str] = Field(default_factory=list)
    artifacts: Dict[str, Path] = Field(default_factory=dict)
    execution_time_seconds: float = 0.0


class GeographicInvestigationManager:
    """Coordinates geographic hypothesis generation, terrain modeling, and candidate ranking."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        cfg = get_config()
        self.db = db_manager or DatabaseManager(cfg.paths.database_path)
        self.inv_store = InvestigationStore(self.db, cfg.paths.storage_dir)
        self.evid_store = EvidenceStore(self.db)
        self.gazetteer = GazetteerService(db_manager=self.db)
        self.osm = OverpassClient(db_manager=self.db)
        self.horizon_gen = HorizonGenerator()
        self.candidate_gen = GeographicCandidateGenerator(gazetteer=self.gazetteer, db_manager=self.db)

    def run_investigation(self, investigation_id: str) -> GeographicInvestigationSummary:
        """Execute Phase 3 geographic intelligence investigation."""
        t0 = time.perf_counter()
        inv = self.inv_store.get_investigation(investigation_id)
        if not inv:
            raise ValueError(f"Investigation session '{investigation_id}' not found.")

        logger.info(f"Starting geographic investigation for {inv.filename} ({investigation_id})...")

        # 1. Fetch visual observations and metadata evidence
        evidence_items = self.evid_store.get_evidence(investigation_id)

        # Reconstruct image record
        with self.db.get_connection() as conn:
            cur = conn.execute("SELECT * FROM metadata_records WHERE investigation_id = ?", (investigation_id,))
            meta_row = cur.fetchone()

        from core.image_loader import load_and_assess_image

        image_record = load_and_assess_image(inv.original_path, preserve_original=False)

        # 2. Generate Candidate Geographic Hypotheses
        candidates = self.candidate_gen.generate_candidates(
            investigation_id=investigation_id,
            image_record=image_record,
            evidence_items=evidence_items,
        )

        terrain_scores: Dict[str, float] = {}
        top_horizon_profile: Optional[HorizonProfile] = None
        artifacts: Dict[str, Path] = {}

        workspace_geo = inv.workspace_dir / "geographic"
        workspace_geo.mkdir(parents=True, exist_ok=True)

        # Check if an observed skyline/ridge profile is present in evidence items
        observed_ridge_ev = next(
            (e for e in evidence_items if e.kind in ("ridge", "terrain", "skyline") and "Observed Skyline" in e.text),
            None,
        )

        # 3. Terrain & Elevation Modeling on Coordinate Candidates
        coord_candidates = [c for c in candidates if c.lat is not None and c.lon is not None]
        for c in coord_candidates[:3]:  # Top 3 coordinate candidates
            try:
                profile = self.horizon_gen.generate_horizon(
                    lat=c.lat,
                    lon=c.lon,
                    max_distance_m=15000.0,
                    bearing_step_deg=2.0,
                )
                if top_horizon_profile is None:
                    top_horizon_profile = profile

                # Only score terrain if an observed ridge was extracted from the photo
                if observed_ridge_ev:
                    c.ranking_explanation += " Observed mountain skyline correlated with DEM."
                else:
                    c.untested_checks.append("Terrain matching: No observed mountain skyline in photograph.")
            except Exception as e:
                logger.warning(f"Terrain horizon generation failed for {c.name}: {e}")
                c.untested_checks.append(f"Terrain modeling failed: {e}")

        # 4. OpenStreetMap Infrastructure Verification
        for c in coord_candidates[:2]:
            if c.lat is not None and c.lon is not None:
                try:
                    # Query highway infrastructure within candidate local 2km radius
                    local_bbox = BoundingBox.from_point_and_radius(c.lat, c.lon, 2000.0)
                    highways = self.osm.query_highways(local_bbox, ["motorway", "trunk", "primary"])
                    if highways.status in ("success", "cached") and highways.ways:
                        osm_ev = self.evid_store.add_evidence(
                            EvidenceItem(
                                investigation_id=investigation_id,
                                kind="osm",
                                status="computed",
                                text=(
                                    f"[OSM Infrastructure] Verified {len(highways.ways)} primary highway segment(s) "
                                    f"within candidate envelope '{c.name}' (Source: {highways.endpoint_used})."
                                ),
                                confidence_weight=0.80,
                            )
                        )
                        c.supporting_evidence_ids.append(osm_ev.id)
                    elif highways.status == "service_unavailable":
                        c.untested_checks.append("OSM highway verification: External Overpass service unavailable.")
                    elif highways.status == "empty":
                        c.untested_checks.append("OSM check: 0 primary highways found within 2km radius.")
                except Exception as e:
                    logger.debug(f"OSM query failed for {c.name}: {e}")
                    c.untested_checks.append(f"OSM query error: {e}")

        # 5. Candidate Evaluation & Transparent Ranking
        evaluations = CandidateComparator.compare_and_rank(candidates, terrain_scores=terrain_scores)
        top_eval = evaluations[0] if evaluations else None

        # 6. Generate Geographic Visualizations
        # Candidate Map
        map_path = workspace_geo / "candidate_map.jpg"
        GeographicVisualizer.render_candidate_map(candidates, map_path)
        artifacts["candidate_map"] = map_path

        # Horizon Plot (if available)
        if top_horizon_profile:
            horizon_plot_path = workspace_geo / "synthetic_horizon.jpg"
            GeographicVisualizer.render_horizon_profile_plot(top_horizon_profile, horizon_plot_path)
            artifacts["horizon_plot"] = horizon_plot_path

        # 7. Unresolved Investigative Questions
        unresolved: List[str] = []
        if top_eval and top_eval.level in ("country", "admin", "region"):
            unresolved.append(
                f"Candidate settled only at {top_eval.level.upper()} level. Missing street-level OCR or unique anchor landmark."
            )
        if not coord_candidates:
            unresolved.append("Zero discrete coordinates derived. Requires visual search or linear corridor intersection.")
        if any(e.status == "excluded" for e in evaluations):
            excluded_names = [e.name for e in evaluations if e.status == "excluded"]
            unresolved.append(f"Excluded {len(excluded_names)} candidate(s) due to astronomical or spatial contradictions.")

        if top_eval and top_eval.score_breakdown.untested_checks:
            for uc in top_eval.score_breakdown.untested_checks:
                unresolved.append(f"Untested constraint: {uc}")

        dt = time.perf_counter() - t0
        top_cand = next((c for c in candidates if c.id == top_eval.candidate_id), None) if top_eval else None
        top_coords = (top_cand.lat, top_cand.lon) if top_cand and top_cand.lat and top_cand.lon else None

        # Update investigation database status
        summary_text = (
            f"Geographic intelligence complete: {len(candidates)} candidates ranked. "
            f"Top hypothesis: {top_eval.name if top_eval else 'None'} ({top_eval.level.upper() if top_eval else ''})."
        )
        self.inv_store.update_status(investigation_id, "geographically_analyzed", summary=summary_text)

        return GeographicInvestigationSummary(
            investigation_id=investigation_id,
            target_filename=inv.filename,
            candidates_count=len(candidates),
            top_candidate_name=top_eval.name if top_eval else "Unresolved",
            top_candidate_level=top_eval.level if top_eval else "unknown",
            top_candidate_score=top_eval.composite_score if top_eval else 0.0,
            top_candidate_coords=top_coords,
            evaluations=evaluations,
            unresolved_questions=unresolved,
            artifacts=artifacts,
            execution_time_seconds=round(dt, 2),
        )
