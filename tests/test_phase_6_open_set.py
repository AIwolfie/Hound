"""Unit and integration tests for HOUND Phase 6 Open-Set Geolocation & Abstention."""
from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from agent.investigation_agent import InvestigationAgent
from core.calibration_engine import CalibrationEngine, ConfidenceTier
from core.configuration import AgentConfig
from core.image_loader import load_and_assess_image
from core.spatial_envelope import GeographicResolutionLevel, SpatialEnvelope
from storage.database import DatabaseManager
from storage.evidence_store import EvidenceItem, EvidenceStore
from storage.investigation_store import InvestigationStore


def test_open_set_unfamiliar_location_honestly_abstains():
    """Verify that an unfamiliar location absent from visual index and gazetteer triggers honest abstention."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        db = DatabaseManager(workspace / "open_set.db")
        inv_store = InvestigationStore(db, workspace / "storage")
        evid_store = EvidenceStore(db)

        # Create dummy image without EXIF
        from PIL import Image
        img_path = workspace / "unfamiliar_arctic.jpg"
        img = Image.new("RGB", (640, 480), color=(180, 200, 220))
        img.save(str(img_path))

        rec = load_and_assess_image(img_path, preserve_original=False)
        inv = inv_store.create_investigation(rec)

        # Seed open-set observation (Tromso Arctic Port - unindexed)
        evid_store.add_evidence(
            EvidenceItem(
                investigation_id=inv.id,
                kind="terrain",
                status="observed",
                text="[Terrain] Sub-arctic fjord with snow peaks and commercial fishing vessels.",
                confidence_weight=0.75,
            )
        )

        config = AgentConfig(
            max_iterations=6,
            max_tool_calls=10,
            offline_mode=True,  # No live internet geocoding
        )
        agent = InvestigationAgent(db_manager=db, config=config)
        dossier = agent.investigate(investigation_id=inv.id, max_iterations=6, offline=True)

        cands = evid_store.get_candidates(inv.id)
        # In strict offline mode without online gazetteer or matching prototype,
        # agent must NOT fabricate pinpoint coordinates
        for c in cands:
            if c.lat is not None and c.lon is not None:
                # If any candidate exists, it must not be falsely marked definitive
                env = SpatialEnvelope.from_point_and_radius(c.lat, c.lon, radius_m=c.radius_m or 50000.0, area_name=c.name)
                est = CalibrationEngine.evaluate_candidate(
                    candidate_id=c.id,
                    name=c.name,
                    raw_score=c.score,
                    spatial_envelope=env,
                    all_evidence=evid_store.get_evidence(inv.id),
                )
                assert est.confidence_tier in (ConfidenceTier.TENTATIVE, ConfidenceTier.INSUFFICIENT)
                assert est.calibrated_probability < 0.70


def test_open_set_negative_controls_abstain_with_zero_false_positives():
    """Verify that featureless negative controls achieve 100% honest abstention."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        db = DatabaseManager(workspace / "open_set_neg.db")
        inv_store = InvestigationStore(db, workspace / "storage")
        evid_store = EvidenceStore(db)

        from PIL import Image
        img_path = workspace / "featureless_wall.jpg"
        img = Image.new("RGB", (640, 480), color=(200, 200, 200))
        img.save(str(img_path))

        rec = load_and_assess_image(img_path, preserve_original=False)
        inv = inv_store.create_investigation(rec)

        config = AgentConfig(max_iterations=4, offline_mode=True)
        agent = InvestigationAgent(db_manager=db, config=config)
        dossier = agent.investigate(investigation_id=inv.id, max_iterations=4, offline=True)

        # Agent must terminate with insufficient_evidence
        assert dossier.final_status == "insufficient_evidence"
        assert dossier.geographic_precision_justified in ("Unlocalized", "unlocalized")
        assert len(dossier.abstention_reasons) > 0


def test_open_set_calibration_flags_insufficient_evidence():
    """Verify that unlocalized spatial envelopes always recommend abstention."""
    envelope = SpatialEnvelope.unlocalized("Ambiguous ocean vista without identifiable landmarks.")
    est = CalibrationEngine.evaluate_candidate(
        candidate_id="unlocalized_test",
        name="Unlocalized Ocean",
        raw_score=10.0,
        spatial_envelope=envelope,
        all_evidence=[],
    )

    assert est.resolution_level == GeographicResolutionLevel.UNLOCALIZED
    assert est.abstention_recommended is True
    assert "No definitive geographic anchors" in est.abstention_reason
