"""Tests for HOUND Phase 6 real-world photographic intake and pipeline execution."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
import pytest

from agent.investigation_agent import InvestigationAgent
from core.configuration import AgentConfig
from core.image_loader import load_and_assess_image
from storage.database import DatabaseManager
from storage.investigation_store import InvestigationStore

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
PHASE6_DIR = WORKSPACE_DIR / "data" / "benchmarks" / "phase_6"
IMAGES_DIR = PHASE6_DIR / "images"
MANIFEST_FILE = PHASE6_DIR / "metadata" / "manifest.json"


def test_manifest_matches_physical_photographs():
    """Verify that all manifest entries correspond to physical photograph files on disk with matching SHA-256."""
    if not MANIFEST_FILE.exists() or not IMAGES_DIR.exists():
        pytest.skip("Phase 6 dataset not curated.")

    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert len(manifest) >= 12

    for entry in manifest:
        cid = entry["case_id"]
        img_path = IMAGES_DIR / f"{cid}.jpg"
        assert img_path.exists(), f"Missing physical image file for {cid}"

        # Verify hash integrity
        data = img_path.read_bytes()
        computed_sha = hashlib.sha256(data).hexdigest()
        assert computed_sha == entry["sha256"], f"SHA256 mismatch on {cid}: expected {entry['sha256']}, got {computed_sha}"


def test_real_photograph_forensic_intake():
    """Verify that a genuine camera photograph ingests cleanly through forensic intake."""
    if not IMAGES_DIR.exists():
        pytest.skip("Phase 6 dataset not curated.")

    real_img = IMAGES_DIR / "REAL-01.jpg"
    assert real_img.exists()

    rec = load_and_assess_image(real_img, preserve_original=False)
    assert rec.format == "JPEG"
    assert rec.width > 500
    assert rec.height > 500
    assert rec.exif.gps is None  # Confirms EXIF GPS was stripped
    assert rec.integrity.status in ("verified", "warning", "unverified", "absent")


def test_real_photograph_agent_lifecycle():
    """Verify that the autonomous investigation agent executes an end-to-end investigation on a real photograph."""
    if not IMAGES_DIR.exists():
        pytest.skip("Phase 6 dataset not curated.")

    real_img = IMAGES_DIR / "REAL-11.jpg"  # Meeting room negative control
    assert real_img.exists()

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        db = DatabaseManager(workspace / "real_test.db")
        inv_store = InvestigationStore(db, workspace / "storage")

        rec = load_and_assess_image(real_img, preserve_original=False)
        inv = inv_store.create_investigation(rec)

        config = AgentConfig(max_iterations=4, offline_mode=True)
        agent = InvestigationAgent(db_manager=db, config=config)
        dossier = agent.investigate(investigation_id=inv.id, max_iterations=4, offline=True)

        assert dossier.investigation_id == inv.id
        assert dossier.total_iterations >= 1
        assert dossier.final_status in ("insufficient_evidence", "candidates_remaining")
