"""Tests for HOUND Database, Evidence Store, and Investigation Store."""
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from core.image_loader import load_and_assess_image
from storage.database import DatabaseManager
from storage.evidence_store import CandidateItem, EvidenceItem, EvidenceStore
from storage.investigation_store import InvestigationStore


class TestDatabaseAndStores(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)
        self.db_path = self.dir_path / "test_hound.db"
        self.db_manager = DatabaseManager(self.db_path)
        self.inv_store = InvestigationStore(self.db_manager, self.dir_path / "storage")
        self.evid_store = EvidenceStore(self.db_manager)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_investigation_lifecycle(self):
        # Create test image
        img_path = self.dir_path / "target.jpg"
        img = Image.new("RGB", (320, 240), color="blue")
        img.save(img_path)

        record = load_and_assess_image(img_path, storage_dir=self.dir_path / "storage")
        inv = self.inv_store.create_investigation(record)

        self.assertIsNotNone(inv.id)
        self.assertEqual(inv.status, "initialized")
        self.assertTrue(inv.workspace_dir.is_dir())

        # Fetch back
        fetched = self.inv_store.get_investigation(inv.id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.sha256, record.sha256)

        # Update status
        self.inv_store.update_status(inv.id, "completed", summary="Test geolocation completed.")
        updated = self.inv_store.get_investigation(inv.id)
        self.assertEqual(updated.status, "completed")
        self.assertEqual(updated.summary, "Test geolocation completed.")
        self.assertIsNotNone(updated.completed_at)

    def test_evidence_and_candidate_scoring(self):
        # Create a valid investigation first for foreign key integrity
        img_path = self.dir_path / "dummy.jpg"
        img = Image.new("RGB", (100, 100), color="white")
        img.save(img_path)
        record = load_and_assess_image(img_path, storage_dir=self.dir_path / "storage")
        inv = self.inv_store.create_investigation(record)
        inv_id = inv.id

        # Add evidence item
        ev = EvidenceItem(
            investigation_id=inv_id,
            kind="ocr",
            status="read",
            text="Highway 101 North exit sign",
            confidence_weight=1.0,
        )
        saved_ev = self.evid_store.add_evidence(ev)
        self.assertEqual(saved_ev.id, ev.id)

        # Add Candidate
        cand = CandidateItem(
            investigation_id=inv_id,
            level="city",
            name="San Francisco",
            lat=37.7749,
            lon=-122.4194,
            radius_m=10000.0,
            score=1.0,
        )
        saved_cand = self.evid_store.add_candidate(cand)
        self.assertEqual(saved_cand.score, 1.0)

        # Link evidence with likelihood ratio 2.5
        self.evid_store.link_evidence_to_candidate(
            candidate_id=saved_cand.id,
            evidence_id=saved_ev.id,
            likelihood_ratio=2.5,
            direction="for",
            why="Highway 101 runs through San Francisco",
        )

        candidates = self.evid_store.get_candidates(inv_id)
        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0].score, 2.5)


if __name__ == "__main__":
    unittest.main()
