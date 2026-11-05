"""Unit tests for HOUND Geographic Investigation Manager."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image

from agent.investigation_manager import GeographicInvestigationManager
from core.configuration import get_config
from core.geographic_search import BoundingBox
from integrations.geocoding.gazetteer import GeocodingResult
from storage.database import DatabaseManager
from storage.evidence_store import EvidenceItem, EvidenceStore
from storage.investigation_store import InvestigationStore


class TestGeographicInvestigationManager(unittest.TestCase):
    """Test suite for Phase 3 GeographicInvestigationManager orchestration."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        self.db_path = self.base_path / "hound_test.db"
        self.db = DatabaseManager(self.db_path)
        self.inv_store = InvestigationStore(self.db, self.base_path / "storage")
        self.evid_store = EvidenceStore(self.db)

        # Create a sample test image
        self.img_path = self.base_path / "mountain_road.jpg"
        img = Image.new("RGB", (640, 480), color=(80, 100, 120))
        img.save(self.img_path, format="JPEG")

        from core.image_loader import load_and_assess_image
        self.image_record = load_and_assess_image(self.img_path, storage_dir=self.base_path / "storage")
        self.inv = self.inv_store.create_investigation(self.image_record)

        # Insert metadata record into DB
        with self.db.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO metadata_records (
                    investigation_id, format, width, height, has_exif,
                    integrity_status, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.inv.id,
                    "JPEG",
                    640,
                    480,
                    0,
                    "complete",
                    "{}",
                ),
            )

        # Add OCR evidence item
        self.ocr_ev = self.evid_store.add_evidence(
            EvidenceItem(
                investigation_id=self.inv.id,
                kind="ocr",
                status="computed",
                text='[OCR Visual Reading] Recognized text: "Highway 99 North" (Confidence: 0.94). Category: road_number',
                confidence_weight=0.94,
            )
        )

        self.manager = GeographicInvestigationManager(db_manager=self.db)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_run_investigation_orchestration(self):
        """Verify full geographic intelligence pipeline executes end-to-end."""
        # Mock gazetteer forward geocode so test doesn't rely on live Nominatim network
        mock_gazetteer_result = GeocodingResult(
            display_name="BC-99, British Columbia, Canada",
            lat=50.1163,
            lon=-122.9574,
            bbox=BoundingBox(min_lat=49.5, min_lon=-123.5, max_lat=50.5, max_lon=-122.5),
            place_type="road",
            country="Canada",
            state="British Columbia",
        )

        mock_horizon = MagicMock()
        mock_horizon.peak_elevation_deg = 3.5
        mock_horizon.peak_azimuth_deg = 180.0
        mock_horizon.elevation_angles_deg = [0.0] * 180
        mock_horizon.bearings_deg = [float(b) for b in range(180)]
        mock_horizon.center_coord = (50.1163, -122.9574)
        mock_horizon.observer_elevation_m = 650.0

        with patch.object(self.manager.gazetteer, "forward_geocode", return_value=[mock_gazetteer_result]), \
             patch.object(self.manager.horizon_gen, "generate_horizon", return_value=mock_horizon), \
             patch.object(self.manager.osm, "query_highways") as mock_osm, \
             patch("reports.geographic_visualizer.SatelliteTileService.build_mosaic", return_value=None):
            mock_osm.return_value.ways = {"w1": MagicMock()}
            summary = self.manager.run_investigation(self.inv.id)

            self.assertEqual(summary.investigation_id, self.inv.id)
            self.assertGreaterEqual(summary.candidates_count, 1)
            self.assertIn("BC-99", summary.top_candidate_name)
            self.assertEqual(summary.top_candidate_level, "area")
            self.assertGreater(summary.top_candidate_score, 0.0)
            self.assertIsNotNone(summary.top_candidate_coords)

            # Check generated artifacts
            self.assertIn("candidate_map", summary.artifacts)
            map_file = summary.artifacts["candidate_map"]
            self.assertTrue(map_file.is_file())

            # Check updated database investigation status
            updated_inv = self.inv_store.get_investigation(self.inv.id)
            self.assertIsNotNone(updated_inv)
            self.assertEqual(updated_inv.status, "geographically_analyzed")


if __name__ == "__main__":
    unittest.main()
