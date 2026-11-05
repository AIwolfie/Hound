"""Unit tests for HOUND Geographic Candidate Generator."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from agent.candidate_generator import GeographicCandidateGenerator
from core.geographic_search import BoundingBox
from core.image_loader import ExifMetadata, GPSCoordinates, ImageRecord, IntegrityAssessment
from integrations.geocoding.gazetteer import GeocodingResult, GazetteerService
from storage.database import DatabaseManager
from storage.evidence_store import EvidenceItem


class TestCandidateGenerator(unittest.TestCase):
    """Test suite for GeographicCandidateGenerator evidence synthesis."""

    def setUp(self):
        self.mock_gazetteer = MagicMock(spec=GazetteerService)

    def _make_dummy_image_record(self, gps: GPSCoordinates | None = None) -> ImageRecord:
        exif = ExifMetadata(gps=gps)
        return ImageRecord(
            sha256="0" * 64,
            filename="dummy.jpg",
            original_path=Path("dummy.jpg"),
            file_size_bytes=1024,
            format="JPEG",
            width=800,
            height=600,
            aspect_ratio=800 / 600,
            mode="RGB",
            exif=exif,
            integrity=IntegrityAssessment(
                is_valid_image=True,
                status="complete",
                findings=[],
                has_gps=gps is not None,
                has_camera_info=False,
                has_timestamp=False,
                is_edited_or_exported=False,
            ),
        )

    def test_fallback_unlocalized_candidate(self):
        """Verify fallback candidate when no clues or GPS exist."""
        generator = GeographicCandidateGenerator(gazetteer=self.mock_gazetteer)
        img_rec = self._make_dummy_image_record()

        candidates = generator.generate_candidates(
            investigation_id="inv_test_001",
            image_record=img_rec,
            evidence_items=[],
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].level, "region")
        self.assertIn("Unlocalized", candidates[0].name)
        self.assertIsNone(candidates[0].lat)

    def test_direct_gps_anchor_candidate(self):
        """Verify candidate generation from embedded EXIF GPS telemetry."""
        self.mock_gazetteer.reverse_geocode.return_value = GeocodingResult(
            display_name="Mount Rainier National Park, Washington, USA",
            lat=46.8523,
            lon=-121.7603,
            bbox=BoundingBox(min_lat=46.8, min_lon=-121.8, max_lat=46.9, max_lon=-121.7),
            place_type="administrative",
            country="United States",
            state="Washington",
        )

        generator = GeographicCandidateGenerator(gazetteer=self.mock_gazetteer)
        img_rec = self._make_dummy_image_record(gps=GPSCoordinates(latitude=46.8523, longitude=-121.7603))

        gps_ev = EvidenceItem(
            investigation_id="inv_test_002",
            kind="exif",
            status="confirmed",
            text="Embedded GPS telemetry: (46.8523, -121.7603)",
        )

        candidates = generator.generate_candidates(
            investigation_id="inv_test_002",
            image_record=img_rec,
            evidence_items=[gps_ev],
        )

        self.assertEqual(len(candidates), 1)
        cand = candidates[0]
        self.assertEqual(cand.level, "point")
        self.assertAlmostEqual(cand.lat, 46.8523)
        self.assertAlmostEqual(cand.lon, -121.7603)
        self.assertEqual(cand.score, 10.0)
        self.assertIn("Mount Rainier", cand.name)

    def test_ocr_road_number_candidate(self):
        """Verify candidate corridor created from OCR road number."""
        self.mock_gazetteer.forward_geocode.return_value = [
            GeocodingResult(
                display_name="Highway 101, California, USA",
                lat=37.7749,
                lon=-122.4194,
                bbox=BoundingBox(min_lat=37.0, min_lon=-123.0, max_lat=38.0, max_lon=-122.0),
                place_type="road",
                country="United States",
            )
        ]

        generator = GeographicCandidateGenerator(gazetteer=self.mock_gazetteer)
        img_rec = self._make_dummy_image_record()

        ocr_ev = EvidenceItem(
            investigation_id="inv_test_003",
            kind="ocr",
            status="computed",
            text='[OCR Visual Reading] Recognized text: "US-101 North" (Confidence: 0.95). Category: road_number',
        )

        candidates = generator.generate_candidates(
            investigation_id="inv_test_003",
            image_record=img_rec,
            evidence_items=[ocr_ev],
        )

        self.assertEqual(len(candidates), 1)
        cand = candidates[0]
        self.assertEqual(cand.level, "area")
        self.assertIn("Highway 101", cand.name)
        self.assertEqual(cand.supporting_evidence_ids, [ocr_ev.id])

    def test_solar_elevation_latitude_exclusion(self):
        """Verify candidate is falsified/excluded if observed sun elevation is astronomically impossible."""
        # High latitude candidate (e.g., Iceland at 65°N)
        self.mock_gazetteer.forward_geocode.return_value = [
            GeocodingResult(
                display_name="Reykjavik Route 1, Iceland",
                lat=65.0,
                lon=-18.0,
                bbox=BoundingBox(min_lat=64.0, min_lon=-19.0, max_lat=66.0, max_lon=-17.0),
                place_type="road",
                country="Iceland",
            )
        ]

        generator = GeographicCandidateGenerator(gazetteer=self.mock_gazetteer)
        img_rec = self._make_dummy_image_record()

        ocr_ev = EvidenceItem(
            investigation_id="inv_test_004",
            kind="ocr",
            status="computed",
            text='[OCR Visual Reading] Recognized text: "Route 1" (Confidence: 0.90). Category: road_number',
        )

        # Sun elevation of 60° is impossible at 65°N (max possible is ~90 - |65 - 23.5| = 48.5°)
        solar_ev = EvidenceItem(
            investigation_id="inv_test_004",
            kind="sun",
            status="computed",
            text="[Solar Geometry] Sun Elevation: 62.0°, Sun Azimuth: 180.0°",
        )

        candidates = generator.generate_candidates(
            investigation_id="inv_test_004",
            image_record=img_rec,
            evidence_items=[ocr_ev, solar_ev],
        )

        self.assertEqual(len(candidates), 1)
        cand = candidates[0]
        self.assertEqual(cand.status, "excluded")
        self.assertIn(solar_ev.id, cand.contradictory_evidence_ids)
        self.assertIsNotNone(cand.falsification_condition)
        self.assertIn("exceeds maximum possible", cand.falsification_condition)


if __name__ == "__main__":
    unittest.main()
