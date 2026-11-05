"""Tests for HOUND OCR Engine, Coordinate Mapping, and Clue Classification."""
import tempfile
import unittest
from pathlib import Path
from PIL import Image, ImageDraw

from core.image_loader import load_and_assess_image
from investigators.ocr.ocr_engine import (
    BaseOCRBackend,
    OCREngine,
    RawOCRBox,
    classify_geographic_clue,
    compute_iou,
    detect_script,
)
from storage.database import DatabaseManager


class MockOCRBackend(BaseOCRBackend):
    """Predictable mock backend for unit testing coordinate remapping and deduplication."""

    def __init__(self, detections: list[RawOCRBox]):
        self.detections = detections

    def detect_text(self, image_input):
        return self.detections


class TestOCREngine(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)
        self.db_path = self.dir_path / "test_ocr.db"
        self.db_manager = DatabaseManager(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_iou_computation(self):
        box1 = (0.0, 0.0, 10.0, 10.0)
        box2 = (0.0, 0.0, 10.0, 10.0)
        self.assertEqual(compute_iou(box1, box2), 1.0)

        box3 = (10.0, 10.0, 20.0, 20.0)
        self.assertEqual(compute_iou(box1, box3), 0.0)

        box4 = (5.0, 0.0, 15.0, 10.0)
        # Inter = 5 * 10 = 50. Union = 100 + 100 - 50 = 150. IoU = 50 / 150 = 0.3333
        self.assertAlmostEqual(compute_iou(box1, box4), 0.3333, places=3)

    def test_script_detection(self):
        self.assertEqual(detect_script("Main Street 101"), "Latin")
        self.assertEqual(detect_script("Красная площадь"), "Cyrillic")
        self.assertEqual(detect_script("北京市朝阳区"), "Han")

    def test_geographic_clue_classification(self):
        self.assertEqual(classify_geographic_clue("I-80"), "road_number")
        self.assertEqual(classify_geographic_clue("US-101"), "road_number")
        self.assertEqual(classify_geographic_clue("NH 48"), "road_number")
        self.assertEqual(classify_geographic_clue("Broadway St"), "street_name")
        self.assertEqual(classify_geographic_clue("Grand Avenue"), "street_name")
        self.assertEqual(classify_geographic_clue("Müllerstrasse"), "street_name")
        self.assertEqual(classify_geographic_clue("Central Station"), "railway_station")
        self.assertEqual(classify_geographic_clue("Tokyo Metro"), "transport_authority")
        self.assertEqual(classify_geographic_clue("+1 (555) 234-5678"), "telephone_number")
        self.assertEqual(classify_geographic_clue("90210"), "postal_code")
        self.assertEqual(classify_geographic_clue("Random Text 123"), "unknown")

    def test_ocr_investigation_and_persistence(self):
        # Create an image with simple drawn text
        img_path = self.dir_path / "sign_test.jpg"
        img = Image.new("RGB", (400, 300), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), "US-101 North", fill="black")
        img.save(img_path)

        record = load_and_assess_image(img_path, storage_dir=self.dir_path / "storage")

        # Mock detections
        mock_box = RawOCRBox(
            text="US-101 North",
            confidence=0.95,
            box=(50.0, 50.0, 180.0, 75.0),
            pass_name="full_normalized",
            backend="Mock",
        )
        mock_backend = MockOCRBackend([mock_box])
        engine = OCREngine(backend=mock_backend)

        # Ensure investigation exists in DB
        with self.db_manager.get_connection() as conn:
            conn.execute(
                "INSERT INTO investigations (id, sha256, filename, original_path, status, started_at) VALUES (?, ?, ?, ?, ?, ?)",
                ("inv_ocr_test", record.sha256, record.filename, str(record.original_path), "init", "2026-11-05"),
            )
            conn.commit()

        items = engine.run(
            investigation_id="inv_ocr_test",
            image_record=record,
            context={"workspace_dir": self.dir_path / "work", "db_manager": self.db_manager},
        )

        self.assertGreaterEqual(len(items), 1)
        self.assertEqual(items[0].kind, "ocr")
        self.assertEqual(items[0].status, "read")
        self.assertIn("US-101 North", items[0].text)

        # Check DB persistence
        with self.db_manager.get_connection() as conn:
            cur = conn.execute("SELECT * FROM ocr_observations WHERE investigation_id = ?", ("inv_ocr_test",))
            rows = cur.fetchall()
            self.assertGreaterEqual(len(rows), 1)
            self.assertEqual(rows[0]["text"], "US-101 North")
            self.assertEqual(rows[0]["category"], "road_number")


if __name__ == "__main__":
    unittest.main()
