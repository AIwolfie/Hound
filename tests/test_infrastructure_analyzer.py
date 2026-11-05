"""Tests for HOUND Visual Infrastructure Investigators."""
import tempfile
import unittest
from pathlib import Path
from PIL import Image, ImageDraw

from core.image_loader import load_and_assess_image
from investigators.infrastructure.classical_detector import ClassicalInfrastructureDetector
from investigators.infrastructure.infrastructure_analyzer import InfrastructureAnalyzer
from storage.database import DatabaseManager


class TestInfrastructureAnalyzer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)
        self.db_path = self.dir_path / "test_infra.db"
        self.db_manager = DatabaseManager(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_classical_detector_on_synthetic_scene(self):
        # Create synthetic scene with vertical pole, rectangular sign, and lane marking
        img_path = self.dir_path / "scene.jpg"
        img = Image.new("RGB", (800, 600), color=(180, 200, 220))
        draw = ImageDraw.Draw(img)

        # 1. Vertical utility pole (high aspect ratio line in middle ground)
        draw.rectangle([250, 100, 258, 480], fill=(40, 40, 45))

        # 2. Rectangular road sign board
        draw.rectangle([450, 150, 560, 220], fill=(20, 80, 160))

        # 3. Yellow road lane marking in lower third
        draw.polygon([(300, 450), (320, 580), (335, 580), (315, 450)], fill=(240, 210, 20))

        img.save(img_path)

        detector = ClassicalInfrastructureDetector()
        features = detector.detect(img_path)

        self.assertGreaterEqual(len(features), 1)

        # Verify feature properties
        for f in features:
            self.assertIn(f.class_name, ["utility_pole", "road_sign_panel", "road_marking"])
            self.assertGreater(f.confidence, 0.40)
            self.assertEqual(len(f.box), 4)
            self.assertLess(f.box[0], f.box[2])
            self.assertLess(f.box[1], f.box[3])

    def test_infrastructure_analyzer_investigation_run(self):
        img_path = self.dir_path / "pole_scene.jpg"
        img = Image.new("RGB", (600, 500), color=(150, 170, 190))
        draw = ImageDraw.Draw(img)
        # Vertical pole
        draw.rectangle([200, 50, 208, 400], fill=(30, 30, 30))
        img.save(img_path)

        record = load_and_assess_image(img_path, storage_dir=self.dir_path / "storage")

        # Create investigation record in DB for foreign key constraint
        with self.db_manager.get_connection() as conn:
            conn.execute(
                "INSERT INTO investigations (id, sha256, filename, original_path, status, started_at) VALUES (?, ?, ?, ?, ?, ?)",
                ("inv_infra_test", record.sha256, record.filename, str(record.original_path), "init", "2026-11-05"),
            )
            conn.commit()

        analyzer = InfrastructureAnalyzer()
        evidence_items = analyzer.run(
            investigation_id="inv_infra_test",
            image_record=record,
            context={"db_manager": self.db_manager},
        )

        # Check DB persistence
        with self.db_manager.get_connection() as conn:
            cur = conn.execute("SELECT * FROM infrastructure_detections WHERE investigation_id = ?", ("inv_infra_test",))
            rows = cur.fetchall()
            self.assertGreaterEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
