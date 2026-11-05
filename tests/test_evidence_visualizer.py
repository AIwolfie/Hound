"""Tests for HOUND Evidence Visualizer."""
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from core.image_preprocessor import VariantInfo
from investigators.infrastructure.base_detector import DetectedFeature
from investigators.ocr.ocr_engine import OCRObservation
from investigators.shadows.solar_analyzer import ShadowCandidate
from reports.evidence_visualizer import EvidenceVisualizer


class TestEvidenceVisualizer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_ocr_overlay_generation(self):
        img_path = self.dir_path / "base.jpg"
        img = Image.new("RGB", (400, 300), color=(200, 200, 200))
        img.save(img_path)

        obs = [
            OCRObservation(
                investigation_id="test_inv",
                text="MAIN ST",
                confidence=0.92,
                box_original=(50.0, 50.0, 180.0, 80.0),
                category="street_name",
                language_script="Latin",
                pass_name="full_normalized",
            )
        ]
        out_path = self.dir_path / "ocr_overlay.jpg"
        result_path = EvidenceVisualizer.draw_ocr_overlay(img_path, obs, out_path)

        self.assertTrue(result_path.is_file())
        with Image.open(result_path) as out_img:
            self.assertEqual(out_img.size, (400, 300))

    def test_infrastructure_overlay_generation(self):
        img_path = self.dir_path / "base_infra.jpg"
        img = Image.new("RGB", (400, 300), color=(100, 120, 140))
        img.save(img_path)

        feats = [
            DetectedFeature(
                class_name="utility_pole",
                confidence=0.88,
                box=(100.0, 20.0, 115.0, 280.0),
                detector_backend="test",
            )
        ]
        out_path = self.dir_path / "infra_overlay.jpg"
        result_path = EvidenceVisualizer.draw_infrastructure_overlay(img_path, feats, out_path)

        self.assertTrue(result_path.is_file())

    def test_shadow_overlay_generation(self):
        img_path = self.dir_path / "base_shadow.jpg"
        img = Image.new("RGB", (400, 300), color=(180, 180, 180))
        img.save(img_path)

        shadows = [
            ShadowCandidate(
                contour_id=1,
                center_point=(150.0, 150.0),
                area_px=500.0,
                length_px=100.0,
                width_px=20.0,
                aspect_ratio=5.0,
                orientation_deg=45.0,
                confidence=0.75,
                notes="test shadow",
            )
        ]
        out_path = self.dir_path / "shadow_overlay.jpg"
        result_path = EvidenceVisualizer.draw_shadow_overlay(img_path, shadows, out_path)

        self.assertTrue(result_path.is_file())

    def test_comparison_panel_generation(self):
        # Create 2 small variant files
        var1_path = self.dir_path / "var1.jpg"
        var2_path = self.dir_path / "var2.jpg"
        Image.new("RGB", (320, 240), color="blue").save(var1_path)
        Image.new("RGB", (320, 240), color="gray").save(var2_path)

        variants = [
            VariantInfo(name="full_normalized", dimensions=(320, 240), file_path=var1_path, description="Orig"),
            VariantInfo(name="grayscale", dimensions=(320, 240), file_path=var2_path, description="Gray"),
        ]

        out_path = self.dir_path / "comparison.jpg"
        panel_path = EvidenceVisualizer.generate_comparison_panel(variants, out_path)

        self.assertTrue(panel_path.is_file())
        with Image.open(panel_path) as panel_img:
            self.assertGreater(panel_img.width, 600)
            self.assertGreater(panel_img.height, 400)


if __name__ == "__main__":
    unittest.main()
