"""Tests for HOUND Image Preprocessor."""
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from core.image_preprocessor import ImagePreprocessor


class TestImagePreprocessor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)
        self.preprocessor = ImagePreprocessor(
            max_dimension=1024,
            tile_grid=(2, 2),
            tile_overlap_ratio=0.15,
            enable_upscaling=True,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_full_preprocessing_pipeline(self):
        # Create a sample test image
        img_path = self.dir_path / "sample_target.jpg"
        img = Image.new("RGB", (600, 400), color=(120, 150, 180))
        img.save(img_path, format="JPEG")

        out_dir = self.dir_path / "preprocessed_out"
        manifest = self.preprocessor.process(img_path, out_dir, generate_tiles=True, generate_edges=True)

        self.assertEqual(manifest.source_image, "sample_target.jpg")
        self.assertEqual(manifest.original_dimensions, (600, 400))
        self.assertEqual(manifest.normalized_dimensions, (600, 400))
        self.assertGreater(manifest.processing_time_seconds, 0.0)

        # Verify analysis variants
        variant_names = {v.name for v in manifest.analysis_variants}
        self.assertIn("full_normalized", variant_names)
        self.assertIn("contrast_enhanced", variant_names)
        self.assertIn("grayscale", variant_names)
        self.assertIn("clahe", variant_names)
        self.assertIn("edge_enhanced", variant_names)
        self.assertIn("sharpened", variant_names)
        self.assertIn("upscaled_2x", variant_names)

        # Check all variant files exist
        for v in manifest.analysis_variants:
            self.assertTrue(v.file_path.is_file(), f"Variant file missing: {v.file_path}")

        # Verify edge and tile crops
        self.assertGreater(len(manifest.crops), 8)
        crop_names = {c.name for c in manifest.crops}
        self.assertIn("edge_north", crop_names)
        self.assertIn("corner_nw", crop_names)
        self.assertIn("tile_r0_c0", crop_names)

        # Verify coordinate bounds
        for crop in manifest.crops:
            x0, y0, x1, y1 = crop.box_original
            self.assertGreaterEqual(x0, 0)
            self.assertGreaterEqual(y0, 0)
            self.assertLessEqual(x1, 600)
            self.assertLessEqual(y1, 400)
            self.assertGreater(x1, x0)
            self.assertGreater(y1, y0)
            self.assertTrue(crop.file_path.is_file())

        # Verify manifest JSON file on disk
        manifest_path = out_dir / "preprocessing_manifest.json"
        self.assertTrue(manifest_path.is_file())
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(data["source_image"], "sample_target.jpg")


if __name__ == "__main__":
    unittest.main()
