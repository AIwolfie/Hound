"""Tests for HOUND Metadata Analyzer."""
import tempfile
import unittest
from pathlib import Path
from PIL import Image, ExifTags

from core.image_loader import load_and_assess_image
from investigators.metadata.exif_analyzer import ExifAnalyzer


class TestMetadataAnalyzer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)
        self.analyzer = ExifAnalyzer()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_analyzer_on_full_exif_image(self):
        img_path = self.dir_path / "camera_shot.jpg"
        img = Image.new("RGB", (1000, 750), color="green")

        exif = img.getexif()
        exif[ExifTags.Base.Make] = "Nikon"
        exif[ExifTags.Base.Model] = "Z9"
        exif[ExifTags.Base.DateTime] = "2025:10:05 11:20:00"

        # SubIFD optics
        sub_ifd = exif.get_ifd(ExifTags.IFD.Exif)
        sub_ifd[ExifTags.Base.FocalLengthIn35mmFilm] = 50

        # GPS
        gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
        gps_ifd[1] = "N"
        gps_ifd[2] = (35, 41, 22.0)
        gps_ifd[3] = "E"
        gps_ifd[4] = (139, 41, 30.0)
        img.save(img_path, format="JPEG", exif=exif)

        record = load_and_assess_image(img_path, storage_dir=self.dir_path)
        items = self.analyzer.run(investigation_id="inv_test", image_record=record)

        self.assertGreater(len(items), 3)

        # Check for GPS item
        gps_items = [i for i in items if i.kind == "exif" and "Embedded GPS" in i.text]
        self.assertEqual(len(gps_items), 1)
        self.assertIn("35.689", gps_items[0].text)

        # Check for Optics & FOV item
        fov_items = [i for i in items if "horizontal FOV" in i.text]
        self.assertEqual(len(fov_items), 1)
        # 50mm lens on 36mm sensor: 2 * atan(36 / 100) = 39.6 degrees
        self.assertIn("39.6°", fov_items[0].text)

        # Check for Temporal item
        time_items = [i for i in items if "October" in i.text]
        self.assertEqual(len(time_items), 1)


if __name__ == "__main__":
    unittest.main()
