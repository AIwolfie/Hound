"""Tests for HOUND Image Loader & Forensic Metadata Extraction."""
import tempfile
import unittest
from pathlib import Path
from PIL import Image, ExifTags

from core.image_loader import (
    GPSCoordinates,
    _convert_dms_to_decimal,
    load_and_assess_image,
)


class TestImageLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_dms_to_decimal_conversion(self):
        # 48° 51' 29.88" N = 48.8583
        val = (48, 51, 29.88)
        dec_n = _convert_dms_to_decimal(val, "N")
        self.assertAlmostEqual(dec_n, 48.8583, places=3)

        # 2° 17' 40.2" W = -2.2945
        val_w = (2, 17, 40.2)
        dec_w = _convert_dms_to_decimal(val_w, "W")
        self.assertAlmostEqual(dec_w, -2.2945, places=3)

    def test_load_valid_image_without_exif(self):
        img_path = self.dir_path / "test_clean.jpg"
        img = Image.new("RGB", (640, 480), color=(73, 109, 137))
        img.save(img_path, format="JPEG")

        record = load_and_assess_image(img_path, storage_dir=self.dir_path)
        self.assertEqual(record.filename, "test_clean.jpg")
        self.assertEqual(record.width, 640)
        self.assertEqual(record.height, 480)
        self.assertEqual(record.aspect_ratio, 1.3333)
        self.assertEqual(len(record.sha256), 64)
        self.assertEqual(record.integrity.status, "absent")
        self.assertFalse(record.integrity.has_gps)
        self.assertIsNotNone(record.preserved_path)
        self.assertTrue(record.preserved_path.is_file())

    def test_load_image_with_gps_exif(self):
        img_path = self.dir_path / "test_gps.jpg"
        img = Image.new("RGB", (800, 600), color=(200, 200, 200))

        exif = img.getexif()
        # Set Make and Model
        exif[ExifTags.Base.Make] = "Sony"
        exif[ExifTags.Base.Model] = "ILCE-7RM4"
        exif[ExifTags.Base.DateTime] = "2024:06:15 14:30:00"

        # Set GPS SubIFD
        gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
        gps_ifd[1] = "N"  # GPSLatitudeRef
        gps_ifd[2] = (40, 42, 46.1)  # GPSLatitude
        gps_ifd[3] = "W"  # GPSLongitudeRef
        gps_ifd[4] = (74, 0, 21.4)  # GPSLongitude
        gps_ifd[5] = 0  # GPSAltitudeRef (above sea level)
        gps_ifd[6] = 12.5  # GPSAltitude
        gps_ifd[17] = 180.0  # GPSImgDirection
        img.save(img_path, format="JPEG", exif=exif)

        record = load_and_assess_image(img_path, storage_dir=self.dir_path)
        self.assertTrue(record.exif.has_exif)
        self.assertEqual(record.exif.make, "Sony")
        self.assertEqual(record.exif.model, "ILCE-7RM4")
        self.assertIsNotNone(record.exif.gps)
        self.assertAlmostEqual(record.exif.gps.latitude, 40.7128, places=2)
        self.assertAlmostEqual(record.exif.gps.longitude, -74.0059, places=2)
        self.assertEqual(record.exif.gps.altitude_m, 12.5)
        self.assertEqual(record.exif.gps.heading_deg, 180.0)
        self.assertEqual(record.integrity.status, "complete")
        self.assertTrue(record.integrity.has_gps)

    def test_unsupported_format(self):
        txt_path = self.dir_path / "file.txt"
        txt_path.write_text("not an image")
        with self.assertRaises(ValueError):
            load_and_assess_image(txt_path)


if __name__ == "__main__":
    unittest.main()
