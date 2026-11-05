"""Tests for HOUND CLI commands."""
import tempfile
import unittest
from pathlib import Path
from click.testing import CliRunner
from PIL import Image

from interface.cli import cli


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_cli_doctor(self):
        result = self.runner.invoke(cli, ["doctor"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("HOUND", result.output)
        self.assertIn("Python Runtime", result.output)
        self.assertIn("SQLite database", result.output)

    def test_cli_intake_and_list(self):
        img_path = self.dir_path / "cli_sample.jpg"
        img = Image.new("RGB", (400, 300), color="red")
        img.save(img_path)

        # Run intake
        result = self.runner.invoke(cli, ["intake", str(img_path)])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Target acquired", result.output)
        self.assertIn("Investigation initialized", result.output)
        self.assertIn("Target Forensic Profile", result.output)

        # Run list
        list_result = self.runner.invoke(cli, ["list"])
        self.assertEqual(list_result.exit_code, 0)
        self.assertIn("cli_sample", list_result.output)

    def test_cli_inspect(self):
        img_path = self.dir_path / "cli_inspect.jpg"
        img = Image.new("RGB", (200, 200), color="blue")
        img.save(img_path)

        result = self.runner.invoke(cli, ["inspect", str(img_path)])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Forensic Inspection", result.output)
        self.assertIn("cli_inspect.jpg", result.output)

    def test_cli_preprocess(self):
        img_path = self.dir_path / "cli_prep.jpg"
        img = Image.new("RGB", (300, 200), color="green")
        img.save(img_path)

        out_dir = self.dir_path / "cli_prep_out"
        result = self.runner.invoke(cli, ["preprocess", str(img_path), "-o", str(out_dir)])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Preprocessing complete", result.output)
        self.assertIn("full_normalized", result.output)

    def test_cli_ocr(self):
        img_path = self.dir_path / "cli_ocr.jpg"
        img = Image.new("RGB", (300, 200), color="white")
        img.save(img_path)

        result = self.runner.invoke(cli, ["ocr", str(img_path)])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("OCR investigation complete", result.output)

    def test_cli_analyze(self):
        img_path = self.dir_path / "cli_analyze.jpg"
        img = Image.new("RGB", (400, 300), color=(180, 180, 180))
        img.save(img_path)

        result = self.runner.invoke(cli, ["analyze", str(img_path)])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("HOUND VISUAL INVESTIGATION", result.output)
        self.assertIn("VISUAL INTELLIGENCE SUMMARY", result.output)
        self.assertIn("Text & OCR Observations", result.output)
        self.assertIn("Infrastructure Clues", result.output)


if __name__ == "__main__":
    unittest.main()
