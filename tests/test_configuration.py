"""Tests for HOUND Configuration Management."""
import os
import unittest
from pathlib import Path
import tempfile
import yaml

from core.configuration import HoundConfig, load_config


class TestConfiguration(unittest.TestCase):
    def test_default_config_loading(self):
        config = load_config()
        self.assertIsInstance(config, HoundConfig)
        self.assertEqual(config.hound.version, "1.0.0")
        self.assertTrue(config.paths.storage_dir.is_absolute())
        self.assertTrue(config.paths.database_path.is_absolute())

    def test_custom_yaml_override(self):
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml", encoding="utf-8") as tmp:
            yaml.dump(
                {
                    "hound": {"log_level": "DEBUG"},
                    "network": {"timeout_seconds": 45},
                },
                tmp,
            )
            tmp_path = tmp.name

        try:
            config = load_config(tmp_path)
            self.assertEqual(config.hound.log_level, "DEBUG")
            self.assertEqual(config.network.timeout_seconds, 45)
        finally:
            os.remove(tmp_path)

    def test_env_var_override(self):
        os.environ["HOUND_LOG_LEVEL"] = "WARNING"
        try:
            config = load_config()
            self.assertEqual(config.hound.log_level, "WARNING")
        finally:
            del os.environ["HOUND_LOG_LEVEL"]


if __name__ == "__main__":
    unittest.main()
