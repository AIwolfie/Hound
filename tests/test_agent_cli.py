"""Unit tests for HOUND Phase 4 Agent CLI Commands.

Tests:
- hound agent investigate
- hound agent status
- hound agent hypotheses
- hound agent history
- hound agent verify
- hound agent report
"""
import tempfile
import unittest
from pathlib import Path
from click.testing import CliRunner
from PIL import Image

from core.configuration import get_config
from core.image_loader import load_and_assess_image
from interface.cli import cli
from storage.database import DatabaseManager
from storage.investigation_store import InvestigationStore


class TestAgentCLI(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        cfg = get_config()
        self.db = DatabaseManager(cfg.paths.database_path)
        self.inv_store = InvestigationStore(self.db, cfg.paths.storage_dir)

        # Create dummy image and investigation
        img_path = self.workspace / "agent_cli_test.jpg"
        img = Image.new("RGB", (400, 300), color="green")
        img.save(img_path)

        rec = load_and_assess_image(img_path, preserve_original=False)
        self.inv = self.inv_store.create_investigation(rec)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_cli_agent_investigate(self):
        """Test 'hound agent investigate <id>' execution."""
        result = self.runner.invoke(
            cli,
            ["agent", "investigate", self.inv.id, "--max-iterations", "3", "--offline"],
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("HOUND Autonomous Investigation Agent", result.output)
        self.assertIn("Autonomous Investigation Concluded", result.output)
        self.assertIn("Final Status:", result.output)

    def test_cli_agent_status(self):
        """Test 'hound agent status <id>' output."""
        # Run investigate first to populate state
        self.runner.invoke(cli, ["agent", "investigate", self.inv.id, "--max-iterations", "2", "--offline"])

        result = self.runner.invoke(cli, ["agent", "status", self.inv.id])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Agent Session Telemetry", result.output)
        self.assertIn("Session State", result.output)
        self.assertIn("Iterations Completed", result.output)

    def test_cli_agent_hypotheses(self):
        """Test 'hound agent hypotheses <id>' output."""
        self.runner.invoke(cli, ["agent", "investigate", self.inv.id, "--max-iterations", "2", "--offline"])

        result = self.runner.invoke(cli, ["agent", "hypotheses", self.inv.id])
        self.assertEqual(result.exit_code, 0)
        # Should display table or no hypotheses message
        self.assertTrue("Agent Competing Hypotheses Ledger" in result.output or "No hypotheses recorded" in result.output)

    def test_cli_agent_history(self):
        """Test 'hound agent history <id>' output."""
        self.runner.invoke(cli, ["agent", "investigate", self.inv.id, "--max-iterations", "2", "--offline"])

        result = self.runner.invoke(cli, ["agent", "history", self.inv.id])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Investigation Timeline", result.output)
        self.assertIn("Iter", result.output)

    def test_cli_agent_verify(self):
        """Test 'hound agent verify <id>' output."""
        self.runner.invoke(cli, ["agent", "investigate", self.inv.id, "--max-iterations", "2", "--offline"])

        result = self.runner.invoke(cli, ["agent", "verify", self.inv.id])
        self.assertEqual(result.exit_code, 0)
        self.assertTrue("Generating Adversarial Verification Questions" in result.output or "No hypotheses found" in result.output)

    def test_cli_agent_report(self):
        """Test 'hound agent report <id>' output and saving to file."""
        self.runner.invoke(cli, ["agent", "investigate", self.inv.id, "--max-iterations", "2", "--offline"])

        report_file = self.workspace / "dossier.md"
        result = self.runner.invoke(cli, ["agent", "report", self.inv.id, "-o", str(report_file)])
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(report_file.is_file())
        content = report_file.read_text(encoding="utf-8")
        self.assertIn("HOUND INVESTIGATION DOSSIER", content)
        self.assertIn("Leading Geographic Hypothesis", content)
        self.assertIn("Facts vs. Deductions", content)
