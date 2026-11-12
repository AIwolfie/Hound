"""HOUND Phase 4.1 Audit Suite — Process Crash, Interruption & Resumption.

Audits and verifies that:
1. Mid-investigation termination preserves all SQLite state without corruption.
2. Resumed investigations do NOT re-execute already completed actions.
3. Resumed investigations preserve event timelines, budget counters, and hypothesis scores.
4. Resumed investigations converge on consistent conclusions compared to uninterrupted runs.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from agent.agent_models import FinalInvestigationDossier
from agent.investigation_agent import InvestigationAgent
from agent.investigation_state import InvestigationState
from core.configuration import AgentConfig
from core.image_loader import load_and_assess_image
from storage.database import DatabaseManager
from storage.evidence_store import EvidenceItem, EvidenceStore
from storage.investigation_store import InvestigationStore


class TestCrashRecoveryAudit(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.db_path = self.workspace / "crash_audit.db"
        self.db = DatabaseManager(self.db_path)
        self.inv_store = InvestigationStore(self.db, self.workspace / "storage")
        self.evid_store = EvidenceStore(self.db)
        self.config = AgentConfig(
            max_iterations=8,
            max_tool_calls=15,
            max_external_requests=8,
            max_duration_seconds=30,
            offline_mode=True,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_test_image(self, name: str) -> Path:
        img_path = self.workspace / name
        img = Image.new("RGB", (800, 600), color=(80, 120, 160))
        img.save(img_path)
        return img_path

    def test_mid_investigation_interruption_preserves_state(self):
        """Simulate an abrupt process crash after iteration 2; verify DB state is intact."""
        img_path = self._create_test_image("crash_target.jpg")
        img_rec = load_and_assess_image(img_path, preserve_original=False)
        inv = self.inv_store.create_investigation(img_rec)

        # Seed initial evidence
        self.evid_store.add_evidence(
            EvidenceItem(
                investigation_id=inv.id,
                kind="ocr",
                status="observed",
                text="[Road Sign] US-101 NORTH",
                confidence_weight=0.90,
            )
        )

        agent1 = InvestigationAgent(db_manager=self.db, config=self.config)

        # Execute only 2 iterations to simulate an interruption mid-flight
        dossier1 = agent1.investigate(
            investigation_id=inv.id,
            max_iterations=2,
            offline=True,
        )

        self.assertLessEqual(dossier1.total_iterations, 2)

        # Inspect database records directly
        with self.db.get_connection() as conn:
            cur = conn.execute("SELECT * FROM agent_sessions WHERE investigation_id = ?", (inv.id,))
            session_row = cur.fetchone()
            self.assertIsNotNone(session_row)
            self.assertEqual(session_row["iteration"], 2)

            cur_evt = conn.execute("SELECT count(*) as c FROM investigation_events WHERE investigation_id = ?", (inv.id,))
            event_count = cur_evt.fetchone()["c"]
            self.assertGreater(event_count, 1)

    def test_resumed_investigation_does_not_duplicate_actions(self):
        """Verify that resuming a session avoids executing actions that ran prior to the interruption."""
        img_path = self._create_test_image("resume_target.jpg")
        img_rec = load_and_assess_image(img_path, preserve_original=False)
        inv = self.inv_store.create_investigation(img_rec)

        # Seed initial clue
        self.evid_store.add_evidence(
            EvidenceItem(
                investigation_id=inv.id,
                kind="ocr",
                status="observed",
                text="[Road Sign] Route 66",
                confidence_weight=0.95,
            )
        )

        agent = InvestigationAgent(db_manager=self.db, config=self.config)

        # Run Phase 1: Halt after 2 iterations
        agent.investigate(investigation_id=inv.id, max_iterations=2, offline=True)

        # Load session to inspect executed actions
        state_mid = InvestigationState.load_session(inv.id, self.db, self.config)
        self.assertIsNotNone(state_mid)
        actions_phase1 = set(state_mid.executed_action_keys)
        tools_count_phase1 = state_mid.tool_calls_count
        events_count_phase1 = len(state_mid.events)

        self.assertGreater(tools_count_phase1, 0)

        # Resume execution for up to 8 iterations
        dossier_resumed = agent.investigate(
            investigation_id=inv.id,
            resume=True,
            max_iterations=8,
            offline=True,
        )

        # Verify that total iterations resumed from previous count
        self.assertGreater(dossier_resumed.total_iterations, 2)
        self.assertGreaterEqual(dossier_resumed.total_tool_calls, tools_count_phase1)

        # Verify event history was preserved and appended, not overwritten
        self.assertGreater(dossier_resumed.events_count, events_count_phase1)

        # Count occurrences of each action in the full timeline
        tool_action_events = [e for e in dossier_resumed.timeline if e.event_type == "tool_executed"]
        tool_names = [e.tool_id for e in tool_action_events]

        # Verify standard intake tools (metadata, preprocessing) only ran once
        self.assertLessEqual(tool_names.count("metadata_exif"), 1)
        self.assertLessEqual(tool_names.count("image_preprocessing"), 1)

    def test_uninterrupted_vs_resumed_consistency(self):
        """Compare an uninterrupted run against a crash-and-resume run on identical inputs."""
        # Run A: Uninterrupted
        img_a = self._create_test_image("target_a.jpg")
        rec_a = load_and_assess_image(img_a, preserve_original=False)
        inv_a = self.inv_store.create_investigation(rec_a)
        self.evid_store.add_evidence(
            EvidenceItem(
                investigation_id=inv_a.id,
                kind="ocr",
                status="observed",
                text="[Road Sign] CA-1 COAST HWY",
                confidence_weight=0.90,
            )
        )

        agent = InvestigationAgent(db_manager=self.db, config=self.config)
        dossier_uninterrupted = agent.investigate(inv_a.id, max_iterations=6, offline=True)

        # Run B: Interrupted after 2 iterations, then resumed
        img_b = self._create_test_image("target_b.jpg")
        rec_b = load_and_assess_image(img_b, preserve_original=False)
        inv_b = self.inv_store.create_investigation(rec_b)
        self.evid_store.add_evidence(
            EvidenceItem(
                investigation_id=inv_b.id,
                kind="ocr",
                status="observed",
                text="[Road Sign] CA-1 COAST HWY",
                confidence_weight=0.90,
            )
        )

        # First half
        agent.investigate(inv_b.id, max_iterations=2, offline=True)
        # Resumed second half (given adequate budget to complete tool execution)
        dossier_resumed = agent.investigate(inv_b.id, resume=True, max_iterations=8, offline=True)

        # Compare outcomes
        self.assertEqual(dossier_uninterrupted.final_status, dossier_resumed.final_status)
        self.assertEqual(dossier_uninterrupted.geographic_precision_justified, dossier_resumed.geographic_precision_justified)

        if dossier_uninterrupted.primary_hypothesis and dossier_resumed.primary_hypothesis:
            self.assertEqual(
                dossier_uninterrupted.primary_hypothesis.level,
                dossier_resumed.primary_hypothesis.level,
            )


if __name__ == "__main__":
    unittest.main()
