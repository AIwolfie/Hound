"""End-to-End Investigation Scenarios for HOUND Phase 4 Autonomous Agent.

Implements three realistic benchmark scenarios:
1. Strong Evidence: Independent multi-modal clues converge on landmark.
2. Conflicting Evidence: Competing candidate challenged & rejected via contradiction.
3. Weak Evidence: Featureless negative control resulting in honest abstention.
"""
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from agent.agent_models import FinalInvestigationDossier
from agent.investigation_agent import InvestigationAgent
from core.configuration import AgentConfig
from core.image_loader import ImageRecord, load_and_assess_image
from storage.database import DatabaseManager
from storage.evidence_store import EvidenceItem, EvidenceStore
from storage.investigation_store import InvestigationStore


class TestAgentScenarios(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.db_path = self.workspace / "test_hound.db"
        self.db = DatabaseManager(self.db_path)
        self.inv_store = InvestigationStore(self.db, self.workspace / "storage")
        self.evid_store = EvidenceStore(self.db)
        self.config = AgentConfig(
            max_iterations=10,
            max_tool_calls=20,
            max_external_requests=10,
            max_duration_seconds=30,
            offline_mode=False,
        )
        self.agent = InvestigationAgent(db_manager=self.db, config=self.config)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_dummy_image(self, filename: str) -> Path:
        img_path = self.workspace / filename
        img = Image.new("RGB", (600, 400), color="blue")
        img.save(img_path)
        return img_path

    def test_scenario_a_strong_evidence_convergence(self):
        """Scenario A: Multiple independent clues converge to localize target."""
        img_path = self._create_dummy_image("scenario_a.jpg")
        img_rec = load_and_assess_image(img_path, preserve_original=False)
        inv = self.inv_store.create_investigation(img_rec)

        # Seed pre-existing evidence items from intake
        # 1. OCR clue
        self.evid_store.add_evidence(
            EvidenceItem(
                investigation_id=inv.id,
                kind="ocr",
                status="observed",
                text="[Road Sign] US-101 NORTH / CA COAST",
                confidence_weight=0.90,
            )
        )
        # 2. Solar clue (sun elevation 42.0°)
        self.evid_store.add_evidence(
            EvidenceItem(
                investigation_id=inv.id,
                kind="solar",
                status="computed",
                text="[Solar Ephemeris] Estimated solar elevation: 42.1°",
                confidence_weight=0.85,
            )
        )
        # 3. Observed skyline clue
        self.evid_store.add_evidence(
            EvidenceItem(
                investigation_id=inv.id,
                kind="terrain",
                status="observed",
                text="[Terrain] Observed Skyline profile detected along northern horizon",
                confidence_weight=0.80,
            )
        )

        # Run autonomous agent investigation
        dossier: FinalInvestigationDossier = self.agent.investigate(
            investigation_id=inv.id,
            max_iterations=8,
            offline=True,  # Test in reliable offline mode
        )

        # Assertions
        self.assertIsNotNone(dossier.primary_hypothesis)
        self.assertIn(dossier.final_status, ("localized", "region_identified", "candidates_remaining"))
        self.assertGreaterEqual(dossier.total_tool_calls, 1)
        self.assertGreater(len(dossier.timeline), 1)

        # Top candidate should be US-101 corridor or California
        top = dossier.primary_hypothesis
        self.assertTrue(any(k in top.name for k in ("US-101", "California", "Route")))
        self.assertGreater(top.heuristic_score, 5.0)

    def test_scenario_b_conflicting_evidence_rejection(self):
        """Scenario B: Two candidate regions exist; one violates solar physics and is rejected."""
        img_path = self._create_dummy_image("scenario_b.jpg")
        img_rec = load_and_assess_image(img_path, preserve_original=False)
        inv = self.inv_store.create_investigation(img_rec)

        # High solar elevation observed (65.0° noon sun)
        self.evid_store.add_evidence(
            EvidenceItem(
                investigation_id=inv.id,
                kind="solar",
                status="computed",
                text="[Solar Ephemeris] Estimated solar elevation: 65.0°",
                confidence_weight=0.95,
            )
        )
        self.evid_store.add_evidence(
            EvidenceItem(
                investigation_id=inv.id,
                kind="ocr",
                status="observed",
                text="[Road Sign] Route 1",
                confidence_weight=0.85,
            )
        )

        # Pre-register two competing candidate hypotheses:
        # Candidate 1: Arctic Norway (70°N) -> Impossible for 65° solar elevation!
        arctic_hyp = self.agent.hyp_mgr.register_candidate(
            investigation_id=inv.id,
            name="Route 1 (Finnmark, Norway)",
            level="road",
            lat=70.0,
            lon=25.0,
            initial_score=4.0,
        )
        # Candidate 2: California (35°N) -> Feasible!
        cal_hyp = self.agent.hyp_mgr.register_candidate(
            investigation_id=inv.id,
            name="Route 1 (California, USA)",
            level="road",
            lat=35.0,
            lon=-120.5,
            initial_score=4.0,
        )

        state = self.agent.investigate(
            investigation_id=inv.id,
            max_iterations=6,
            offline=True,
        )

        # Verify Arctic Norway was challenged or rejected due to solar ceiling violation
        # Reload hypotheses from DB
        loaded_state = self.agent.inv_store.get_investigation(inv.id)
        self.assertIsNotNone(loaded_state)

        # Check rejected hypotheses in dossier
        rejected_names = [h.name for h in state.rejected_hypotheses]
        self.assertIn("Route 1 (Finnmark, Norway)", rejected_names)

        # California candidate remains primary or active
        self.assertIsNotNone(state.primary_hypothesis)
        self.assertEqual(state.primary_hypothesis.name, "Route 1 (California, USA)")

    def test_scenario_c_weak_evidence_honest_abstention(self):
        """Scenario C: Featureless indoor room with no geographic clues; agent honestly abstains."""
        img_path = self._create_dummy_image("scenario_c_indoor.jpg")
        img_rec = load_and_assess_image(img_path, preserve_original=False)
        inv = self.inv_store.create_investigation(img_rec)

        # Run investigation with zero geographic clues
        dossier: FinalInvestigationDossier = self.agent.investigate(
            investigation_id=inv.id,
            max_iterations=6,
            offline=True,
        )

        # System must honestly abstain
        self.assertEqual(dossier.final_status, "insufficient_evidence")
        self.assertEqual(dossier.stopping_reason, "insufficient_evidence")
        self.assertEqual(dossier.geographic_precision_justified, "Unlocalized")
        self.assertGreater(len(dossier.abstention_reasons), 0)
        self.assertIn("no discernible geographic landmarks", dossier.abstention_reasons[0])
