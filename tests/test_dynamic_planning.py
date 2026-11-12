"""HOUND Phase 4.1 Audit Suite — Dynamic Planning & Adaptive Reasoning.

Validates that HOUND's autonomous planner:
1. Re-evaluates evidence after each action.
2. Dynamically alters tool selection when clues or hypotheses change.
3. Adapts when contradictory evidence is injected halfway through execution.
4. Strictly suppresses already executed or redundant actions.
5. Prioritizes falsification and independent multi-modal verification via VOI.
6. Can choose not to execute an available tool when conditions do not warrant it.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image

from agent.agent_models import FinalInvestigationDossier, HypothesisRecord, InvestigationAction
from agent.hypothesis_manager import HypothesisManager
from agent.investigation_agent import InvestigationAgent
from agent.investigation_state import InvestigationState
from agent.planner import InvestigationPlanner
from agent.tool_registry import ToolExecutionContext, ToolRegistry
from core.configuration import AgentConfig
from core.image_loader import load_and_assess_image
from storage.database import DatabaseManager
from storage.evidence_store import EvidenceItem, EvidenceStore
from storage.investigation_store import InvestigationStore


class TestDynamicPlanningAudit(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.db_path = self.workspace / "audit_hound.db"
        self.db = DatabaseManager(self.db_path)
        self.inv_store = InvestigationStore(self.db, self.workspace / "storage")
        self.evid_store = EvidenceStore(self.db)
        self.config = AgentConfig(
            max_iterations=10,
            max_tool_calls=20,
            max_external_requests=10,
            max_duration_seconds=30,
            offline_mode=True,
        )
        self.registry = ToolRegistry()
        self.planner = InvestigationPlanner(self.registry)
        self.hyp_mgr = HypothesisManager(self.db)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_test_image(self, name: str) -> Path:
        img_path = self.workspace / name
        img = Image.new("RGB", (640, 480), color=(100, 150, 200))
        img.save(img_path)
        return img_path

    def test_planner_evaluates_new_evidence_and_adapts_tool_selection(self):
        """Verify that the planner's selected action dynamically updates as clues are acquired."""
        state = InvestigationState("inv-dyn-1", config=self.config, db_manager=self.db)

        # Initially, with zero clues, metadata extraction is prioritized
        action1 = self.planner.select_next_action(state)
        self.assertIsNotNone(action1)
        self.assertEqual(action1.tool_id, "metadata_exif")

        # Simulate metadata execution
        state.clues["metadata_analyzed"] = True
        state.record_action("tool:metadata_exif")

        # Next action shifts to visual intake (solar_shadow or ocr_extraction)
        action2 = self.planner.select_next_action(state)
        self.assertIsNotNone(action2)
        self.assertIn(action2.tool_id, ("solar_shadow", "ocr_extraction", "image_preprocessing"))

        # Simulate OCR and solar executed with a road clue
        state.clues["ocr_executed"] = True
        state.clues["has_ocr_road"] = True
        state.record_action("tool:ocr_extraction")
        state.clues["solar_executed"] = True
        state.record_action("tool:solar_shadow")

        # Mark preprocessing, infrastructure, and visual retrieval also executed
        state.clues["preprocessed"] = True
        state.record_action("tool:image_preprocessing")
        state.clues["infra_executed"] = True
        state.record_action("tool:infrastructure_detection")
        state.clues["vpr_executed"] = True
        state.record_action("tool:visual_place_search")

        # Next action should now propose candidate generation because road clues exist
        action3 = self.planner.select_next_action(state)
        self.assertIsNotNone(action3)
        self.assertEqual(action3.tool_id, "candidate_generation")

    def test_planner_avoids_redundant_or_completed_actions(self):
        """Verify that executed actions are never re-selected even if gaps exist."""
        state = InvestigationState("inv-dyn-2", config=self.config, db_manager=self.db)

        # Mark all early tools as already executed
        state.record_action("tool:metadata_exif")
        state.record_action("tool:image_preprocessing")
        state.record_action("tool:ocr_extraction")
        state.record_action("tool:infrastructure_detection")
        state.record_action("tool:solar_shadow")

        # Clues remain unflagged, but action keys are recorded
        actions = self.planner._generate_candidate_actions(state)
        tool_ids = [a.tool_id for a in actions]

        self.assertNotIn("metadata_exif", tool_ids)
        self.assertNotIn("image_preprocessing", tool_ids)
        self.assertNotIn("ocr_extraction", tool_ids)
        self.assertNotIn("infrastructure_detection", tool_ids)
        self.assertNotIn("solar_shadow", tool_ids)

    def test_planner_can_choose_not_to_use_available_tools(self):
        """Verify that the planner does not invoke ridge_matching if no skyline clue exists."""
        state = InvestigationState("inv-dyn-3", config=self.config, db_manager=self.db)
        state.clues["candidates_generated"] = True
        state.clues["has_skyline"] = False  # NO mountain skyline observed

        cand = self.hyp_mgr.register_candidate(
            investigation_id="inv-dyn-3",
            name="Rocky Mountain Pass",
            level="point",
            lat=40.0,
            lon=-105.5,
            initial_score=5.0,
        )
        cand.status = "active"
        state.hypotheses[cand.id] = cand

        actions = self.planner._generate_candidate_actions(state)
        tool_ids = [a.tool_id for a in actions]

        # ridge_matching must NOT be scheduled when skyline is absent
        self.assertNotIn("ridge_matching", tool_ids)

        # Now activate skyline clue
        state.clues["has_skyline"] = True
        actions_with_skyline = self.planner._generate_candidate_actions(state)
        tool_ids_with_skyline = [a.tool_id for a in actions_with_skyline]

        # ridge_matching must now be scheduled
        self.assertIn("ridge_matching", tool_ids_with_skyline)

    def test_contradictory_evidence_injected_halfway_alters_hypotheses_and_plan(self):
        """Crucial Phase 4.1 test: Inject contradictory evidence midway through execution.

        Verify that:
        1. Leading candidate score drops / status changes to rejected/challenged.
        2. Planner dynamically redirects verification away from the contradicted candidate
           and onto the surviving candidate.
        """
        img_path = self._create_test_image("test_midway.jpg")
        img_rec = load_and_assess_image(img_path, preserve_original=False)
        inv = self.inv_store.create_investigation(img_rec)

        agent = InvestigationAgent(db_manager=self.db, config=self.config)
        state = InvestigationState(inv.id, config=self.config, db_manager=self.db)

        # 1. Setup two competing candidates
        cand_a = agent.hyp_mgr.register_candidate(
            investigation_id=inv.id,
            name="Candidate Alpha (Norway Route E6)",
            level="road",
            lat=68.5,
            lon=17.5,
            initial_score=8.0,
        )
        cand_a.status = "active"
        state.hypotheses[cand_a.id] = cand_a

        cand_b = agent.hyp_mgr.register_candidate(
            investigation_id=inv.id,
            name="Candidate Beta (California Route 1)",
            level="road",
            lat=36.2,
            lon=-121.8,
            initial_score=6.0,
        )
        cand_b.status = "active"
        state.hypotheses[cand_b.id] = cand_b

        # Candidate Alpha is currently leading (Score 24 vs 18)
        self.assertGreater(cand_a.heuristic_score, cand_b.heuristic_score)

        # Mark visual tools done so planner focuses on candidate verification
        state.clues["metadata_analyzed"] = True
        state.clues["ocr_executed"] = True
        state.clues["infra_executed"] = True
        state.clues["solar_executed"] = True
        state.clues["candidates_generated"] = True
        state.clues["has_solar_shadow"] = True  # Shadow clue exists

        # Check planner's initial priority: adversarial check on leading Candidate Alpha
        next_act = agent.planner.select_next_action(state)
        self.assertIsNotNone(next_act)
        self.assertEqual(next_act.tool_id, "adversarial_solar_check")
        self.assertEqual(next_act.target_hypothesis_id, cand_a.id)

        # === MIDWAY INJECTION OF CONTRADICTORY EVIDENCE ===
        # Inject new high solar elevation observation (62° noon sun)
        contradictory_ev = agent.evid_store.add_evidence(
            EvidenceItem(
                investigation_id=inv.id,
                kind="solar",
                status="observed",
                text="[Solar Ephemeris] High noon sun observed: Solar Elevation 62.0°",
                confidence_weight=0.98,
            )
        )
        state.evidence_items[contradictory_ev.id] = contradictory_ev

        # Candidate Alpha at 68.5°N has an absolute astronomical ceiling of ~45°
        # Active contradiction recorded:
        agent.hyp_mgr.record_contradiction(
            hypothesis=cand_a,
            evidence_item=contradictory_ev,
            rejection_reason="Observed solar elevation 62° exceeds northern solstice ceiling 45° at lat 68.5°",
            is_definitive_falsification=True,
        )

        # Verify Alpha is rejected and its score is zeroed
        self.assertEqual(cand_a.status, "rejected")
        self.assertEqual(cand_a.heuristic_score, 0.0)

        # Re-evaluate planner: Alpha should be ignored; next action MUST target surviving Candidate Beta!
        subsequent_act = agent.planner.select_next_action(state)
        self.assertIsNotNone(subsequent_act)
        self.assertEqual(subsequent_act.target_hypothesis_id, cand_b.id)
        self.assertEqual(subsequent_act.tool_id, "adversarial_solar_check")

    def test_voi_priority_boosts_falsification_actions(self):
        """Verify that actions with high falsification value receive higher composite priority."""
        state = InvestigationState("inv-voi", config=self.config, db_manager=self.db)

        cand = self.hyp_mgr.register_candidate(
            investigation_id="inv-voi",
            name="Test Target",
            level="point",
            lat=45.0,
            lon=10.0,
            initial_score=5.0,
        )
        cand.status = "active"
        state.hypotheses[cand.id] = cand

        falsification_action = InvestigationAction(
            tool_id="adversarial_solar_check",
            reason="Falsification ceiling test",
            expected_info_gain=0.9,
            cost_estimate=0.02,
        )

        standard_action = InvestigationAction(
            tool_id="infrastructure_detection",
            reason="Heuristic visual search",
            expected_info_gain=0.6,
            cost_estimate=0.10,
        )

        score_falsif = self.planner._compute_priority_score(falsification_action, state)
        score_std = self.planner._compute_priority_score(standard_action, state)

        # Falsification potential multiplier is 2.0x for adversarial checks
        self.assertEqual(score_falsif.falsification_potential, 2.0)
        self.assertEqual(score_std.falsification_potential, 1.0)
        self.assertGreater(score_falsif.composite_priority, score_std.composite_priority)


if __name__ == "__main__":
    unittest.main()
