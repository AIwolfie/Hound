"""Unit tests for HOUND Phase 4 Agent Architecture.

Tests tool registry, planner, hypothesis manager, verification engine,
stopping policy, state persistence, and LLM fallback.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent.agent_models import (
    FinalInvestigationDossier,
    HypothesisRecord,
    InvestigationAction,
    InvestigationEvent,
    ToolResult,
)
from agent.hypothesis_manager import HypothesisManager
from agent.investigation_state import InvestigationState
from agent.llm_provider import GenericHTTPLLM, LLMPlanProposal, OfflineFallbackLLM, get_llm_provider
from agent.planner import InvestigationPlanner
from agent.stopping_policy import StoppingPolicy
from agent.tool_registry import ToolDefinition, ToolExecutionContext, ToolRegistry
from agent.verification_engine import VerificationEngine
from core.configuration import AgentConfig, get_config
from storage.database import DatabaseManager
from storage.evidence_store import EvidenceItem, EvidenceStore
from storage.investigation_store import InvestigationStore


class TestAgentArchitecture(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_hound.db"
        self.db = DatabaseManager(self.db_path)
        self.inv_store = InvestigationStore(self.db, Path(self.temp_dir.name))
        self.evid_store = EvidenceStore(self.db)
        self.config = AgentConfig(
            max_iterations=5,
            max_tool_calls=10,
            max_external_requests=5,
            max_duration_seconds=30,
            offline_mode=True,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_tool_registry_registration_and_execution(self):
        """Verify that allowlisted tools execute properly with context."""
        registry = ToolRegistry()
        tools = registry.list_tools()
        self.assertGreaterEqual(len(tools), 12)

        # Check tool metadata
        meta_tool = registry.get("metadata_exif")
        self.assertIsNotNone(meta_tool)
        self.assertEqual(meta_tool.category, "visual")

        # Non-existent tool returns error ToolResult
        ctx = ToolExecutionContext("inv-fake", db_manager=self.db, offline_mode=True)
        bad_result = registry.execute_tool("arbitrary_dangerous_tool", ctx, {})
        self.assertFalse(bad_result.success)
        self.assertEqual(bad_result.status, "error")

    def test_investigation_state_budgets_and_events(self):
        """Verify budget counters, event logging, and state serialization."""
        state = InvestigationState("inv-101", config=self.config, db_manager=self.db)
        self.assertEqual(state.iteration, 0)
        self.assertEqual(state.tool_calls_count, 0)

        # Record action
        state.record_action("tool:ocr_extraction", is_external=False)
        self.assertEqual(state.tool_calls_count, 1)
        self.assertTrue(state.is_action_already_executed("tool:ocr_extraction"))

        # Log event
        evt = state.add_event(
            event_type="test_event",
            action_summary="Executed test action",
            result_summary="Success",
            tool_id="test_tool",
            evidence_ids=["E1"],
            duration_s=0.05,
        )
        self.assertEqual(len(state.events), 1)
        self.assertEqual(evt.tool_id, "test_tool")

        # Test budget triggers
        state.iteration = 5
        exhausted, reason = state.is_budget_exhausted()
        self.assertTrue(exhausted)
        self.assertEqual(reason, "budget_reached")

    def test_hypothesis_lifecycle_and_non_circular_scoring(self):
        """Test hypothesis creation, independent corroboration, and non-circular scoring."""
        hyp_mgr = HypothesisManager(db_manager=self.db)
        h = hyp_mgr.register_candidate(
            investigation_id="inv-202",
            name="Corridor: Route 66",
            level="road",
            lat=35.1983,
            lon=-111.6513,
            seed_evidence_id="E-SEED-1",
            seed_evidence_kind="ocr_road_shield",
            initial_score=2.0,
        )
        self.assertEqual(h.status, "proposed")
        # Road level multiplier is 3.0 -> initial score is 2.0 * 3.0 = 6.0
        self.assertAlmostEqual(h.heuristic_score, 6.0)

        # 1. Attempt circular reinforcement with seed clue -> MUST BE IGNORED
        seed_ev = EvidenceItem(
            id="E-SEED-1",
            investigation_id="inv-202",
            kind="ocr",
            status="observed",
            text="Route 66",
            confidence_weight=0.9,
        )
        hyp_mgr.add_corroboration(h, seed_ev)
        # Score must remain unchanged
        self.assertAlmostEqual(h.heuristic_score, 6.0)
        self.assertEqual(len(h.supporting_evidence_ids), 0)

        # 2. Add independent corroboration -> score increases
        indep_ev = EvidenceItem(
            id="E-INDEP-2",
            investigation_id="inv-202",
            kind="osm",
            status="computed",
            text="Verified primary highway corridor on OSM",
            confidence_weight=0.8,
        )
        hyp_mgr.add_corroboration(h, indep_ev, corroboration_bonus=3.0)
        self.assertIn("E-INDEP-2", h.supporting_evidence_ids)
        self.assertGreater(h.heuristic_score, 6.0)
        self.assertEqual(h.status, "active")

    def test_adversarial_solar_falsification(self):
        """Verify active falsification when solar elevation ceiling is exceeded."""
        with self.db.get_connection() as conn:
            conn.execute(
                "INSERT INTO investigations (id, sha256, filename, original_path, status, started_at) "
                "VALUES ('inv-solar-test', 'dummyhash', 'solar.jpg', 'solar.jpg', 'open', '2026-11-12T00:00:00')"
            )
            conn.commit()

        hyp_mgr = HypothesisManager(db_manager=self.db)
        verifier = VerificationEngine(hyp_mgr, self.evid_store)

        # Candidate at northern latitude 70.0°N
        h = hyp_mgr.register_candidate(
            investigation_id="inv-solar-test",
            name="Northern Norway Candidate",
            level="point",
            lat=70.0,
            lon=25.0,
        )

        # Test with impossible solar elevation 60° (impossible at 70°N on equinox ceiling 43.44°)
        res = verifier.evaluate_solar_falsification(
            hypothesis=h,
            observed_solar_elevation=60.0,
        )

        self.assertEqual(res.outcome, "contradicted")
        self.assertTrue(res.is_falsified)
        self.assertEqual(h.status, "rejected")
        self.assertEqual(h.heuristic_score, 0.0)
        self.assertIn("solar ceiling", h.rejection_reason.lower())

    def test_stopping_policy_honest_abstention(self):
        """Verify that stopping policy honestly abstains on weak/indoor negative controls."""
        state = InvestigationState("inv-weak", config=self.config, db_manager=self.db)
        # Mark visual tools as executed without geographic anchors
        state.clues["metadata_analyzed"] = True
        state.clues["ocr_executed"] = True
        state.clues["infra_executed"] = True
        state.clues["solar_executed"] = True
        state.clues["has_gps"] = False
        state.clues["has_ocr_road"] = False

        should_stop, reason, status = StoppingPolicy.evaluate(state)
        self.assertTrue(should_stop)
        self.assertEqual(reason, "insufficient_evidence")
        self.assertEqual(status, "insufficient_evidence")

    def test_offline_llm_provider_fallback(self):
        """Verify that offline LLM provider gracefully returns None without raising errors."""
        offline_llm = OfflineFallbackLLM()
        proposal = offline_llm.plan_next_action("obs", "hyp", "tools", "exec")
        self.assertIsNone(proposal)

        factory_llm = get_llm_provider(AgentConfig(offline_mode=True))
        self.assertIsInstance(factory_llm, OfflineFallbackLLM)
