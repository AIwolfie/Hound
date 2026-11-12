"""HOUND Investigation Planner.

Analyzes current observations, detects information gaps, computes expected
value of information (VOI), and prioritizes actions to either advance
the investigation or actively falsify candidate hypotheses.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from agent.agent_models import InvestigationAction
from agent.investigation_state import InvestigationState
from agent.llm_provider import BaseLLMProvider, get_llm_provider
from agent.tool_registry import ToolRegistry
from core.logger import get_logger

logger = get_logger("hound.agent.planner")


class ActionPriorityScore(BaseModel):
    action: InvestigationAction
    uncertainty_reduction: float = 0.5  # 0.0 to 1.0
    source_reliability: float = 0.8     # 0.0 to 1.0
    evidence_independence: float = 1.0  # 1.0 = independent, 0.2 = correlated
    falsification_potential: float = 1.0 # 2.0 = can disprove leading hypothesis, 1.0 = neutral
    cost_penalty: float = 0.1
    composite_priority: float = 1.0


class InvestigationPlanner:
    """Plans and prioritizes next investigative actions based on empirical evidence."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        llm_provider: Optional[BaseLLMProvider] = None,
    ):
        self.tools = tool_registry
        self.llm = llm_provider or get_llm_provider()

    def select_next_action(self, state: InvestigationState) -> Optional[InvestigationAction]:
        """Determine the single highest-value action to execute next."""
        # 1. Generate all candidate actions based on information gaps
        candidate_actions = self._generate_candidate_actions(state)
        if not candidate_actions:
            logger.debug("Planner found no further eligible candidate actions.")
            return None

        # 2. Consult optional LLM provider if configured
        if self.llm and not isinstance(self.llm, type(get_llm_provider(None))):
            llm_action = self._consult_llm(state, candidate_actions)
            if llm_action:
                return llm_action

        # 3. Score and rank actions using deterministic Priority Model
        scored_actions: List[ActionPriorityScore] = []
        for action in candidate_actions:
            score = self._compute_priority_score(action, state)
            scored_actions.append(score)

        # Sort descending by composite priority
        scored_actions.sort(key=lambda x: x.composite_priority, reverse=True)

        selected = scored_actions[0].action
        logger.info(
            f"Planner selected action '{selected.tool_id}' "
            f"(Priority: {scored_actions[0].composite_priority:.2f}, Reason: {selected.reason})"
        )
        return selected

    def _generate_candidate_actions(self, state: InvestigationState) -> List[InvestigationAction]:
        """Identify available actions that address current information gaps."""
        actions: List[InvestigationAction] = []

        # GAP 1: Intake Metadata Analysis
        if not state.clues["metadata_analyzed"]:
            act_key = "tool:metadata_exif"
            if not state.is_action_already_executed(act_key):
                actions.append(
                    InvestigationAction(
                        tool_id="metadata_exif",
                        reason="Inspect image EXIF metadata, camera optics FOV, and embedded GPS.",
                        priority=10.0,
                        expected_info_gain=0.9,
                        cost_estimate=0.01,
                        is_external_request=False,
                    )
                )

        # GAP 2: Image Preprocessing & Crop Pyramids
        if not state.clues.get("preprocessed", False):
            act_key = "tool:image_preprocessing"
            if not state.is_action_already_executed(act_key):
                actions.append(
                    InvestigationAction(
                        tool_id="image_preprocessing",
                        reason="Generate CLAHE contrast normalization, edge maps, and zoomed tile pyramids.",
                        priority=6.5,
                        expected_info_gain=0.5,
                        cost_estimate=0.05,
                        is_external_request=False,
                    )
                )

        # GAP 3: Multi-Pass OCR Text & Clue Extraction
        if not state.clues["ocr_executed"]:
            act_key = "tool:ocr_extraction"
            if not state.is_action_already_executed(act_key):
                actions.append(
                    InvestigationAction(
                        tool_id="ocr_extraction",
                        reason="Extract visible text, scripts, and road signage designations across image variants.",
                        priority=9.5,
                        expected_info_gain=0.85,
                        cost_estimate=0.15,
                        is_external_request=False,
                    )
                )

        # GAP 3b: Visual Place Search (VPR) Reference Matching
        if not state.clues.get("vpr_executed", False):
            act_key = "tool:visual_place_search"
            if not state.is_action_already_executed(act_key):
                actions.append(
                    InvestigationAction(
                        tool_id="visual_place_search",
                        reason="Query georeferenced reference index using global visual descriptors to discover candidate locations.",
                        priority=8.9,
                        expected_info_gain=0.85,
                        cost_estimate=0.12,
                        is_external_request=False,
                    )
                )

        # GAP 4: Infrastructure Detection (Poles, Catenary, Markings)
        if not state.clues["infra_executed"]:
            act_key = "tool:infrastructure_detection"
            if not state.is_action_already_executed(act_key):
                actions.append(
                    InvestigationAction(
                        tool_id="infrastructure_detection",
                        reason="Detect overhead powerlines, catenary masts, and road markings.",
                        priority=7.0,
                        expected_info_gain=0.6,
                        cost_estimate=0.10,
                        is_external_request=False,
                    )
                )

        # GAP 5: Solar Ephemeris & Shadow Extraction
        if not state.clues["solar_executed"]:
            act_key = "tool:solar_shadow"
            if not state.is_action_already_executed(act_key):
                actions.append(
                    InvestigationAction(
                        tool_id="solar_shadow",
                        reason="Compute astronomical solar angles and extract cast ground shadows.",
                        priority=7.5,
                        expected_info_gain=0.7,
                        cost_estimate=0.05,
                        is_external_request=False,
                    )
                )

        # GAP 6: Geographic Candidate Generation
        # Triggered once visual intake has run and candidates haven't been synthesized
        if not state.clues["candidates_generated"] and (state.clues["ocr_executed"] or state.clues["has_gps"] or state.clues.get("has_vpr_match", False)):
            act_key = "tool:candidate_generation"
            if not state.is_action_already_executed(act_key):
                actions.append(
                    InvestigationAction(
                        tool_id="candidate_generation",
                        reason="Synthesize visual clues into multi-scale geographic candidate hypotheses.",
                        priority=9.6,
                        expected_info_gain=0.85,
                        cost_estimate=0.20,
                        is_external_request=False,
                    )
                )

        # GAP 7: Adversarial Verification on Leading Hypotheses (only after candidates generated)
        if state.clues.get("candidates_generated", False):
            active_candidates = [h for h in state.hypotheses.values() if h.status in ("proposed", "active", "supported")]
            active_candidates.sort(key=lambda h: h.heuristic_score, reverse=True)

            for cand in active_candidates[:3]:  # Top 3 candidates
                if cand.lat is not None and cand.lon is not None:
                    # 7a. Solar Feasibility Falsification (high priority if shadow clue exists)
                    if state.clues.get("has_solar_shadow", False):
                        solar_key = f"adversarial_solar_check:{cand.id}"
                        if not state.is_action_already_executed(solar_key):
                            actions.append(
                                InvestigationAction(
                                    tool_id="adversarial_solar_check",
                                    params={"candidate_id": cand.id, "lat": cand.lat},
                                    reason=f"Falsification test: check if candidate '{cand.name}' latitude violates solar ceiling.",
                                    priority=9.8,
                                    expected_info_gain=0.9,
                                    target_hypothesis_id=cand.id,
                                    cost_estimate=0.02,
                                    is_external_request=False,
                                )
                            )

                    # 7b. OSM Highway Corridor Verification
                    osm_key = f"osm_road_search:{cand.id}"
                    if not state.is_action_already_executed(osm_key) and not state.config.offline_mode:
                        actions.append(
                            InvestigationAction(
                                tool_id="osm_road_search",
                                params={"lat": cand.lat, "lon": cand.lon, "radius_m": 2500.0},
                                reason=f"Verify road network geometry near candidate '{cand.name}' via OpenStreetMap.",
                                priority=9.5,
                                expected_info_gain=0.85,
                                target_hypothesis_id=cand.id,
                                cost_estimate=0.10,
                                is_external_request=True,
                            )
                        )

                    # 7c. DEM 360° Terrain Horizon Profiling
                    dem_key = f"dem_horizon_profile:{cand.id}"
                    if not state.is_action_already_executed(dem_key) and not state.config.offline_mode:
                        actions.append(
                            InvestigationAction(
                                tool_id="dem_horizon_profile",
                                params={"lat": cand.lat, "lon": cand.lon, "max_distance_m": 20000.0},
                                reason=f"Synthesize 360° synthetic terrain horizon profile for candidate '{cand.name}'.",
                                priority=7.8,
                                expected_info_gain=0.7,
                                target_hypothesis_id=cand.id,
                                cost_estimate=0.30,
                                is_external_request=True,
                            )
                        )

                    # 7d. Mountain Ridge Matching (if skyline is observed)
                    if state.clues.get("has_skyline", False):
                        ridge_key = f"ridge_matching:{cand.id}"
                        if not state.is_action_already_executed(ridge_key):
                            actions.append(
                                InvestigationAction(
                                    tool_id="ridge_matching",
                                    params={"candidate_id": cand.id, "lat": cand.lat, "lon": cand.lon},
                                    reason=f"Correlate observed mountain skyline with DEM horizon for candidate '{cand.name}'.",
                                    priority=8.5,
                                    expected_info_gain=0.8,
                                    target_hypothesis_id=cand.id,
                                    cost_estimate=0.15,
                                    is_external_request=False,
                                )
                            )

                    # 7e. Road Geometry & Curvature Alignment
                    if cand.level in ("road", "area") or "corridor" in cand.name.lower() or state.clues.get("has_ocr_road", False):
                        road_key = f"road_geometry_matching:{cand.id}"
                        if not state.is_action_already_executed(road_key):
                            actions.append(
                                InvestigationAction(
                                    tool_id="road_geometry_matching",
                                    params={"candidate_id": cand.id, "corridor_name": cand.name},
                                    reason=f"Align visual road curvature and lane geometry against corridor '{cand.name}'.",
                                    priority=8.7,
                                    expected_info_gain=0.75,
                                    target_hypothesis_id=cand.id,
                                    cost_estimate=0.10,
                                    is_external_request=False,
                                )
                            )

                    # 7f. Advanced Photographic Skyline Profiling
                    if state.clues.get("has_skyline", False):
                        adv_skyline_key = f"advanced_skyline_profiling:{cand.id}"
                        if not state.is_action_already_executed(adv_skyline_key):
                            actions.append(
                                InvestigationAction(
                                    tool_id="advanced_skyline_profiling",
                                    params={"candidate_id": cand.id, "lat": cand.lat, "lon": cand.lon},
                                    reason=f"Multi-parametric horizon cross-correlation (heading, roll, pitch) for '{cand.name}'.",
                                    priority=8.6,
                                    expected_info_gain=0.80,
                                    target_hypothesis_id=cand.id,
                                    cost_estimate=0.18,
                                    is_external_request=False,
                                )
                            )

                    # 7g. Spatial Envelope Refinement
                    env_key = f"spatial_envelope_refinement:{cand.id}"
                    if not state.is_action_already_executed(env_key):
                        actions.append(
                            InvestigationAction(
                                tool_id="spatial_envelope_refinement",
                                params={"candidate_id": cand.id},
                                reason=f"Refine spatial uncertainty envelope and multi-modal score for '{cand.name}'.",
                                priority=7.2,
                                expected_info_gain=0.50,
                                target_hypothesis_id=cand.id,
                                cost_estimate=0.05,
                                is_external_request=False,
                            )
                        )

        # GAP 8: Candidate Comparison & Re-ranking
        # If we have hypotheses and new evidence has been gathered since last comparison
        if state.hypotheses and state.clues.get("new_evidence_since_ranking", True):
            rank_key = f"candidate_comparison:iter_{state.iteration}"
            if not state.is_action_already_executed(rank_key):
                actions.append(
                    InvestigationAction(
                        tool_id="candidate_comparison",
                        reason="Re-score and re-rank competing hypotheses based on latest corroborations and checks.",
                        priority=6.5,
                        expected_info_gain=0.4,
                        cost_estimate=0.05,
                        is_external_request=False,
                    )
                )

        return actions

    def _compute_priority_score(self, action: InvestigationAction, state: InvestigationState) -> ActionPriorityScore:
        """Calculate mathematical composite priority for an action."""
        # 1. Base uncertainty reduction
        delta_u = action.expected_info_gain

        # 2. Source reliability (EXIF > Solar Math > OCR > Heuristic Visuals)
        rel_map = {
            "metadata_exif": 0.98,
            "adversarial_solar_check": 0.95,
            "solar_shadow": 0.90,
            "dem_horizon_profile": 0.88,
            "osm_road_search": 0.95,
            "ocr_extraction": 0.85,
            "visual_place_search": 0.90,
            "road_geometry_matching": 0.85,
            "advanced_skyline_profiling": 0.88,
            "ridge_matching": 0.80,
            "spatial_envelope_refinement": 0.70,
            "infrastructure_detection": 0.70,
            "candidate_generation": 0.75,
            "image_preprocessing": 0.60,
            "candidate_comparison": 0.50,
        }
        r = rel_map.get(action.tool_id, 0.70)

        # 3. Evidence independence
        i = 1.0  # Default to independent unless marked correlated

        # 4. Falsification potential: higher weight if it can definitively falsify
        f = 2.0 if action.tool_id.startswith("adversarial_") else 1.0

        # 5. Cost and time penalty
        c = max(0.05, action.cost_estimate)
        if action.is_external_request and state.config.offline_mode:
            c += 10.0  # Massive penalty if offline

        # Priority formula: (ΔU * R * I * F) / (C + 0.1)
        composite = (delta_u * r * i * f) / (c + 0.1) * 2.5

        return ActionPriorityScore(
            action=action,
            uncertainty_reduction=delta_u,
            source_reliability=r,
            evidence_independence=i,
            falsification_potential=f,
            cost_penalty=c,
            composite_priority=composite,
        )

    def _consult_llm(
        self,
        state: InvestigationState,
        candidate_actions: List[InvestigationAction],
    ) -> Optional[InvestigationAction]:
        """Consult optional LLM provider for contextual action prioritization."""
        try:
            obs_summary = "\n".join(f"• [{e.kind}] {e.text}" for e in list(state.evidence_items.values())[:10])
            hyp_summary = "\n".join(
                f"• {h.name} (Level: {h.level}, Score: {h.heuristic_score:.2f}, Status: {h.status})"
                for h in list(state.hypotheses.values())[:5]
            ) or "None generated yet."
            tools_summary = ", ".join(self.tools._tools.keys())
            exec_summary = ", ".join(state.executed_action_keys) or "None."

            proposal = self.llm.plan_next_action(
                observations_summary=obs_summary,
                active_hypotheses_summary=hyp_summary,
                available_tools_summary=tools_summary,
                executed_actions_summary=exec_summary,
            )
            if not proposal:
                return None

            # Verify target tool exists in candidate actions
            matched = next((a for a in candidate_actions if a.tool_id == proposal.target_tool_id), None)
            if matched:
                matched.reason = f"[LLM Prioritized] {proposal.rationale}"
                return matched
        except Exception as e:
            logger.debug(f"LLM consultation skipped: {e}")

        return None
