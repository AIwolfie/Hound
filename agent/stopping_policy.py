"""HOUND Investigation Stopping Policy & Bounded Execution.

Evaluates evidence convergence, budget limits, candidate separation,
and determines when the agent must conclude or honestly abstain.
"""
from __future__ import annotations

from typing import Optional, Tuple
from agent.agent_models import FinalStatus, HypothesisRecord, StoppingReason
from agent.investigation_state import InvestigationState
from core.logger import get_logger

logger = get_logger("hound.agent.stopping")


class StoppingPolicy:
    """Enforces execution limits and evaluates stopping conditions."""

    @classmethod
    def evaluate(cls, state: InvestigationState) -> Tuple[bool, Optional[StoppingReason], Optional[FinalStatus]]:
        """Evaluate whether the investigation loop should terminate.

        Returns:
            Tuple of (should_stop, stopping_reason, final_status)
        """
        # 1. Check for Weak Evidence / Honest Abstention (Negative Control)
        # If visual inspection (metadata, OCR, infra, solar) has run, but no clues found
        if (
            state.clues["metadata_analyzed"]
            and state.clues["ocr_executed"]
            and state.clues["infra_executed"]
            and state.clues["solar_executed"]
            and not state.clues["has_gps"]
            and not state.clues["has_ocr_road"]
            and not state.clues.get("has_skyline", False)
            and len(state.hypotheses) <= 1
        ):
            # Check if only an unlocalized or fallback candidate exists
            non_fallback = [
                h for h in state.hypotheses.values()
                if "unlocalized" not in h.name.lower() and h.status != "rejected"
            ]
            if not non_fallback:
                logger.info("Stopping condition: Weak evidence detected. Abstaining honestly.")
                return True, "insufficient_evidence", "insufficient_evidence"

        # 2. Check Hard Budgets
        budget_exceeded, reason = state.is_budget_exhausted()
        if budget_exceeded:
            logger.info(f"Stopping condition triggered: execution budget reached ({reason}).")
            # If we already have a strong candidate when budget hit, keep it categorized
            active = [h for h in state.hypotheses.values() if h.status in ("proposed", "active", "supported")]
            if active:
                return True, reason, "candidates_remaining"
            return True, reason, "insufficient_evidence" if not state.hypotheses else "investigation_limit_reached"

        # 3. Check for Strong Convergence (Sufficient Evidence)
        active_hypotheses = [
            h for h in state.hypotheses.values()
            if h.status in ("proposed", "active", "supported") and not h.contradictory_evidence_ids
        ]
        active_hypotheses.sort(key=lambda h: h.heuristic_score, reverse=True)

        if active_hypotheses:
            top = active_hypotheses[0]
            independent_count = len(top.supporting_evidence_ids)

            # Strong convergence criteria:
            # - At least 2 independent corroborations
            # - Heuristic score >= 18.0
            # - Clear lead or verified high-precision level
            if independent_count >= 2 and top.heuristic_score >= 18.0:
                if len(active_hypotheses) == 1:
                    is_dominant = True
                else:
                    runner_up = active_hypotheses[1]
                    is_dominant = top.heuristic_score >= runner_up.heuristic_score * 1.25

                if is_dominant:
                    logger.info(
                        f"Stopping condition: Sufficient evidence for '{top.name}' "
                        f"(Score: {top.heuristic_score:.2f}, Corroborations: {independent_count})."
                    )
                    status: FinalStatus = "localized" if top.level in ("point", "building", "road") else "region_identified"
                    return True, "sufficient_evidence", status

        # 4. Check for Indistinguishable Candidates
        # If multiple candidates have nearly identical scores and all verification checks are complete
        if len(active_hypotheses) >= 2:
            top1 = active_hypotheses[0]
            top2 = active_hypotheses[1]
            diff = abs(top1.heuristic_score - top2.heuristic_score)
            checks_done = (
                state.clues.get("terrain_checked", False)
                and state.clues.get("osm_searched", False)
            )
            if diff < 1.0 and checks_done:
                logger.info(f"Stopping condition: Candidates '{top1.name}' and '{top2.name}' are indistinguishable.")
                return True, "indistinguishable_candidates", "inconclusive"

        # Not ready to stop yet
        return False, None, None
