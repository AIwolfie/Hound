"""HOUND LLM Reasoning Provider Interface.

Provides an optional, provider-independent interface for LLM-assisted
action prioritization and hypothesis challenge formulation, with guaranteed
offline deterministic fallback.
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from agent.agent_models import InvestigationAction, VerificationQuestion
from core.configuration import AgentConfig, get_config
from core.logger import get_logger

logger = get_logger("hound.agent.llm")


class LLMPlanProposal(BaseModel):
    """Structured plan proposal output by LLM reasoning."""
    selected_action: str
    target_tool_id: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    rationale: str
    expected_information_gain: float = 0.5
    falsification_questions: List[str] = Field(default_factory=list)


class BaseLLMProvider(ABC):
    """Abstract interface for LLM reasoning providers."""

    @abstractmethod
    def plan_next_action(
        self,
        observations_summary: str,
        active_hypotheses_summary: str,
        available_tools_summary: str,
        executed_actions_summary: str,
    ) -> Optional[LLMPlanProposal]:
        """Propose the next investigation action and verification challenges."""
        pass


class OfflineFallbackLLM(BaseLLMProvider):
    """Default deterministic offline provider requiring zero API keys and zero network."""

    def plan_next_action(
        self,
        observations_summary: str,
        active_hypotheses_summary: str,
        available_tools_summary: str,
        executed_actions_summary: str,
    ) -> Optional[LLMPlanProposal]:
        # Offline fallback yields None so deterministic rule-based planner handles selection
        return None


class GenericHTTPLLM(BaseLLMProvider):
    """Standard OpenAI/Gemini/generic HTTP provider with structured JSON outputs."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.api_key = config.llm_api_key
        self.model = config.llm_model
        self.base_url = config.llm_base_url or "https://api.openai.com/v1"
        self.timeout = config.llm_timeout_seconds

    def plan_next_action(
        self,
        observations_summary: str,
        active_hypotheses_summary: str,
        available_tools_summary: str,
        executed_actions_summary: str,
    ) -> Optional[LLMPlanProposal]:
        if not self.api_key:
            logger.debug("No LLM API key provided; using offline fallback.")
            return None

        prompt = (
            "You are HOUND's investigative reasoning module. Based ONLY on the verified observations below, "
            "propose the single most informative next action to advance the geolocation or actively challenge a hypothesis.\n\n"
            f"OBSERVATIONS:\n{observations_summary}\n\n"
            f"ACTIVE HYPOTHESES:\n{active_hypotheses_summary}\n\n"
            f"AVAILABLE TOOLS:\n{available_tools_summary}\n\n"
            f"ALREADY EXECUTED ACTIONS:\n{executed_actions_summary}\n\n"
            "Return valid JSON matching this schema:\n"
            "{\n"
            '  "selected_action": "<short description>",\n'
            '  "target_tool_id": "<exact tool_id from available tools>",\n'
            '  "parameters": {},\n'
            '  "rationale": "<reasoning based strictly on evidence>",\n'
            '  "expected_information_gain": <float 0.0 to 1.0>,\n'
            '  "falsification_questions": ["<question that could disprove leading hypothesis>"]\n'
            "}"
        )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a disciplined forensic OSINT agent. Never invent evidence. Return pure JSON."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
        }

        url = f"{self.base_url.rstrip('/')}/chat/completions"

        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                if response.status == 200:
                    body = json.loads(response.read().decode("utf-8"))
                    content_str = body["choices"][0]["message"]["content"]
                    parsed = json.loads(content_str)
                    return LLMPlanProposal.model_validate(parsed)
        except Exception as e:
            logger.warning(f"LLM reasoning request failed ({e}); gracefully falling back to deterministic planner.")
            return None

        return None


def get_llm_provider(config: Optional[AgentConfig] = None) -> BaseLLMProvider:
    """Factory creating configured LLM provider with fallback."""
    cfg = config or get_config().agent
    if not cfg.enable_llm or cfg.offline_mode or cfg.llm_provider in ("none", "", None):
        return OfflineFallbackLLM()

    if cfg.llm_provider in ("openai", "generic_http"):
        return GenericHTTPLLM(cfg)

    return OfflineFallbackLLM()
