"""HOUND Agent Data Models.

Defines typed Pydantic models for autonomous investigation planning,
tool selection, hypothesis lifecycle, verification questions, and event telemetry.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field

from agent.candidate_comparator import ScoreBreakdown
from core.geographic_search import BoundingBox
from storage.evidence_store import EvidenceItem


HypothesisStatus = Literal[
    "proposed",
    "active",
    "supported",
    "challenged",
    "rejected",
    "inconclusive",
]

StoppingReason = Literal[
    "sufficient_evidence",
    "paths_exhausted",
    "indistinguishable_candidates",
    "insufficient_evidence",
    "budget_reached",
    "service_unavailable",
    "manual_interruption",
]

FinalStatus = Literal[
    "localized",
    "region_identified",
    "candidates_remaining",
    "insufficient_evidence",
    "inconclusive",
    "investigation_limit_reached",
]

VerificationModality = Literal[
    "ocr",
    "infrastructure",
    "terrain",
    "geometry",
    "solar",
    "satellite",
    "geocoding",
]

VerificationStatus = Literal[
    "pending",
    "confirmed",
    "contradicted",
    "untested",
    "data_unavailable",
]


class InvestigationAction(BaseModel):
    """An action selected by the planner to be executed."""
    action_id: str = Field(default_factory=lambda: f"ACT-{uuid.uuid4().hex[:6].upper()}")
    tool_id: str
    params: Dict[str, Any] = Field(default_factory=dict)
    reason: str
    priority: float = 1.0
    expected_info_gain: float = 0.5
    target_hypothesis_id: Optional[str] = None
    target_modality: Optional[str] = None
    cost_estimate: float = 0.05
    is_external_request: bool = False


class ToolResult(BaseModel):
    """Standardized output returned by any tool execution."""
    tool_id: str
    action_id: Optional[str] = None
    success: bool
    status: Literal["success", "partial", "error", "skipped", "service_unavailable"] = "success"
    output_data: Dict[str, Any] = Field(default_factory=dict)
    evidence_items: List[EvidenceItem] = Field(default_factory=list)
    execution_time_s: float = 0.0
    error_message: Optional[str] = None
    cached: bool = False


class InvestigationEvent(BaseModel):
    """A single logged event in the investigation timeline."""
    event_id: str = Field(default_factory=lambda: f"EVT-{uuid.uuid4().hex[:6].upper()}")
    investigation_id: str
    iteration: int
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_type: str  # "tool_executed", "hypothesis_updated", "contradiction_found", "action_selected", "stopped"
    tool_id: Optional[str] = None
    action_summary: str
    result_summary: str
    evidence_ids: List[str] = Field(default_factory=list)
    duration_s: float = 0.0
    error: Optional[str] = None


class VerificationQuestion(BaseModel):
    """A targeted adversarial check challenging a candidate hypothesis."""
    question_id: str = Field(default_factory=lambda: f"VQ-{uuid.uuid4().hex[:6].upper()}")
    hypothesis_id: str
    question_text: str
    modality: VerificationModality
    target_check: str
    status: VerificationStatus = "pending"
    justification: str = ""
    evidence_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class HypothesisRecord(BaseModel):
    """A structured geographic candidate hypothesis maintained across the investigation lifecycle."""
    id: str
    investigation_id: str
    parent_id: Optional[str] = None
    level: str  # "point", "building", "road", "area", "city", "region"
    name: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    bbox: Optional[BoundingBox] = None
    radius_m: Optional[float] = None
    status: HypothesisStatus = "proposed"
    heuristic_score: float = 1.0
    score_breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    seed_evidence_ids: List[str] = Field(default_factory=list)
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    contradictory_evidence_ids: List[str] = Field(default_factory=list)
    untested_constraints: List[str] = Field(default_factory=list)
    verification_questions: List[VerificationQuestion] = Field(default_factory=list)
    completed_checks: List[str] = Field(default_factory=list)
    remaining_checks: List[str] = Field(default_factory=list)
    rejection_reason: Optional[str] = None
    uncertainty_description: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class FinalInvestigationDossier(BaseModel):
    """Comprehensive investigation report compiling evidence, facts, inferences, and conclusions."""
    investigation_id: str
    target_filename: str
    sha256_hash: str
    image_resolution: str
    final_status: FinalStatus
    stopping_reason: StoppingReason
    geographic_precision_justified: str  # "Point (<=100m)", "Building", "Road Corridor", "City/District", "Region", "Unlocalized"
    total_iterations: int
    total_tool_calls: int
    total_duration_s: float
    events_count: int
    primary_hypothesis: Optional[HypothesisRecord] = None
    competing_hypotheses: List[HypothesisRecord] = Field(default_factory=list)
    rejected_hypotheses: List[HypothesisRecord] = Field(default_factory=list)
    facts: List[str] = Field(default_factory=list)  # Direct verified observations (EXIF, OCR, hashes)
    inferences: List[str] = Field(default_factory=list)  # Model deductions & spatial correlates
    untested_constraints: List[str] = Field(default_factory=list)
    abstention_reasons: List[str] = Field(default_factory=list)
    data_sources_used: List[str] = Field(default_factory=list)
    known_limitations: List[str] = Field(default_factory=list)
    timeline: List[InvestigationEvent] = Field(default_factory=list)
