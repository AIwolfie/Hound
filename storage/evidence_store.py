"""HOUND Evidence Store.

Manages evidence items, hypothesis candidates, likelihood updates, and audit logging.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from storage.database import DatabaseManager
from core.logger import get_logger

logger = get_logger("hound.evidence_store")


class EvidenceItem(BaseModel):
    id: str = Field(default_factory=lambda: f"E{uuid.uuid4().hex[:6].upper()}")
    investigation_id: str
    kind: str  # "exif", "ocr", "visual", "sun", "terrain", "osm", "street", "clue"
    status: str  # "read", "observed", "inferred", "computed"
    text: str
    confidence_weight: float = 1.0  # neutral=1.0, supportive > 1.0, contradictory < 1.0
    file_path: Optional[Path] = None
    resolves_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CandidateItem(BaseModel):
    id: str = Field(default_factory=lambda: f"C{uuid.uuid4().hex[:6].upper()}")
    investigation_id: str
    parent_id: Optional[str] = None
    level: str  # "country", "admin", "city", "area", "road", "point"
    name: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    radius_m: Optional[float] = None
    score: float = 1.0
    status: str = "active"  # "active", "excluded", "confirmed", "downweighted"
    falsification_condition: Optional[str] = None
    exclusion_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EvidenceStore:
    """Stores and queries structured evidence and candidate hypotheses."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def add_evidence(self, evidence: EvidenceItem) -> EvidenceItem:
        """Add a new evidence item to the investigation ledger."""
        with self.db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO evidence_items (
                    id, investigation_id, kind, status, text, confidence_weight, file_path, resolves_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evidence.id,
                    evidence.investigation_id,
                    evidence.kind,
                    evidence.status,
                    evidence.text,
                    evidence.confidence_weight,
                    str(evidence.file_path) if evidence.file_path else None,
                    evidence.resolves_id,
                    evidence.created_at.isoformat(),
                ),
            )
            conn.commit()

        self.db.log_audit(
            evidence.investigation_id,
            "evidence_added",
            {
                "evidence_id": evidence.id,
                "kind": evidence.kind,
                "status": evidence.status,
                "text": evidence.text[:100],
            },
        )
        return evidence

    def get_evidence(self, investigation_id: str) -> List[EvidenceItem]:
        """Fetch all evidence recorded for an investigation."""
        with self.db.get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM evidence_items WHERE investigation_id = ? ORDER BY created_at ASC",
                (investigation_id,),
            )
            items = []
            for row in cur.fetchall():
                items.append(
                    EvidenceItem(
                        id=row["id"],
                        investigation_id=row["investigation_id"],
                        kind=row["kind"],
                        status=row["status"],
                        text=row["text"],
                        confidence_weight=row["confidence_weight"],
                        file_path=Path(row["file_path"]) if row["file_path"] else None,
                        resolves_id=row["resolves_id"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                    )
                )
            return items

    def add_candidate(self, candidate: CandidateItem) -> CandidateItem:
        """Register a new candidate hypothesis on the board."""
        with self.db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO candidates (
                    id, investigation_id, parent_id, level, name, lat, lon, radius_m, score,
                    status, falsification_condition, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.id,
                    candidate.investigation_id,
                    candidate.parent_id,
                    candidate.level,
                    candidate.name,
                    candidate.lat,
                    candidate.lon,
                    candidate.radius_m,
                    candidate.score,
                    candidate.status,
                    candidate.falsification_condition,
                    candidate.created_at.isoformat(),
                ),
            )
            conn.commit()

        self.db.log_audit(
            candidate.investigation_id,
            "candidate_added",
            {"candidate_id": candidate.id, "level": candidate.level, "name": candidate.name},
        )
        return candidate

    def link_evidence_to_candidate(
        self,
        candidate_id: str,
        evidence_id: str,
        likelihood_ratio: float,
        direction: str,  # "for" or "against"
        why: Optional[str] = None,
    ) -> None:
        """Apply likelihood ratio update to a candidate hypothesis based on evidence."""
        # Sanitize clamped bounds according to hard rules
        # Inferred: clamp [0.33, 3.0], Observed: clamp [0.2, 5.0], Computed: unrestricted
        with self.db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO candidate_evidence (candidate_id, evidence_id, likelihood_ratio, direction, why)
                VALUES (?, ?, ?, ?, ?)
                """,
                (candidate_id, evidence_id, likelihood_ratio, direction, why),
            )
            # Update candidate composite score
            conn.execute(
                "UPDATE candidates SET score = score * ? WHERE id = ?",
                (likelihood_ratio, candidate_id),
            )
            conn.commit()

    def get_candidates(self, investigation_id: str) -> List[CandidateItem]:
        """Fetch all candidates sorted by descending score."""
        with self.db.get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM candidates WHERE investigation_id = ? ORDER BY score DESC",
                (investigation_id,),
            )
            candidates = []
            for row in cur.fetchall():
                candidates.append(
                    CandidateItem(
                        id=row["id"],
                        investigation_id=row["investigation_id"],
                        parent_id=row["parent_id"],
                        level=row["level"],
                        name=row["name"],
                        lat=row["lat"],
                        lon=row["lon"],
                        radius_m=row["radius_m"],
                        score=row["score"],
                        status=row["status"],
                        falsification_condition=row["falsification_condition"],
                        exclusion_reason=row["exclusion_reason"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                    )
                )
            return candidates
