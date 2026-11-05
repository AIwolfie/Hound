"""HOUND Database Layer.

Provides thread-safe SQLite storage for investigations, metadata, evidence, and candidates.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from core.logger import get_logger

logger = get_logger("hound.database")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS investigations (
    id TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL,
    filename TEXT NOT NULL,
    original_path TEXT NOT NULL,
    preserved_path TEXT,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    summary TEXT
);

CREATE TABLE IF NOT EXISTS metadata_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    investigation_id TEXT NOT NULL,
    format TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    has_exif INTEGER NOT NULL,
    camera_make TEXT,
    camera_model TEXT,
    datetime_original TEXT,
    gps_lat REAL,
    gps_lon REAL,
    altitude_m REAL,
    heading_deg REAL,
    integrity_status TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    FOREIGN KEY(investigation_id) REFERENCES investigations(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evidence_items (
    id TEXT PRIMARY KEY,
    investigation_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    text TEXT NOT NULL,
    confidence_weight REAL NOT NULL,
    file_path TEXT,
    resolves_id TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(investigation_id) REFERENCES investigations(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS candidates (
    id TEXT PRIMARY KEY,
    investigation_id TEXT NOT NULL,
    parent_id TEXT,
    level TEXT NOT NULL,
    name TEXT NOT NULL,
    lat REAL,
    lon REAL,
    radius_m REAL,
    score REAL NOT NULL DEFAULT 1.0,
    status TEXT NOT NULL DEFAULT 'active',
    falsification_condition TEXT,
    exclusion_reason TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(investigation_id) REFERENCES investigations(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS candidate_evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL,
    evidence_id TEXT NOT NULL,
    likelihood_ratio REAL NOT NULL,
    direction TEXT NOT NULL,
    why TEXT,
    FOREIGN KEY(candidate_id) REFERENCES candidates(id) ON DELETE CASCADE,
    FOREIGN KEY(evidence_id) REFERENCES evidence_items(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    investigation_id TEXT NOT NULL,
    action TEXT NOT NULL,
    details TEXT,
    timestamp TEXT NOT NULL,
    FOREIGN KEY(investigation_id) REFERENCES investigations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_investigations_sha ON investigations(sha256);
CREATE INDEX IF NOT EXISTS idx_evidence_investigation ON evidence_items(investigation_id);
CREATE INDEX IF NOT EXISTS idx_candidates_investigation ON candidates(investigation_id);
CREATE INDEX IF NOT EXISTS idx_audit_investigation ON audit_logs(investigation_id);
"""


class DatabaseManager:
    """Manages SQLite database connections and transactions."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with self.get_connection() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()
            logger.debug(f"Database schema initialized at {self.db_path}")

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
        finally:
            conn.close()

    def log_audit(self, investigation_id: str, action: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Record an immutable audit trail entry."""
        now = datetime.now(timezone.utc).isoformat()
        detail_json = json.dumps(details or {}, ensure_ascii=False)
        with self.get_connection() as conn:
            conn.execute(
                "INSERT INTO audit_logs (investigation_id, action, details, timestamp) VALUES (?, ?, ?, ?)",
                (investigation_id, action, detail_json, now),
            )
            conn.commit()
