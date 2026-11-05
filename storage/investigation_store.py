"""HOUND Investigation Store.

Manages investigation lifecycles, file artifacts, and database records.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from core.image_loader import ImageRecord
from storage.database import DatabaseManager
from core.logger import get_logger

logger = get_logger("hound.investigation_store")


class InvestigationRecord(BaseModel):
    id: str
    sha256: str
    filename: str
    original_path: Path
    preserved_path: Optional[Path] = None
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    summary: Optional[str] = None
    workspace_dir: Path


class InvestigationStore:
    """Coordinates filesystem workspace and database records for an investigation."""

    def __init__(self, db_manager: DatabaseManager, storage_dir: Path | str):
        self.db = db_manager
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def create_investigation(self, image_record: ImageRecord) -> InvestigationRecord:
        """Initialize a new investigation session for an ingested image."""
        investigation_id = f"inv_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        workspace = self.storage_dir / "investigations" / investigation_id
        for sub in ["intake", "crops", "evidence", "reports", "tiles"]:
            (workspace / sub).mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc)
        record = InvestigationRecord(
            id=investigation_id,
            sha256=image_record.sha256,
            filename=image_record.filename,
            original_path=image_record.original_path,
            preserved_path=image_record.preserved_path,
            status="initialized",
            started_at=now,
            workspace_dir=workspace,
        )

        with self.db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO investigations (id, sha256, filename, original_path, preserved_path, status, started_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.sha256,
                    record.filename,
                    str(record.original_path),
                    str(record.preserved_path) if record.preserved_path else None,
                    record.status,
                    record.started_at.isoformat(),
                ),
            )

            # Store metadata record
            gps_lat = image_record.exif.gps.latitude if image_record.exif.gps else None
            gps_lon = image_record.exif.gps.longitude if image_record.exif.gps else None
            alt = image_record.exif.gps.altitude_m if image_record.exif.gps else None
            heading = image_record.exif.gps.heading_deg if image_record.exif.gps else None
            ts_orig = (
                image_record.exif.datetime_original.isoformat()
                if image_record.exif.datetime_original
                else None
            )

            conn.execute(
                """
                INSERT INTO metadata_records (
                    investigation_id, format, width, height, has_exif, camera_make, camera_model,
                    datetime_original, gps_lat, gps_lon, altitude_m, heading_deg, integrity_status, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    image_record.format,
                    image_record.width,
                    image_record.height,
                    1 if image_record.exif.has_exif else 0,
                    image_record.exif.make,
                    image_record.exif.model,
                    ts_orig,
                    gps_lat,
                    gps_lon,
                    alt,
                    heading,
                    image_record.integrity.status,
                    image_record.model_dump_json(),
                ),
            )
            conn.commit()

        self.db.log_audit(
            investigation_id,
            "investigation_created",
            {"sha256": image_record.sha256, "filename": image_record.filename},
        )
        logger.info(f"Created investigation {investigation_id} in {workspace}")
        return record

    def get_investigation(self, investigation_id: str) -> Optional[InvestigationRecord]:
        """Fetch an investigation record by ID."""
        with self.db.get_connection() as conn:
            cur = conn.execute("SELECT * FROM investigations WHERE id = ?", (investigation_id,))
            row = cur.fetchone()
            if not row:
                return None

            workspace = self.storage_dir / "investigations" / investigation_id
            return InvestigationRecord(
                id=row["id"],
                sha256=row["sha256"],
                filename=row["filename"],
                original_path=Path(row["original_path"]),
                preserved_path=Path(row["preserved_path"]) if row["preserved_path"] else None,
                status=row["status"],
                started_at=datetime.fromisoformat(row["started_at"]),
                completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
                summary=row["summary"],
                workspace_dir=workspace,
            )

    def update_status(self, investigation_id: str, status: str, summary: Optional[str] = None) -> None:
        """Update an investigation's completion status."""
        now = datetime.now(timezone.utc).isoformat() if status in ("completed", "failed") else None
        with self.db.get_connection() as conn:
            conn.execute(
                "UPDATE investigations SET status = ?, completed_at = ?, summary = ? WHERE id = ?",
                (status, now, summary, investigation_id),
            )
            conn.commit()

        self.db.log_audit(
            investigation_id,
            "status_updated",
            {"status": status, "summary": summary},
        )

    def list_investigations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List past investigations."""
        with self.db.get_connection() as conn:
            cur = conn.execute(
                "SELECT id, sha256, filename, status, started_at, completed_at, summary FROM investigations ORDER BY started_at DESC LIMIT ?",
                (limit,),
            )
            return [dict(row) for row in cur.fetchall()]
