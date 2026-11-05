"""HOUND Base Investigator Contract.

Defines the abstract interface for all specialized analysis engines.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from core.image_loader import ImageRecord
from storage.evidence_store import EvidenceItem
from core.logger import get_logger


class BaseInvestigator(ABC):
    """Abstract base class for all investigative modules in HOUND."""

    def __init__(self, name: str, category: str):
        self.name = name
        self.category = category
        self.logger = get_logger(f"hound.investigators.{name}")

    @abstractmethod
    def run(
        self,
        investigation_id: str,
        image_record: ImageRecord,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[EvidenceItem]:
        """Execute investigative analysis and return generated evidence items.

        Args:
            investigation_id: Current investigation identifier.
            image_record: Ingested target image record.
            context: Additional runtime parameters or shared state.

        Returns:
            List of generated EvidenceItem instances.
        """
        pass
