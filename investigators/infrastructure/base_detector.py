"""HOUND Base Infrastructure Detector Interface.

Defines the contract for visual infrastructure and geometric feature detectors.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field


class DetectedFeature(BaseModel):
    class_name: str
    confidence: float
    box: Tuple[float, float, float, float]  # (x0, y0, x1, y1) in image pixel coordinates
    detector_backend: str
    attributes: Dict[str, Any] = Field(default_factory=dict)
    description: str = ""


class BaseInfrastructureDetector(ABC):
    """Abstract interface for infrastructure detection backends."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def detect(self, image_input: Path | np.ndarray) -> List[DetectedFeature]:
        """Detect infrastructure elements and return structured features."""
        pass
