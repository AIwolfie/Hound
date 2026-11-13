"""HOUND Modular Visual Place Recognition (VPR) Base Interface.

Provides a unified, strict contract for all global and local visual place recognition models:
- Enforces explicit model provenance and licensing documentation.
- Supports lazy loading, CUDA execution, and CPU fallback.
- Enforces strict failure handling: never silently substitutes a failed model.
- Tracks inference latency, GPU/RAM memory consumption, and execution telemetry.
"""
from __future__ import annotations

import gc
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from core.logger import get_logger

logger = get_logger("hound.vpr.base")


class VPRModelError(Exception):
    """Base exception for all visual place recognition model errors."""
    pass


class ModelUnavailableError(VPRModelError):
    """Raised when a requested VPR model or its required weights cannot be located or loaded."""
    pass


class VPRExtractionError(VPRModelError):
    """Raised when feature extraction fails on an input image without silent substitution."""
    pass


@dataclass
class ModelProvenance:
    """Documented academic, scientific, and legal provenance of a VPR model."""
    model_name: str
    paper_title: str
    paper_url: str
    authors: str
    year: int
    repository_url: str
    license_type: str
    training_dataset: str
    weights_source: str
    descriptor_dim: int
    default_input_resolution: Tuple[int, int]  # (height, width)


@dataclass
class VPRTelemetry:
    """Telemetry recorded during model extraction or lifecycle operations."""
    model_name: str
    device: str
    extraction_time_ms: float
    memory_consumption_mb: float
    output_dimension: int
    is_l2_normalized: bool
    timestamp: float = field(default_factory=time.time)


class BaseVPRModel(ABC):
    """Abstract base class for all HOUND visual place recognition models."""

    def __init__(self, use_cuda: bool = True):
        self.use_cuda = use_cuda
        self.device = "cpu"
        self._is_loaded = False
        self._last_telemetry: Optional[VPRTelemetry] = None

    @property
    @abstractmethod
    def provenance(self) -> ModelProvenance:
        """Return the complete documented provenance and license metadata for this model."""
        pass

    @property
    def model_name(self) -> str:
        return self.provenance.model_name

    @property
    def descriptor_dim(self) -> int:
        return self.provenance.descriptor_dim

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def last_telemetry(self) -> Optional[VPRTelemetry]:
        return self._last_telemetry

    @abstractmethod
    def _load_model(self) -> None:
        """Internal routine to load weights and initialize model on device."""
        pass

    def load(self) -> None:
        """Explicitly load model weights into memory (lazy loading entrypoint)."""
        if not self._is_loaded:
            logger.info(f"Loading VPR model '{self.model_name}'...")
            t0 = time.perf_counter()
            self._load_model()
            self._is_loaded = True
            load_duration = (time.perf_counter() - t0) * 1000
            logger.info(f"VPR model '{self.model_name}' loaded in {load_duration:.1f} ms on device '{self.device}'.")

    def unload(self) -> None:
        """Unload model from GPU/RAM to respect host memory limits."""
        if self._is_loaded:
            logger.info(f"Unloading VPR model '{self.model_name}' from memory...")
            self._unload_model()
            self._is_loaded = False
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info(f"VPR model '{self.model_name}' unloaded successfully.")

    @abstractmethod
    def _unload_model(self) -> None:
        """Internal routine to release model weights and tensors."""
        pass

    @abstractmethod
    def _extract(self, image_rgb: np.ndarray) -> np.ndarray:
        """Extract descriptor from preprocessed RGB image."""
        pass

    def extract_descriptor(self, image_np: np.ndarray) -> np.ndarray:
        """Extract a fixed-dimension, L2-normalized feature vector from an RGB image.

        Ensures lazy loading, records telemetry, and strictly forbids silent substitution.
        """
        if image_np is None or image_np.size == 0:
            raise VPRExtractionError(f"Model '{self.model_name}': Input image is empty or None.")

        # Ensure model is loaded (lazy loading)
        if not self._is_loaded:
            self.load()

        # Format image to standard RGB
        if len(image_np.shape) == 2:
            img_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:
            img_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        elif image_np.shape[2] == 3:
            img_rgb = image_np
        else:
            raise VPRExtractionError(f"Model '{self.model_name}': Unsupported image shape {image_np.shape}.")

        t0 = time.perf_counter()
        try:
            vec = self._extract(img_rgb)
        except Exception as e:
            # Strictly do NOT substitute with another model; re-raise as VPRExtractionError
            raise VPRExtractionError(f"Model '{self.model_name}' extraction failed: {e}") from e

        extraction_ms = (time.perf_counter() - t0) * 1000

        # Validate vector integrity
        if not isinstance(vec, np.ndarray):
            raise VPRExtractionError(f"Model '{self.model_name}': Expected numpy.ndarray, got {type(vec)}.")
        if vec.shape != (self.descriptor_dim,):
            raise VPRExtractionError(
                f"Model '{self.model_name}': Extracted shape {vec.shape} does not match expected ({self.descriptor_dim},)."
            )

        # Enforce exact L2 normalization
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-7:
            raise VPRExtractionError(f"Model '{self.model_name}': Extracted zero-norm descriptor vector.")
        vec_normed = (vec / norm).astype(np.float32)

        # Record telemetry
        mem_mb = self.get_memory_usage_mb()
        self._last_telemetry = VPRTelemetry(
            model_name=self.model_name,
            device=self.device,
            extraction_time_ms=round(extraction_ms, 2),
            memory_consumption_mb=round(mem_mb, 2),
            output_dimension=self.descriptor_dim,
            is_l2_normalized=True,
        )

        return vec_normed

    def get_memory_usage_mb(self) -> float:
        """Return estimated memory usage in megabytes for this model."""
        try:
            import torch
            if self.device == "cuda" and torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024.0 * 1024.0)
        except Exception:
            pass
        return 0.0
