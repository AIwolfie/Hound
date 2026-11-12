"""HOUND Adversarial Defense & Prompt Injection Sanitizer.

Defends HOUND's autonomous investigation pipeline against adversarial attacks
including:
1. Prompt injection attempts embedded in image OCR text (graffiti, signs, banners).
2. Reference database poisoning (out-of-bounds coordinates, corrupted descriptors).
3. Misleading and contradictory geographic signals.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from core.logger import get_logger

logger = get_logger("hound.adversarial")

# Common prompt injection patterns embedded in adversarial visual scenes
INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+)?(?:prior|previous|above)\s+instructions",
    r"system\s+(?:prompt|message|directive)",
    r"developer\s+(?:mode|override|instructions)",
    r"admin\s+(?:mode|override|access)",
    r"you\s+are\s+now\s+(?:a|an)?\s*[a-z0-9_-]+",
    r"disregard\s+(?:all\s+)?(?:prior|previous)\s+prompts?",
    r"output\s+only\s+(?:the\s+)?coordinates?",
    r"always\s+(?:output|reply|answer)\s+(?:with)?",
    r"jailbreak",
]
COMPILED_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


class AdversarialSanitizationResult(BaseModel):
    """Result of adversarial inspection on visual evidence text."""
    is_adversarial: bool
    cleaned_text: str
    attack_category: Optional[str] = None
    confidence_penalty: float = 0.0
    quarantine_reason: Optional[str] = None


class AdversarialTextSanitizer:
    """Detects and neutralizes adversarial prompt injection payloads inside OCR text."""

    @classmethod
    def inspect_text(cls, text: str) -> AdversarialSanitizationResult:
        if not text:
            return AdversarialSanitizationResult(is_adversarial=False, cleaned_text="")

        normalized = text.strip()
        for pattern in COMPILED_INJECTION_RE:
            match = pattern.search(normalized)
            if match:
                matched_str = match.group(0)
                logger.warning(f"Adversarial prompt injection pattern detected in visual text: '{matched_str}'")
                return AdversarialSanitizationResult(
                    is_adversarial=True,
                    cleaned_text=pattern.sub("[REDACTED_INJECTION_PAYLOAD]", normalized),
                    attack_category="prompt_injection",
                    confidence_penalty=0.95,  # Severely neutralize confidence
                    quarantine_reason=f"Detected adversarial prompt injection payload matching '{matched_str}'.",
                )

        return AdversarialSanitizationResult(
            is_adversarial=False,
            cleaned_text=normalized,
            confidence_penalty=0.0,
            quarantine_reason=None,
        )


class ReferencePoisoningGuard:
    """Validates georeferenced reference items against poisoning attacks."""

    @classmethod
    def validate_reference_item(cls, item: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        lat = item.get("lat")
        lon = item.get("lon")

        # 1. WGS-84 Coordinate bounds
        if lat is None or lon is None:
            return False, "Missing coordinates in reference item."
        if not (-90.0 <= lat <= 90.0):
            return False, f"Poisoned latitude: {lat} outside [-90, +90]."
        if not (-180.0 <= lon <= 180.0):
            return False, f"Poisoned longitude: {lon} outside [-180, +180]."

        # 2. Null island check (0.0, 0.0) without explicit ocean tag
        if abs(lat) < 1e-5 and abs(lon) < 1e-5:
            tags = item.get("tags", [])
            if "null_island" not in tags and "gulf_of_guinea" not in tags:
                return False, "Suspicious (0.0, 0.0) Null Island coordinates."

        # 3. Descriptor dimension check
        desc = item.get("descriptor")
        if desc is not None and len(desc) not in (512, 1224, 2048):
            return False, f"Corrupted descriptor dimension: {len(desc)}."

        # 4. Uncertainty radius sanity
        uncertainty_m = item.get("uncertainty_radius_m", item.get("uncertainty_m", 1000.0))
        if uncertainty_m <= 0 or uncertainty_m > 20_000_000:
            return False, f"Invalid spatial uncertainty radius: {uncertainty_m} m."

        return True, None
