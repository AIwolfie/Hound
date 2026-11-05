"""HOUND Evidence Visualizer.

Generates forensic image overlays, OCR bounding box annotations, infrastructure
detection highlights, shadow vectors, and multi-variant comparison sheets.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

from core.image_preprocessor import VariantInfo
from core.logger import get_logger
from investigators.infrastructure.base_detector import DetectedFeature
from investigators.ocr.ocr_engine import OCRObservation
from investigators.shadows.solar_analyzer import ShadowCandidate

logger = get_logger("hound.reports.visualizer")


class EvidenceVisualizer:
    """Renders structured visual forensic overlays on investigative imagery."""

    CATEGORY_COLORS = {
        "street_name": (0, 215, 255),       # Gold / Amber
        "road_number": (0, 255, 127),       # Spring Green
        "railway_station": (255, 165, 0),   # Orange
        "transport_authority": (30, 144, 255),  # Dodger Blue
        "telephone_number": (238, 130, 238),# Violet
        "postal_code": (255, 105, 180),     # Hot Pink
        "abbreviation": (175, 238, 238),    # Pale Turquoise
        "unknown": (220, 220, 220),         # Light Gray
    }

    INFRA_COLORS = {
        "utility_pole": (0, 255, 255),      # Cyan
        "road_sign_panel": (255, 215, 0),   # Gold
        "road_marking": (50, 205, 50),      # Lime Green
        "transmission_tower": (255, 69, 0), # Red-Orange
    }

    @classmethod
    def draw_ocr_overlay(
        cls,
        image_input: Path | Image.Image,
        observations: List[OCRObservation],
        output_path: Path,
    ) -> Path:
        """Render color-coded OCR bounding boxes with category pills onto the image."""
        if isinstance(image_input, Path):
            img = Image.open(image_input).convert("RGB")
        else:
            img = image_input.copy().convert("RGB")

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        font = None
        try:
            # Try default or standard scalable font if available
            font = ImageFont.load_default()
        except Exception:
            pass

        for obs in observations:
            x0, y0, x1, y1 = obs.box_original
            rgb = cls.CATEGORY_COLORS.get(obs.category, (200, 200, 200))
            rgba_border = rgb + (220,)
            rgba_fill = rgb + (40,)

            # Draw semi-transparent filled box
            draw.rectangle([x0, y0, x1, y1], outline=rgba_border, fill=rgba_fill, width=2)

            # Draw category and text tag above the box
            label = f"[{obs.category[:10]}] {obs.text[:20]}"
            tag_y0 = max(0.0, y0 - 16.0)
            tag_x1 = min(float(img.width), x0 + len(label) * 7.0 + 8.0)
            draw.rectangle([x0, tag_y0, tag_x1, y0], fill=rgb + (200,))
            draw.text((x0 + 4, tag_y0 + 2), label, fill=(0, 0, 0, 255), font=font)

        out_img = Image.alpha_composite(img.convert("RGBA"), overlay)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_img.convert("RGB").save(output_path, format="JPEG", quality=92)
        logger.debug(f"Saved OCR overlay to {output_path}")
        return output_path

    @classmethod
    def draw_infrastructure_overlay(
        cls,
        image_input: Path | Image.Image,
        detections: List[DetectedFeature],
        output_path: Path,
    ) -> Path:
        """Render detected physical infrastructure bounding boxes with confidence scores."""
        if isinstance(image_input, Path):
            img = Image.open(image_input).convert("RGB")
        else:
            img = image_input.copy().convert("RGB")

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        font = None
        try:
            font = ImageFont.load_default()
        except Exception:
            pass

        for feat in detections:
            x0, y0, x1, y1 = feat.box
            rgb = cls.INFRA_COLORS.get(feat.class_name, (0, 255, 255))
            rgba_border = rgb + (240,)
            rgba_fill = rgb + (35,)

            draw.rectangle([x0, y0, x1, y1], outline=rgba_border, fill=rgba_fill, width=2)

            label = f"{feat.class_name.replace('_', ' ')}: {feat.confidence:.2f}"
            tag_y0 = max(0.0, y0 - 16.0)
            tag_x1 = min(float(img.width), x0 + len(label) * 7.0 + 8.0)
            draw.rectangle([x0, tag_y0, tag_x1, y0], fill=rgb + (220,))
            draw.text((x0 + 4, tag_y0 + 2), label, fill=(0, 0, 0, 255), font=font)

        out_img = Image.alpha_composite(img.convert("RGBA"), overlay)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_img.convert("RGB").save(output_path, format="JPEG", quality=92)
        logger.debug(f"Saved Infrastructure overlay to {output_path}")
        return output_path

    @classmethod
    def draw_shadow_overlay(
        cls,
        image_input: Path | Image.Image,
        shadows: List[ShadowCandidate],
        output_path: Path,
    ) -> Path:
        """Render primary candidate shadow vector orientations onto the image."""
        if isinstance(image_input, Path):
            img = Image.open(image_input).convert("RGB")
        else:
            img = image_input.copy().convert("RGB")

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for s in shadows[:10]:
            cx, cy = s.center_point
            rad = math.radians(s.orientation_deg)
            half_len = min(60.0, s.length_px / 2.0)
            dx = half_len * math.cos(rad)
            dy = half_len * math.sin(rad)

            # Draw directional axis line
            p0 = (cx - dx, cy - dy)
            p1 = (cx + dx, cy + dy)
            draw.line([p0, p1], fill=(255, 50, 50, 220), width=3)
            # Center anchor dot
            draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(255, 255, 0, 255))

        out_img = Image.alpha_composite(img.convert("RGBA"), overlay)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_img.convert("RGB").save(output_path, format="JPEG", quality=92)
        logger.debug(f"Saved Shadow overlay to {output_path}")
        return output_path

    @classmethod
    def generate_comparison_panel(
        cls,
        variants: List[VariantInfo],
        output_path: Path,
        thumb_size: Tuple[int, int] = (640, 480),
    ) -> Path:
        """Create a multi-image comparison contact sheet for visual inspection."""
        selected_names = ["full_normalized", "contrast_enhanced", "clahe", "edge_enhanced", "sharpened"]
        items = [v for v in variants if v.name in selected_names]
        if not items:
            items = variants[:4]

        cols = 2
        rows = int(math.ceil(len(items) / float(cols)))
        tw, th = thumb_size
        padding = 10
        header_h = 30

        canvas_w = cols * tw + (cols + 1) * padding
        canvas_h = rows * (th + header_h) + (rows + 1) * padding
        canvas = Image.new("RGB", (canvas_w, canvas_h), color=(30, 30, 30))
        draw = ImageDraw.Draw(canvas)

        for idx, item in enumerate(items):
            r = idx // cols
            c = idx % cols
            x = padding + c * (tw + padding)
            y = padding + r * (th + header_h + padding)

            # Draw header
            label = f"{item.name.upper()}: {item.description[:45]}"
            draw.text((x + 4, y + 6), label, fill=(220, 220, 220))

            # Load and thumbnail image
            try:
                with Image.open(item.file_path) as thumb:
                    thumb_rgb = thumb.convert("RGB")
                    thumb_rgb.thumbnail((tw, th), Image.Resampling.BILINEAR)
                    canvas.paste(thumb_rgb, (x, y + header_h))
            except Exception as e:
                logger.warning(f"Failed to paste thumbnail {item.name}: {e}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path, format="JPEG", quality=90)
        logger.debug(f"Saved comparison panel to {output_path}")
        return output_path
