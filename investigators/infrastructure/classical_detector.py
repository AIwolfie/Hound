"""HOUND Classical Computer Vision Infrastructure Detector.

Implements lightweight, CPU-compatible geometric detection of utility poles,
road signage, lane markings, and structural lattice features using OpenCV.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np

from investigators.infrastructure.base_detector import BaseInfrastructureDetector, DetectedFeature
from core.logger import get_logger

logger = get_logger("hound.infrastructure.classical")


class ClassicalInfrastructureDetector(BaseInfrastructureDetector):
    """Detects utility poles, road signs, and road markings via geometric computer vision."""

    def __init__(self):
        super().__init__(name="opencv_classical_v1")

    def detect(self, image_input: Path | np.ndarray) -> List[DetectedFeature]:
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                logger.error(f"Failed to load image at {image_input}")
                return []
        else:
            img = image_input

        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features: List[DetectedFeature] = []

        # -------------------------------------------------------------
        # 1. Vertical Infrastructure: Utility Poles & Catenary Masts
        # -------------------------------------------------------------
        pole_features = self._detect_poles(gray, width, height)
        features.extend(pole_features)

        # -------------------------------------------------------------
        # 2. Road Signage: Rectangular, Circular, Triangular Panels
        # -------------------------------------------------------------
        sign_features = self._detect_sign_boards(gray, img, width, height)
        features.extend(sign_features)

        # -------------------------------------------------------------
        # 3. Road Markings & Lane Dividers (Lower Third of Frame)
        # -------------------------------------------------------------
        marking_features = self._detect_road_markings(img, width, height)
        features.extend(marking_features)

        return features

    def _detect_poles(self, gray: np.ndarray, width: int, height: int) -> List[DetectedFeature]:
        """Detect vertical utility poles and masts using Hough line clustering."""
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=int(height * 0.15),
            maxLineGap=int(height * 0.05),
        )
        if lines is None:
            return []

        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            # Filter nearly vertical lines (dy >> dx)
            if dy > 0 and (dx / float(dy)) < 0.15:
                top_y = min(y1, y2)
                bot_y = max(y1, y2)
                avg_x = (x1 + x2) / 2.0
                vertical_lines.append((avg_x, top_y, bot_y, dy))

        if not vertical_lines:
            return []

        # Cluster vertical lines by X coordinate proximity
        vertical_lines.sort(key=lambda item: item[0])
        clusters: List[List[Tuple[float, int, int, int]]] = []
        cluster_threshold_px = max(15, int(width * 0.02))

        for line in vertical_lines:
            if not clusters:
                clusters.append([line])
            else:
                last_cluster = clusters[-1]
                avg_cluster_x = sum(l[0] for l in last_cluster) / len(last_cluster)
                if abs(line[0] - avg_cluster_x) <= cluster_threshold_px:
                    last_cluster.append(line)
                else:
                    clusters.append([line])

        pole_detections: List[DetectedFeature] = []
        for c in clusters:
            # Require at least 2 consistent line segments
            if len(c) < 2:
                continue

            min_x = max(0, int(min(l[0] for l in c) - 10))
            max_x = min(width, int(max(l[0] for l in c) + 10))
            min_y = max(0, min(l[1] for l in c))
            max_y = min(height, max(l[2] for l in c))
            span_h = max_y - min_y
            span_w = max(1, max_x - min_x)

            # High vertical aspect ratio check
            if span_h >= int(height * 0.20) and (span_h / float(span_w)) >= 2.5:
                conf = min(0.85, 0.45 + (len(c) * 0.08))
                pole_detections.append(
                    DetectedFeature(
                        class_name="utility_pole",
                        confidence=round(conf, 2),
                        box=(float(min_x), float(min_y), float(max_x), float(max_y)),
                        detector_backend=self.name,
                        attributes={"aspect_ratio": round(span_h / float(span_w), 1), "line_count": len(c)},
                        description="Vertical linear structure consistent with utility pole or catenary mast.",
                    )
                )

        # Return top 5 most prominent pole detections
        pole_detections.sort(key=lambda d: (d.box[3] - d.box[1]), reverse=True)
        return pole_detections[:5]

    def _detect_sign_boards(
        self, gray: np.ndarray, img_bgr: np.ndarray, width: int, height: int
    ) -> List[DetectedFeature]:
        """Detect candidate road and traffic signs via geometric shape fitting."""
        # Focus on upper 80% of image where signs typically appear
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 120)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sign_features: List[DetectedFeature] = []

        min_sign_dim = max(16, int(min(width, height) * 0.03))
        max_sign_dim = int(min(width, height) * 0.35)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (min_sign_dim * min_sign_dim) or area > (max_sign_dim * max_sign_dim):
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w < min_sign_dim or h < min_sign_dim or w > max_sign_dim or h > max_sign_dim:
                continue

            # Don't look in extreme bottom corners (usually car hood/road close to camera)
            if y > int(height * 0.85):
                continue

            aspect_ratio = float(w) / float(h)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            vertices = len(approx)

            # Check for rectangle (4 vertices), triangle (3 vertices), or circle (high circularity)
            circularity = 4.0 * math.pi * (area / (peri * peri + 1e-9))
            is_rectangle = (vertices == 4) and (0.4 <= aspect_ratio <= 3.5)
            is_triangle = (vertices == 3) and (0.6 <= aspect_ratio <= 1.6)
            is_circle = (circularity > 0.70) and (0.75 <= aspect_ratio <= 1.3)

            if is_rectangle or is_triangle or is_circle:
                shape_label = "rectangle" if is_rectangle else ("triangle" if is_triangle else "circle")
                conf = 0.60 if is_rectangle else 0.70
                sign_features.append(
                    DetectedFeature(
                        class_name="road_sign_panel",
                        confidence=conf,
                        box=(float(x), float(y), float(x + w), float(y + h)),
                        detector_backend=self.name,
                        attributes={"shape": shape_label, "aspect_ratio": round(aspect_ratio, 2)},
                        description=f"Geometric signboard outline ({shape_label}) typical of regulatory or wayfinding signage.",
                    )
                )

        # Deduplicate overlapping sign boxes
        return self._non_max_suppress(sign_features, iou_thresh=0.40)[:8]

    def _detect_road_markings(self, img_bgr: np.ndarray, width: int, height: int) -> List[DetectedFeature]:
        """Detect roadway lane markings in the lower visual field."""
        # Process lower 45% of image
        lower_y = int(height * 0.55)
        roi = img_bgr[lower_y:height, 0:width]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # White lane line mask (high V, low S)
        white_mask = cv2.inRange(hsv_roi, np.array([0, 0, 190]), np.array([180, 45, 255]))

        # Yellow lane line mask (hue 15-35, high S, moderate/high V)
        yellow_mask = cv2.inRange(hsv_roi, np.array([15, 70, 130]), np.array([35, 255, 255]))

        marking_features: List[DetectedFeature] = []

        for color_name, mask in [("white", white_mask), ("yellow", yellow_mask)]:
            # Clean mask
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 150:
                    continue

                rect = cv2.minAreaRect(cnt)
                (_, _), (rw, rh), _ = rect
                length = max(rw, rh)
                thickness = max(1.0, min(rw, rh))

                # Elongated painted stripe check
                if length >= int(height * 0.08) and (length / thickness) >= 3.0:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    full_y0 = lower_y + by
                    full_y1 = full_y0 + bh
                    marking_features.append(
                        DetectedFeature(
                            class_name="road_marking",
                            confidence=0.72,
                            box=(float(bx), float(full_y0), float(bx + bw), float(full_y1)),
                            detector_backend=self.name,
                            attributes={"color": color_name, "aspect_ratio": round(length / thickness, 1)},
                            description=f"Elongated painted {color_name} road marking or lane divider.",
                        )
                    )

        return self._non_max_suppress(marking_features, iou_thresh=0.35)[:6]

    def _non_max_suppress(self, features: List[DetectedFeature], iou_thresh: float) -> List[DetectedFeature]:
        """Simple non-maximum suppression for overlapping bounding boxes."""
        if not features:
            return []
        sorted_feats = sorted(features, key=lambda f: f.confidence, reverse=True)
        kept: List[DetectedFeature] = []

        for feat in sorted_feats:
            clash = False
            for k in kept:
                # Calculate IoU
                ax0, ay0, ax1, ay1 = feat.box
                bx0, by0, bx1, by1 = k.box
                ix0, iy0 = max(ax0, bx0), max(ay0, by0)
                ix1, iy1 = min(ax1, bx1), min(ay1, by1)
                iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
                inter = iw * ih
                union = ((ax1 - ax0) * (ay1 - ay0)) + ((bx1 - bx0) * (by1 - by0)) - inter
                iou = inter / union if union > 0 else 0
                if iou > iou_thresh:
                    clash = True
                    break
            if not clash:
                kept.append(feat)

        return kept
