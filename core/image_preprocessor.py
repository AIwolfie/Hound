"""HOUND Image Preprocessing Engine.

Generates normalized analysis variants, contrast enhancements (CLAHE), edge maps,
and overlapping coordinate-mapped crop pyramids without modifying original media.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pydantic import BaseModel, Field

from core.logger import get_logger

logger = get_logger("hound.preprocessor")


class VariantInfo(BaseModel):
    name: str
    dimensions: Tuple[int, int]
    file_path: Path
    description: str


class CropInfo(BaseModel):
    name: str
    crop_type: str  # "edge", "corner", "tile", "region"
    box_original: Tuple[int, int, int, int]  # (x0, y0, x1, y1) in original image space
    dimensions: Tuple[int, int]
    file_path: Path


class PreprocessingManifest(BaseModel):
    source_image: str
    original_dimensions: Tuple[int, int]  # (width, height)
    normalized_dimensions: Tuple[int, int]
    analysis_variants: List[VariantInfo] = Field(default_factory=list)
    crops: List[CropInfo] = Field(default_factory=list)
    processing_time_seconds: float = 0.0
    output_dir: Path


class ImagePreprocessor:
    """Produces multi-scale, enhanced, and coordinate-mapped image variants."""

    def __init__(
        self,
        max_dimension: int = 4096,
        tile_grid: Tuple[int, int] = (2, 2),
        tile_overlap_ratio: float = 0.15,
        enable_upscaling: bool = True,
    ):
        self.max_dimension = max_dimension
        self.tile_grid = tile_grid
        self.tile_overlap_ratio = tile_overlap_ratio
        self.enable_upscaling = enable_upscaling

    @staticmethod
    def normalize_orientation(image: Image.Image) -> Image.Image:
        """Apply EXIF rotation to guarantee orientation-accurate coordinates."""
        try:
            return ImageOps.exif_transpose(image) or image
        except Exception as e:
            logger.debug(f"EXIF transpose failed: {e}")
            return image

    def process(
        self,
        image_path: Path | str,
        output_dir: Path | str,
        generate_tiles: bool = True,
        generate_edges: bool = True,
    ) -> PreprocessingManifest:
        """Execute full preprocessing pipeline and save artifacts to output_dir."""
        t0 = time.perf_counter()
        in_path = Path(image_path).resolve()
        out_path = Path(output_dir).resolve()
        variants_dir = out_path / "variants"
        crops_dir = out_path / "crops"
        variants_dir.mkdir(parents=True, exist_ok=True)
        crops_dir.mkdir(parents=True, exist_ok=True)

        with Image.open(in_path) as raw_img:
            orig_dims = raw_img.size  # (W, H)
            img = self.normalize_orientation(raw_img)
            norm_dims = img.size

            # Ensure RGB mode
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Check if resizing is necessary for memory preservation
            w, h = norm_dims
            if max(w, h) > self.max_dimension:
                scale = self.max_dimension / float(max(w, h))
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                norm_dims = img.size
                logger.info(f"Downsampled oversized image from {orig_dims} to {norm_dims}")

        variants: List[VariantInfo] = []
        crops: List[CropInfo] = []

        # 1. Base Normalized RGB
        full_path = variants_dir / f"{in_path.stem}_normalized.jpg"
        img.save(full_path, format="JPEG", quality=95)
        variants.append(
            VariantInfo(
                name="full_normalized",
                dimensions=norm_dims,
                file_path=full_path,
                description="Orientation-normalized full resolution RGB image.",
            )
        )

        # 2. Contrast Enhanced (Pillow)
        contrast_enhancer = ImageEnhance.Contrast(img)
        contrast_img = contrast_enhancer.enhance(1.4)
        contrast_path = variants_dir / f"{in_path.stem}_contrast.jpg"
        contrast_img.save(contrast_path, format="JPEG", quality=92)
        variants.append(
            VariantInfo(
                name="contrast_enhanced",
                dimensions=norm_dims,
                file_path=contrast_path,
                description="Global linear contrast enhanced variant.",
            )
        )

        # Convert to OpenCV numpy array for advanced filter operations
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv_gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # 3. Grayscale
        gray_path = variants_dir / f"{in_path.stem}_grayscale.jpg"
        cv2.imwrite(str(gray_path), cv_gray)
        variants.append(
            VariantInfo(
                name="grayscale",
                dimensions=norm_dims,
                file_path=gray_path,
                description="Monochrome luminance representation.",
            )
        )

        # 4. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        clahe_gray = clahe.apply(cv_gray)
        clahe_bgr = cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2BGR)
        clahe_path = variants_dir / f"{in_path.stem}_clahe.jpg"
        cv2.imwrite(str(clahe_path), clahe_bgr)
        variants.append(
            VariantInfo(
                name="clahe",
                dimensions=norm_dims,
                file_path=clahe_path,
                description="Adaptive local histogram equalization (CLAHE) for shadowed/low-contrast features.",
            )
        )

        # 5. Edge Enhanced (Canny edges)
        edges = cv2.Canny(cv_gray, 50, 150)
        edges_inv = cv2.bitwise_not(edges)
        edge_path = variants_dir / f"{in_path.stem}_edges.png"
        cv2.imwrite(str(edge_path), edges_inv)
        variants.append(
            VariantInfo(
                name="edge_enhanced",
                dimensions=norm_dims,
                file_path=edge_path,
                description="Canny structural edge map highlighting infrastructure contours.",
            )
        )

        # 6. Sharpened Variant
        sharp_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
        sharpened_cv = cv2.filter2D(cv_img, -1, sharp_kernel)
        sharp_path = variants_dir / f"{in_path.stem}_sharpened.jpg"
        cv2.imwrite(str(sharp_path), sharpened_cv)
        variants.append(
            VariantInfo(
                name="sharpened",
                dimensions=norm_dims,
                file_path=sharp_path,
                description="High-pass spatial frequency sharpened variant for license plates and signage.",
            )
        )

        # 7. Upscaled 2x (if original resolution is moderate)
        if self.enable_upscaling and max(norm_dims) <= 2048:
            up_w, up_h = norm_dims[0] * 2, norm_dims[1] * 2
            upscaled = img.resize((up_w, up_h), Image.Resampling.LANCZOS)
            up_path = variants_dir / f"{in_path.stem}_upscaled2x.jpg"
            upscaled.save(up_path, format="JPEG", quality=90)
            variants.append(
                VariantInfo(
                    name="upscaled_2x",
                    dimensions=(up_w, up_h),
                    file_path=up_path,
                    description="2x Lanczos super-sampling variant for small textual details.",
                )
            )

        width, height = norm_dims

        # 8. Edge and Corner Crops
        if generate_edges:
            edge_defs = [
                ("edge_north", (0, 0, width, int(height * 0.30))),
                ("edge_south", (0, int(height * 0.70), width, height)),
                ("edge_west", (0, 0, int(width * 0.30), height)),
                ("edge_east", (int(width * 0.70), 0, width, height)),
                ("corner_nw", (0, 0, int(width * 0.35), int(height * 0.35))),
                ("corner_ne", (int(width * 0.65), 0, width, int(height * 0.35))),
                ("corner_sw", (0, int(height * 0.65), int(width * 0.35), height)),
                ("corner_se", (int(width * 0.65), int(height * 0.65), width, height)),
            ]
            for name, box in edge_defs:
                x0, y0, x1, y1 = box
                crop_img = img.crop((x0, y0, x1, y1))
                crop_file = crops_dir / f"{in_path.stem}_{name}.jpg"
                crop_img.save(crop_file, format="JPEG", quality=92)
                crops.append(
                    CropInfo(
                        name=name,
                        crop_type="corner" if name.startswith("corner") else "edge",
                        box_original=(x0, y0, x1, y1),
                        dimensions=crop_img.size,
                        file_path=crop_file,
                    )
                )

        # 9. Overlapping Grid Tiles
        if generate_tiles and self.tile_grid:
            cols, rows = self.tile_grid
            step_x = width / float(cols)
            step_y = height / float(rows)
            pad_x = int(step_x * self.tile_overlap_ratio)
            pad_y = int(step_y * self.tile_overlap_ratio)

            for r in range(rows):
                for c in range(cols):
                    x0 = max(0, int(c * step_x) - pad_x)
                    y0 = max(0, int(r * step_y) - pad_y)
                    x1 = min(width, int((c + 1) * step_x) + pad_x)
                    y1 = min(height, int((r + 1) * step_y) + pad_y)

                    tile_name = f"tile_r{r}_c{c}"
                    tile_img = img.crop((x0, y0, x1, y1))
                    tile_file = crops_dir / f"{in_path.stem}_{tile_name}.jpg"
                    tile_img.save(tile_file, format="JPEG", quality=92)
                    crops.append(
                        CropInfo(
                            name=tile_name,
                            crop_type="tile",
                            box_original=(x0, y0, x1, y1),
                            dimensions=tile_img.size,
                            file_path=tile_file,
                        )
                    )

        dt = time.perf_counter() - t0
        manifest = PreprocessingManifest(
            source_image=in_path.name,
            original_dimensions=orig_dims,
            normalized_dimensions=norm_dims,
            analysis_variants=variants,
            crops=crops,
            processing_time_seconds=round(dt, 3),
            output_dir=out_path,
        )

        manifest_file = out_path / "preprocessing_manifest.json"
        manifest_file.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        logger.info(
            f"Preprocessed {in_path.name}: {len(variants)} variants, {len(crops)} crops ({dt:.2f}s)"
        )
        return manifest
