"""HOUND Image Loader & Forensic Metadata Extractor.

Handles image validation, hashing, preservation, EXIF parsing, and integrity checks.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image, ExifTags
from pydantic import BaseModel, Field

from core.logger import get_logger

logger = get_logger("hound.image_loader")

# Supported image formats
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".heic"}


class GPSCoordinates(BaseModel):
    latitude: float
    longitude: float
    altitude_m: Optional[float] = None
    heading_deg: Optional[float] = None
    datum: str = "WGS-84"

    def to_tuple(self) -> Tuple[float, float]:
        return (self.latitude, self.longitude)

    def to_formatted_str(self) -> str:
        lat_dir = "N" if self.latitude >= 0 else "S"
        lon_dir = "E" if self.longitude >= 0 else "W"
        return f"{abs(self.latitude):.6f}°{lat_dir}, {abs(self.longitude):.6f}°{lon_dir}"


class ExifMetadata(BaseModel):
    has_exif: bool = False
    make: Optional[str] = None
    model: Optional[str] = None
    software: Optional[str] = None
    datetime_original: Optional[datetime] = None
    datetime_digitized: Optional[datetime] = None
    offset_time: Optional[str] = None
    exposure_time: Optional[str] = None
    f_number: Optional[float] = None
    iso: Optional[int] = None
    focal_length_mm: Optional[float] = None
    focal_length_35mm: Optional[int] = None
    lens_make: Optional[str] = None
    lens_model: Optional[str] = None
    orientation: Optional[int] = None
    gps: Optional[GPSCoordinates] = None
    raw_tags: Dict[str, Any] = Field(default_factory=dict)


class IntegrityAssessment(BaseModel):
    is_valid_image: bool
    status: str  # "complete", "incomplete", "absent", "suspicious"
    findings: List[str] = Field(default_factory=list)
    has_gps: bool = False
    has_camera_info: bool = False
    has_timestamp: bool = False
    is_edited_or_exported: bool = False


class ImageRecord(BaseModel):
    sha256: str
    filename: str
    original_path: Path
    preserved_path: Optional[Path] = None
    file_size_bytes: int
    format: str
    width: int
    height: int
    aspect_ratio: float
    mode: str
    exif: ExifMetadata
    integrity: IntegrityAssessment


def _convert_dms_to_decimal(dms_values: Any, ref: str) -> Optional[float]:
    """Convert degrees, minutes, seconds rational tuple to decimal degrees."""
    try:
        degrees = float(dms_values[0])
        minutes = float(dms_values[1])
        seconds = float(dms_values[2])
        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
        if ref in ("S", "W"):
            decimal = -decimal
        return round(decimal, 7)
    except (TypeError, ValueError, IndexError, ZeroDivisionError) as err:
        logger.debug(f"Failed to convert DMS {dms_values} ref {ref}: {err}")
        return None


def _parse_exif_timestamp(ts_str: Optional[str]) -> Optional[datetime]:
    """Parse EXIF format timestamp strings (e.g., '2024:08:14 15:32:01')."""
    if not ts_str or not isinstance(ts_str, str):
        return None
    cleaned = ts_str.strip()
    formats = [
        "%Y:%m:%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y:%m:%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


def extract_exif(img: Image.Image) -> ExifMetadata:
    """Extract and normalize EXIF metadata including GPS and camera properties."""
    exif_data = ExifMetadata()
    raw_exif = img.getexif()
    if not raw_exif:
        return exif_data

    exif_data.has_exif = True
    raw_tags: Dict[str, Any] = {}

    # Map main EXIF tags
    for tag_id, value in raw_exif.items():
        tag_name = ExifTags.TAGS.get(tag_id, f"Tag_{tag_id}")
        raw_tags[tag_name] = str(value) if not isinstance(value, (int, float, str)) else value

    # Check for SubIFD (ExifOffset)
    sub_ifd = raw_exif.get_ifd(ExifTags.IFD.Exif)
    for tag_id, value in sub_ifd.items():
        tag_name = ExifTags.TAGS.get(tag_id, f"Exif_{tag_id}")
        raw_tags[tag_name] = str(value) if not isinstance(value, (int, float, str)) else value

    # Check for GPS IFD
    gps_ifd = raw_exif.get_ifd(ExifTags.IFD.GPSInfo)
    for tag_id, value in gps_ifd.items():
        tag_name = ExifTags.GPSTAGS.get(tag_id, f"GPS_{tag_id}")
        raw_tags[tag_name] = str(value) if not isinstance(value, (int, float, str)) else value

    exif_data.raw_tags = raw_tags
    exif_data.make = raw_tags.get("Make")
    exif_data.model = raw_tags.get("Model")
    exif_data.software = raw_tags.get("Software")
    exif_data.orientation = raw_tags.get("Orientation")

    # Timestamps
    ts_orig = raw_tags.get("DateTimeOriginal") or raw_tags.get("DateTime")
    exif_data.datetime_original = _parse_exif_timestamp(ts_orig)
    ts_dig = raw_tags.get("DateTimeDigitized")
    exif_data.datetime_digitized = _parse_exif_timestamp(ts_dig)
    exif_data.offset_time = raw_tags.get("OffsetTimeOriginal") or raw_tags.get("OffsetTime")

    # Camera Optics
    exif_data.exposure_time = raw_tags.get("ExposureTime")
    if "FNumber" in raw_tags:
        try:
            exif_data.f_number = float(raw_tags["FNumber"])
        except (ValueError, TypeError):
            pass
    if "ISOSpeedRatings" in raw_tags:
        try:
            exif_data.iso = int(raw_tags["ISOSpeedRatings"])
        except (ValueError, TypeError):
            pass
    if "FocalLength" in raw_tags:
        try:
            exif_data.focal_length_mm = float(raw_tags["FocalLength"])
        except (ValueError, TypeError):
            pass
    if "FocalLengthIn35mmFilm" in raw_tags:
        try:
            exif_data.focal_length_35mm = int(raw_tags["FocalLengthIn35mmFilm"])
        except (ValueError, TypeError):
            pass

    exif_data.lens_make = raw_tags.get("LensMake")
    exif_data.lens_model = raw_tags.get("LensModel")

    # GPS extraction
    if gps_ifd:
        lat_val = gps_ifd.get(2)  # GPSLatitude
        lat_ref = gps_ifd.get(1, "N")  # GPSLatitudeRef
        lon_val = gps_ifd.get(4)  # GPSLongitude
        lon_ref = gps_ifd.get(3, "E")  # GPSLongitudeRef
        alt_val = gps_ifd.get(6)  # GPSAltitude
        alt_ref = gps_ifd.get(5, 0)  # GPSAltitudeRef
        img_dir = gps_ifd.get(17)  # GPSImgDirection

        if lat_val and lon_val:
            lat = _convert_dms_to_decimal(lat_val, lat_ref)
            lon = _convert_dms_to_decimal(lon_val, lon_ref)
            if lat is not None and lon is not None:
                alt = None
                if alt_val is not None:
                    try:
                        alt = float(alt_val)
                        if alt_ref == 1:  # Below sea level
                            alt = -alt
                    except (ValueError, TypeError):
                        pass

                heading = None
                if img_dir is not None:
                    try:
                        heading = float(img_dir)
                    except (ValueError, TypeError):
                        pass

                exif_data.gps = GPSCoordinates(
                    latitude=lat,
                    longitude=lon,
                    altitude_m=alt,
                    heading_deg=heading,
                )

    return exif_data


def evaluate_integrity(exif: ExifMetadata, img_format: str) -> IntegrityAssessment:
    """Assess metadata authenticity, completeness, and check for editing artifacts."""
    findings: List[str] = []
    has_gps = exif.gps is not None
    has_camera = bool(exif.make or exif.model)
    has_timestamp = exif.datetime_original is not None
    is_edited = False

    if not exif.has_exif:
        return IntegrityAssessment(
            is_valid_image=True,
            status="absent",
            findings=["No EXIF metadata present. Image may be a screenshot, web download, or sanitized."],
            has_gps=False,
            has_camera_info=False,
            has_timestamp=False,
            is_edited_or_exported=False,
        )

    # Check for software traces
    software = (exif.software or "").lower()
    editing_tools = ["photoshop", "gimp", "canva", "lightroom", "snapseed", "picsart", "vsco"]
    for tool in editing_tools:
        if tool in software:
            is_edited = True
            findings.append(f"Exported or modified with graphic software: '{exif.software}'")
            break

    if has_camera:
        findings.append(f"Camera identified: {exif.make or ''} {exif.model or ''}".strip())
    else:
        findings.append("Camera hardware tags missing from EXIF.")

    if has_timestamp:
        findings.append(f"Original capture timestamp recorded: {exif.datetime_original.isoformat()}")
    else:
        findings.append("Capture timestamp missing from EXIF.")

    if has_gps:
        assert exif.gps is not None
        findings.append(f"Embedded GPS detected: {exif.gps.to_formatted_str()}")
        if exif.gps.heading_deg is not None:
            findings.append(f"Camera heading direction embedded: {exif.gps.heading_deg:.1f}°")
    else:
        findings.append("No embedded GPS coordinates found in image metadata.")

    # Determine status
    if is_edited:
        status = "suspicious"
    elif has_gps and has_camera and has_timestamp:
        status = "complete"
    elif has_camera or has_timestamp or has_gps:
        status = "incomplete"
    else:
        status = "absent"

    return IntegrityAssessment(
        is_valid_image=True,
        status=status,
        findings=findings,
        has_gps=has_gps,
        has_camera_info=has_camera,
        has_timestamp=has_timestamp,
        is_edited_or_exported=is_edited,
    )


def load_and_assess_image(
    image_path: Path | str,
    storage_dir: Optional[Path | str] = None,
    preserve_original: bool = True,
) -> ImageRecord:
    """Load, validate, hash, preserve, and forensically analyze an image.

    Args:
        image_path: Path to target file.
        storage_dir: Optional directory to preserve a copy of the image.
        preserve_original: If True, copies the file into storage_dir/intake/<hash>.ext.

    Returns:
        Structured ImageRecord containing file properties, dimensions, EXIF, and integrity.
    """
    path = Path(image_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Target image file not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    # Compute SHA-256 hash
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    sha256 = hasher.hexdigest()
    file_size = path.stat().st_size

    # Validate image via Pillow
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception as e:
        raise ValueError(f"Corrupt or invalid image file '{path.name}': {e}") from e

    # Re-open for metadata reading and properties
    with Image.open(path) as img:
        width, height = img.size
        img_format = img.format or ext.lstrip(".").upper()
        mode = img.mode
        aspect_ratio = round(width / height, 4) if height > 0 else 1.0
        exif = extract_exif(img)

    integrity = evaluate_integrity(exif, img_format)

    # Preserve copy
    preserved_path: Optional[Path] = None
    if preserve_original and storage_dir:
        intake_dir = Path(storage_dir) / "intake" / sha256[:12]
        intake_dir.mkdir(parents=True, exist_ok=True)
        preserved_path = intake_dir / f"original_{path.name}"
        if not preserved_path.is_file():
            shutil.copy2(path, preserved_path)
            logger.info(f"Preserved original image copy at {preserved_path}")

    return ImageRecord(
        sha256=sha256,
        filename=path.name,
        original_path=path,
        preserved_path=preserved_path,
        file_size_bytes=file_size,
        format=img_format,
        width=width,
        height=height,
        aspect_ratio=aspect_ratio,
        mode=mode,
        exif=exif,
        integrity=integrity,
    )
