"""HOUND Configuration Management.

Loads configuration from YAML files and environment variables with type validation.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional
import yaml
from pydantic import BaseModel, Field


class HoundMetadataConfig(BaseModel):
    version: str = "1.0.0"
    environment: str = "development"
    log_level: str = "INFO"


class PathsConfig(BaseModel):
    workspace_dir: Path = Field(default_factory=lambda: Path.cwd())
    storage_dir: Path = Field(default_factory=lambda: Path.cwd() / "data" / "storage")
    cache_dir: Path = Field(default_factory=lambda: Path.cwd() / "data" / "cache")
    reports_dir: Path = Field(default_factory=lambda: Path.cwd() / "data" / "reports")
    database_path: Path = Field(default_factory=lambda: Path.cwd() / "data" / "hound.db")
    geo_sleuth_path: Path = Field(
        default_factory=lambda: Path("C:/Users/mayan/Downloads/geo-sleuth-main/geo-sleuth-main")
    )

    def resolve_paths(self, base_dir: Path) -> None:
        """Resolve any relative paths against the base workspace directory."""
        if not self.workspace_dir.is_absolute():
            self.workspace_dir = (base_dir / self.workspace_dir).resolve()
        if not self.storage_dir.is_absolute():
            self.storage_dir = (base_dir / self.storage_dir).resolve()
        if not self.cache_dir.is_absolute():
            self.cache_dir = (base_dir / self.cache_dir).resolve()
        if not self.reports_dir.is_absolute():
            self.reports_dir = (base_dir / self.reports_dir).resolve()
        if not self.database_path.is_absolute():
            self.database_path = (base_dir / self.database_path).resolve()


class NetworkConfig(BaseModel):
    timeout_seconds: int = 30
    user_agent: str = "HOUND-OSINT/1.0 (+https://github.com/hound-osint/hound; research)"
    max_retries: int = 3
    backoff_factor: float = 1.5
    proxy: Optional[str] = None


class OSMConfig(BaseModel):
    overpass_mirrors: List[str] = Field(
        default_factory=lambda: [
            "https://overpass-api.de/api/interpreter",
            "https://overpass.private.coffee/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
        ]
    )
    nominatim_url: str = "https://nominatim.openstreetmap.org/search"
    rate_limit_delay_seconds: float = 1.0


class ElevationConfig(BaseModel):
    terrarium_tile_url: str = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
    max_zoom: int = 15


class SatelliteConfig(BaseModel):
    google_satellite_url: str = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
    osm_tiles_url: str = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"


class ReverseImageConfig(BaseModel):
    enabled_providers: List[str] = Field(default_factory=lambda: ["yandex", "baidu"])
    headless_browser: bool = True
    browser_timeout_ms: int = 30000


class IntegrationsConfig(BaseModel):
    openstreetmap: OSMConfig = Field(default_factory=OSMConfig)
    elevation: ElevationConfig = Field(default_factory=ElevationConfig)
    satellite: SatelliteConfig = Field(default_factory=SatelliteConfig)
    reverse_image: ReverseImageConfig = Field(default_factory=ReverseImageConfig)


class InvestigationConfig(BaseModel):
    preserve_original_images: bool = True
    max_candidates_to_rank: int = 50
    auto_falsification_enabled: bool = True
    min_evidence_for_high_confidence: int = 3
    default_search_radius_m: int = 5000


class HoundConfig(BaseModel):
    hound: HoundMetadataConfig = Field(default_factory=HoundMetadataConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    integrations: IntegrationsConfig = Field(default_factory=IntegrationsConfig)
    investigation: InvestigationConfig = Field(default_factory=InvestigationConfig)


_CONFIG_CACHE: Optional[HoundConfig] = None


def load_config(config_file: Optional[Path | str] = None) -> HoundConfig:
    """Load HOUND configuration from a YAML file, with environment variable overrides."""
    global _CONFIG_CACHE

    target_path: Optional[Path] = None
    if config_file:
        target_path = Path(config_file)
    elif os.environ.get("HOUND_CONFIG"):
        target_path = Path(os.environ["HOUND_CONFIG"])
    else:
        # Check standard locations
        candidates = [
            Path.cwd() / "config.yaml",
            Path.cwd() / "config.yml",
            Path(__file__).parent.parent / "config.yaml",
        ]
        for c in candidates:
            if c.is_file():
                target_path = c
                break

    data: dict[str, Any] = {}
    base_dir = Path.cwd()
    if target_path and target_path.is_file():
        base_dir = target_path.parent
        with open(target_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    config = HoundConfig.model_validate(data)
    config.paths.resolve_paths(base_dir)

    # Allow environment variable overrides
    if os.environ.get("HOUND_LOG_LEVEL"):
        config.hound.log_level = os.environ["HOUND_LOG_LEVEL"].upper()
    if os.environ.get("HOUND_STORAGE_DIR"):
        config.paths.storage_dir = Path(os.environ["HOUND_STORAGE_DIR"]).resolve()
    if os.environ.get("HOUND_GEO_SLEUTH_PATH"):
        config.paths.geo_sleuth_path = Path(os.environ["HOUND_GEO_SLEUTH_PATH"]).resolve()

    _CONFIG_CACHE = config
    return config


def get_config() -> HoundConfig:
    """Return the cached or default configuration."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = load_config()
    return _CONFIG_CACHE
