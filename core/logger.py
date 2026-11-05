"""HOUND Logging Infrastructure.

Provides Rich-enhanced console logging and structured file logging.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler

_LOGGER_INITIALIZED = False
_CONSOLE: Optional[Console] = None


def get_console() -> Console:
    """Return the shared Rich console with UTF-8 encoding safeguard."""
    global _CONSOLE
    if _CONSOLE is None:
        _CONSOLE = Console(color_system="auto", stderr=False)
    return _CONSOLE


def setup_logger(
    name: str = "hound",
    log_level: str = "INFO",
    log_file: Optional[Path | str] = None,
) -> logging.Logger:
    """Configure and return the root HOUND logger."""
    global _LOGGER_INITIALIZED

    logger = logging.getLogger(name)
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers if setup_logger is called repeatedly
    if not logger.handlers:
        # Rich handler for console
        console_handler = RichHandler(
            console=get_console(),
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
            markup=True,
        )
        console_handler.setLevel(numeric_level)
        logger.addHandler(console_handler)

        # File handler if requested
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    _LOGGER_INITIALIZED = True
    return logger


def get_logger(name: str = "hound") -> logging.Logger:
    """Get a child or named logger for a HOUND module."""
    return logging.getLogger(name)
