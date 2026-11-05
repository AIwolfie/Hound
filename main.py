#!/usr/bin/env python3
"""HOUND — Autonomous Visual Geolocation & OSINT Agent.

Main system entry point.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure Hound workspace is on Python path
workspace_dir = Path(__file__).parent.resolve()
if str(workspace_dir) not in sys.path:
    sys.path.insert(0, str(workspace_dir))

from interface.cli import main

if __name__ == "__main__":
    main()
