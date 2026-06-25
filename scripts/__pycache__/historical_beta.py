#!/usr/bin/env python3
"""Compatibility wrapper for scripts/historical_beta.py."""

from pathlib import Path
import runpy


runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "historical_beta.py"),
    run_name="__main__",
)
