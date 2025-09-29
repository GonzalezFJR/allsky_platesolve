"""Utility package for all-sky calibration models and helpers."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"

__all__ = [
    "BASE_DIR",
    "DATA_DIR",
]
