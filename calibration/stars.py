"""Star catalog helpers used by the calibration web service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Dict, List
import math

import pandas as pd

from . import DATA_DIR
from .model import LensModel
from .observer import create_observer, radec_to_altaz


CATALOG_PATH = DATA_DIR / "named_stars.csv"


@dataclass
class StarEntry:
    name: str
    ra_deg: float
    dec_deg: float
    magnitude: float

    @classmethod
    def from_row(cls, row: pd.Series) -> "StarEntry":
        mag_value = row["Vmag"]
        try:
            magnitude = float(mag_value)
        except (TypeError, ValueError):
            magnitude = math.nan

        return cls(
            name=str(row["IAU Name"]),
            ra_deg=float(row["RA"]),
            dec_deg=float(row["Dec"]),
            magnitude=magnitude,
        )


@lru_cache(maxsize=1)
def load_catalog() -> List[StarEntry]:
    df = pd.read_csv(CATALOG_PATH)
    stars: List[StarEntry] = [StarEntry.from_row(row) for _, row in df.iterrows()]
    return stars


def normalise_name(name: str) -> str:
    import unicodedata

    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return name.strip().lower()


@lru_cache(maxsize=1)
def star_name_lookup() -> Dict[str, StarEntry]:
    return {normalise_name(star.name): star for star in load_catalog()}


def visible_stars(
    latitude: float,
    longitude: float,
    elevation: float,
    capture_time: datetime,
    min_altitude: float = 0.0,
) -> List[Dict]:
    observer = create_observer(latitude, longitude, elevation, capture_time)
    stars = load_catalog()
    visible = []
    for star in stars:
        alt, az = radec_to_altaz(observer, star.ra_deg, star.dec_deg)
        if alt < min_altitude:
            continue
        magnitude = star.magnitude
        if magnitude is None or not math.isfinite(magnitude):
            magnitude_value = None
        else:
            magnitude_value = float(magnitude)

        visible.append({
            "name": star.name,
            "alt": alt,
            "az": az,
            "magnitude": magnitude_value,
            "ra": star.ra_deg,
            "dec": star.dec_deg,
        })
    visible.sort(key=lambda item: -item["alt"])
    return visible


def project_stars(
    model: LensModel,
    latitude: float,
    longitude: float,
    elevation: float,
    capture_time: datetime,
    min_altitude: float = 0.0,
) -> List[Dict]:
    if not model.is_fitted:
        raise RuntimeError("El modelo debe estar ajustado para proyectar las estrellas.")

    stars = visible_stars(latitude, longitude, elevation, capture_time, min_altitude)
    projected = []
    for star in stars:
        try:
            x, y = model.xy_inv(star["alt"], star["az"])
        except Exception:
            continue

        if not (math.isfinite(x) and math.isfinite(y)):
            continue

        projected.append({
            "name": star["name"],
            "alt": star["alt"],
            "az": star["az"],
            "magnitude": star["magnitude"],
            "ra": star["ra"],
            "dec": star["dec"],
            "x": float(x),
            "y": float(y),
        })
    return projected


def lookup_star(name: str) -> StarEntry:
    lookup = star_name_lookup()
    key = normalise_name(name)
    if key not in lookup:
        raise KeyError(f"Estrella '{name}' no encontrada en el catálogo")
    return lookup[key]
