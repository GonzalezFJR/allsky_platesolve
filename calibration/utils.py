"""Utility helpers for the calibration web API."""

from __future__ import annotations

from typing import List

import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder


def prepare_image_array(image_data: np.ndarray, flip_horizontal: bool) -> np.ndarray:
    if image_data.ndim == 3:
        # Convertir a escala de grises usando luminancia estándar
        image_gray = np.dot(image_data[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        image_gray = image_data.astype(float)

    if flip_horizontal:
        image_gray = np.fliplr(image_gray)

    return image_gray.astype(float)


def run_daostar_finder(
    image_data: np.ndarray,
    fwhm: float,
    threshold_sigma: float,
    sharplo: float,
    sharphi: float,
    roundlo: float,
    roundhi: float,
    max_stars: int,
    flip_horizontal: bool = False,
) -> List[dict]:
    data = prepare_image_array(image_data, flip_horizontal)
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    daofind = DAOStarFinder(
        fwhm=max(fwhm, 1.0),
        threshold=max(threshold_sigma, 0.1) * std,
        sharplo=sharplo,
        sharphi=sharphi,
        roundlo=roundlo,
        roundhi=roundhi,
    )
    sources = daofind(data - median)
    if sources is None:
        return []

    sources.sort("peak", reverse=True)
    detections = []
    for row in sources[:max_stars]:
        detections.append(
            {
                "x": float(row["xcentroid"]),
                "y": float(row["ycentroid"]),
                "flux": float(row["flux"]),
                "peak": float(row["peak"]),
            }
        )
    return detections

