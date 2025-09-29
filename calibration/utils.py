import unicodedata
import os
import cv2
import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from datetime import datetime, timedelta, timezone

def detect_stars_in_image(image_path: str, fwhm: float = 5.0, threshold_sigma: float = 5.0):
    """Detecta estrellas en una imagen usando photutils y devuelve sus coordenadas."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []
    data = img.astype(float)
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * std)
    sources = daofind(data - median)
    if sources is None:
        return []
    return [{'x': int(round(s['xcentroid'])), 'y': int(round(s['ycentroid']))} for s in sources]

