import pandas as pd
import requests
from io import StringIO
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import os
from astropy import units as u
from astropy.coordinates import Angle

from app.calibration.model import get_model
from app.calibration.observer import create_observer, radec_to_altaz

# Definir la ruta base de la aplicación de forma consistente
BASE_DIR = Path(os.getenv('APP_BASE_PATH', 'app')).resolve()
STARS_CATALOG_PATH = BASE_DIR / "data" / "named_stars.csv"

def get_named_stars() -> pd.DataFrame:
    """Lee el catálogo de estrellas brillantes desde el archivo CSV."""
    try:
        return pd.read_csv(STARS_CATALOG_PATH)
    except Exception as e:
        print(f"Error al obtener el catálogo de estrellas: {e}")
        return pd.DataFrame()

def get_star_positions(
    capture_time: datetime,
    device_params: dict
) -> List[Dict]:
    """
    Calcula las coordenadas (x, y) de las estrellas del catálogo para una observación dada.

    Args:
        capture_time: Hora de la captura de la imagen.
        device_params: Diccionario con 'latitude', 'longitude', 'elevation' y 'calibration_model'.

    Returns:
        Una lista de diccionarios, cada uno con 'name', 'ra', 'dec', 'x', 'y'.
    """
    stars_df = get_named_stars()
    if stars_df.empty:
        return []

    # Extraer parámetros
    latitude = device_params.get('latitude')
    longitude = device_params.get('longitude')
    elevation = device_params.get('elevation', 0)
    calib_model_params = device_params.get('calibration_model')
    print('Model params: ',calib_model_params)

    if not all([latitude, longitude, calib_model_params]):
        raise ValueError("Faltan parámetros del dispositivo o del modelo de calibración.")

    # Crear observador y modelo
    observer = create_observer(latitude, longitude, elevation, capture_time)
    model = get_model(calib_model_params['model_type'], calib_model_params)

    if not model.is_fitted:
        raise ValueError("El modelo de calibración no está ajustado.")

    projected_stars = []
    for _, star in stars_df.iterrows():
        # Convertir RA de horas a grados
        ra = Angle(star['RA'], unit=u.deg).degree
        dec = Angle(star['Dec'], unit=u.deg).degree

        # Calcular Alt/Az
        alt, az = radec_to_altaz(observer, ra, dec)

        # Si la estrella está por debajo del horizonte, no la proyectamos
        if alt < 0:
            continue

        # Proyectar a coordenadas de píxel
        x, y = model.xy_inv(alt, az)

        projected_stars.append({
            "name": star['IAU Name'],
            "ra": ra,
            "dec": dec,
            "alt": alt,
            "az": az,
            "x": int(x),
            "y": int(y)
        })
    
    return projected_stars
