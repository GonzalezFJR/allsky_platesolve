import ephem
import math
from datetime import datetime

def create_observer(latitude: float, longitude: float, elevation: float, capture_time: datetime) -> ephem.Observer:
    """
    Crea y configura un objeto Observer de PyEphem.
    """
    observer = ephem.Observer()
    observer.lat = str(latitude)
    observer.lon = str(longitude)
    observer.elevation = elevation
    observer.date = capture_time
    # Configurar presión y temperatura a 0 para ignorar la refracción atmosférica,
    # ya que nuestro modelo de lente la maneja implícitamente.
    observer.pressure = 0
    observer.temp = 0
    return observer

def radec_to_altaz(observer: ephem.Observer, ra: float, dec: float) -> tuple[float, float]:
    """
    Convierte coordenadas ecuatoriales (RA, Dec) a coordenadas horizontales (Alt, Az)
    para un observador y tiempo dados.
    """
    star = ephem.FixedBody()
    star._ra = ephem.degrees(str(ra))
    star._dec = ephem.degrees(str(dec))
    
    # Calcular la posición para el observador
    star.compute(observer)
    
    # Devolver altitud y azimut en grados
    alt = math.degrees(star.alt)
    az = math.degrees(star.az)
    
    return alt, az
