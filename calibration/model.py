"""Lens calibration models for the all-sky plate solving web application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
from scipy.optimize import minimize

# --- Funciones de Distorsión Radial ---
# Cada función devuelve la ALTITUD (en grados) dado un radio r (en píxeles)
# y sus coeficientes.


def polynomial_distortion(r: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Modelo polinómico tal que alt = 90 - (c1*r + c2*r^2 + ...)."""
    if coeffs.size == 0:
        return np.full_like(r, 90.0, dtype=float)
    poly_coeffs = np.concatenate((coeffs[::-1], [0.0]))
    return 90.0 - np.polyval(poly_coeffs, r)


def exponential_distortion(r: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Modelo exponencial. alt = 90 * exp(-|c1|*r)."""
    if coeffs.size == 0:
        return np.full_like(r, 90.0, dtype=float)
    c1 = np.abs(coeffs[0])
    return 90.0 * np.exp(-c1 * r)


@dataclass(frozen=True)
class RadialModelInfo:
    key: str
    name: str
    func: callable
    n_coeffs: int


RADIAL_MODELS: Dict[str, RadialModelInfo] = {
    "lin": RadialModelInfo("lin", "Lineal", polynomial_distortion, 1),
    "poly2": RadialModelInfo("poly2", "Polinómico (2º)", polynomial_distortion, 2),
    "poly3": RadialModelInfo("poly3", "Polinómico (3º)", polynomial_distortion, 3),
    "exp1": RadialModelInfo("exp1", "Exponencial", exponential_distortion, 1),
}

# --- Clase Principal del Modelo de Lente ---

class LensModel:
    """
    Modela la proyección de una lente de gran angular (ojo de pez).

    El modelo transforma coordenadas de píxel (x, y) a coordenadas celestes (alt, az)
    mediante tres componentes:
    1. Cénit (xc, yc): El centro óptico de la proyección en el sensor.
    2. Rotación (theta): El ángulo de orientación del sensor respecto al norte celeste.
    3. Distorsión Radial (f(r)): Una función que mapea la distancia radial desde el cénit (r)
       a la altitud (alt).
    """
    def __init__(self, model_type: str = "poly3", max_radius_px: float | None = None):
        if model_type not in RADIAL_MODELS:
            raise ValueError(f"Modelo '{model_type}' no soportado. Opciones: {list(RADIAL_MODELS.keys())}")

        self.model_type = model_type
        info = RADIAL_MODELS[model_type]
        self.radial_func = info.func
        self.n_coeffs = info.n_coeffs

        # Parámetros del modelo (se inicializan en fit)
        self.xc = None
        self.yc = None
        self.theta_deg = None # Rotación en grados
        self.tilt_angle = 0.0 # Inclinación de la cámara en grados
        self.tilt_azimuth = 0.0 # Dirección de la inclinación en grados
        self.coeffs = None
        self.is_fitted = False
        self.max_radius_px = max_radius_px or 1500.0

    def load(self, params: dict):
        """Carga los parámetros del modelo desde un diccionario."""
        self.xc = params.get('xc')
        self.yc = params.get('yc')
        self.theta_deg = params.get('theta_deg')
        self.tilt_angle = params.get('tilt_angle', 0.0)
        self.tilt_azimuth = params.get('tilt_azimuth', 0.0)
        coeffs = params.get('coeffs', [])
        self.coeffs = np.array(coeffs, dtype=float) if coeffs is not None else np.zeros(self.n_coeffs)
        self.max_radius_px = params.get('max_radius_px', self.max_radius_px)
        # El modelo se considera ajustado si todos los parámetros clave están presentes
        self.is_fitted = all(p is not None for p in [self.xc, self.yc, self.theta_deg]) and self.coeffs is not None

    def params(self) -> dict:
        """Devuelve los parámetros del modelo en un diccionario."""
        if not self.is_fitted:
            return {}
        return {
            'xc': self.xc,
            'yc': self.yc,
            'theta_deg': self.theta_deg,
            'tilt_angle': self.tilt_angle,
            'tilt_azimuth': self.tilt_azimuth,
            'coeffs': self.coeffs.tolist() if self.coeffs is not None else [],
            'model_type': self.model_type,
            'max_radius_px': self.max_radius_px,
        }

    def _tilted_to_true_coords(self, alt_cam, az_cam):
        """Convierte coordenadas del sistema inclinado de la cámara a coordenadas celestes reales."""
        if self.tilt_angle == 0:
            return alt_cam, az_cam

        alt_cam_rad = np.deg2rad(alt_cam)
        az_cam_rad = np.deg2rad(az_cam)
        tilt_rad = np.deg2rad(self.tilt_angle)
        tilt_az_rad = np.deg2rad(self.tilt_azimuth)

        # Ley de los cosenos esférica para la nueva altitud
        sin_alt = np.sin(alt_cam_rad) * np.cos(tilt_rad) + \
                    np.cos(alt_cam_rad) * np.sin(tilt_rad) * np.cos(az_cam_rad)
        alt_true = np.rad2deg(np.arcsin(np.clip(sin_alt, -1, 1)))

        # Ley de los senos y cosenos para el nuevo azimut
        y = np.sin(az_cam_rad) * np.cos(alt_cam_rad)
        x = np.cos(az_cam_rad) * np.cos(alt_cam_rad) * np.sin(tilt_rad) - np.sin(alt_cam_rad) * np.cos(tilt_rad)
        az_true = np.rad2deg(np.arctan2(y, x)) + self.tilt_azimuth

        return alt_true, (az_true + 360) % 360

    def _true_to_tilted_coords(self, alt_true, az_true):
        """Convierte coordenadas celestes reales al sistema inclinado de la cámara."""
        if self.tilt_angle == 0:
            return alt_true, az_true

        alt_true_rad = np.deg2rad(alt_true)
        az_true_rad = np.deg2rad(az_true - self.tilt_azimuth)
        tilt_rad = np.deg2rad(self.tilt_angle)

        # Invertimos la rotación
        sin_alt_cam = np.sin(alt_true_rad) * np.cos(tilt_rad) - \
                        np.cos(alt_true_rad) * np.sin(tilt_rad) * np.cos(az_true_rad)
        alt_cam = np.rad2deg(np.arcsin(np.clip(sin_alt_cam, -1, 1)))

        y = np.sin(az_true_rad) * np.cos(alt_true_rad)
        x = np.cos(az_true_rad) * np.cos(alt_true_rad) * np.sin(tilt_rad) + np.sin(alt_true_rad) * np.cos(tilt_rad)
        az_cam = np.rad2deg(np.arctan2(y, x))

        return alt_cam, (az_cam + 360) % 360

    def predict(self, x, y):
        """Predice (alt, az) para un conjunto de coordenadas de píxel (x, y)."""
        if not self.is_fitted:
            raise RuntimeError("El modelo debe ser ajustado antes de poder predecir.")

        # 1. Centrar coordenadas respecto al cénit
        x_centered = x - self.xc
        y_centered = y - self.yc

        # 2. Corregir rotación del sensor
        theta_rad = np.deg2rad(self.theta_deg)
        cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
        x_rot = x_centered * cos_t - y_centered * sin_t
        y_rot = x_centered * sin_t + y_centered * cos_t

        # 3. Calcular radio y altitud
        r_px = np.sqrt(x_rot**2 + y_rot**2)
        alt = self.radial_distortion(r_px)

        # 4. Calcular azimut
        # Azimut: 0º arriba (Norte), 90º izquierda (Oeste), 180º abajo (Sur), 270º derecha (Este)
        # Se invierten las coordenadas x e y para que el sistema de referencia sea:
        # Norte (0º) -> +y, Oeste (90º) -> -x, Sur (180º) -> -y, Este (270º) -> +x
        az_cam = (np.rad2deg(np.arctan2(-x_rot, -y_rot)) + 360) % 360

        # 5. Convertir de coordenadas de cámara a coordenadas celestes verdaderas
        alt_true, az_true = self._tilted_to_true_coords(alt, az_cam)

        return alt_true, az_true

    def altaz(self, x, y):
        """Alias para predict(x, y)."""
        return self.predict(x, y)

    def xy_inv(self, alt, az):
        """Calcula (x, y) para un conjunto de coordenadas celestes (alt, az)."""
        if not self.is_fitted:
            raise RuntimeError("El modelo debe ser ajustado antes de poder predecir.")

        # 1. Convertir coordenadas celestes verdaderas a coordenadas de cámara
        alt_cam, az_cam = self._true_to_tilted_coords(alt, az)

        # 2. Convertir (alt_cam, az_cam) a coordenadas de píxel relativas al cénit
        r_px = self.radial_distortion_inv(alt_cam)

        # 3. Calcular x e y a partir de r y azimut de cámara
        # Invertimos la transformación del azimut para obtener las coordenadas rotadas
        az_cam_rad = np.deg2rad(az_cam)
        x_rot = -r_px * np.sin(az_cam_rad)
        y_rot = -r_px * np.cos(az_cam_rad)

        # 3. Deshacer la rotación del sensor
        theta_rad = np.deg2rad(self.theta_deg)
        cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
        x_centered = x_rot * cos_t + y_rot * sin_t
        y_centered = -x_rot * sin_t + y_rot * cos_t

        # 4. Deshacer el centrado
        x = x_centered + self.xc
        y = y_centered + self.yc

        return x, y

    def radial_distortion(self, r_px):
        """Aplica el modelo de distorsión radial para obtener la altitud."""
        # El modelo ahora directamente mapea r_px a altitud
        return self.radial_func(r_px, self.coeffs)

    def radial_distortion_inv(self, alt):
        """Calcula la distancia radial a partir de la altitud."""
        alt_arr = np.atleast_1d(alt).astype(float)
        r_grid = np.linspace(0, self.max_radius_px, 5000)
        alt_grid = self.radial_distortion(r_grid)
        sort_idx = np.argsort(alt_grid)[::-1]
        alt_sorted = alt_grid[sort_idx]
        r_sorted = r_grid[sort_idx]
        alt_clamped = np.clip(alt_arr, alt_sorted.min(), alt_sorted.max())
        r_interp = np.interp(alt_clamped, alt_sorted, r_sorted)
        return r_interp if alt_arr.ndim > 0 else float(r_interp)

    def _error_function(self, params, x_px, y_px, alt_true, az_true):
        """Función de coste a minimizar: error angular esférico."""
        # Desempaquetar parámetros
        self.xc, self.yc, self.theta_deg, self.tilt_angle, self.tilt_azimuth = params[:5]
        self.coeffs = np.array(params[5:])
        
        self.is_fitted = True # Temporalmente para poder usar predict

        # Predecir valores
        alt_pred, az_pred = self.predict(x_px, y_px)

        # Calcular distancia angular (ley de los cosenos para trigonometría esférica)
        alt_true_rad = np.deg2rad(alt_true)
        alt_pred_rad = np.deg2rad(alt_pred)
        delta_az_rad = np.deg2rad(az_true - az_pred)

        cos_angle = np.sin(alt_true_rad) * np.sin(alt_pred_rad) + \
                    np.cos(alt_true_rad) * np.cos(alt_pred_rad) * np.cos(delta_az_rad)
        
        # Evitar problemas numéricos con cos_angle > 1
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        
        # Devolvemos la suma de los cuadrados de los errores angulares
        return np.sum(angle_rad**2)

    def fit(
        self,
        x_px: np.ndarray,
        y_px: np.ndarray,
        alt_true: np.ndarray,
        az_true: np.ndarray,
        image_width: int,
        image_height: int,
        initial_params: Iterable[float] | None = None,
    ):
        """Ajusta los parámetros del modelo a los datos proporcionados."""

        x_px = np.asarray(x_px, dtype=float)
        y_px = np.asarray(y_px, dtype=float)
        alt_true = np.asarray(alt_true, dtype=float)
        az_true = np.asarray(az_true, dtype=float)

        if x_px.size < self.n_coeffs + 2:
            raise ValueError("Se necesitan más pares de estrellas identificadas para ajustar el modelo.")

        self.max_radius_px = max(image_width, image_height) / 2 * np.sqrt(2)

        # Valores iniciales
        cx = image_width / 2
        cy = image_height / 2
        theta0 = 0.0
        r_px = np.sqrt((x_px - cx) ** 2 + (y_px - cy) ** 2)
        r_px_max = max(np.max(r_px), 1.0)
        alt_min = np.clip(np.min(alt_true), 0.0, 89.5)
        c1 = (90.0 - alt_min) / r_px_max
        default_coeffs = [c1] + [0.0] * (self.n_coeffs - 1)
        tilt_angle0 = 0.0
        tilt_az0 = 0.0

        if initial_params is None:
            initial_params = [cx, cy, theta0, tilt_angle0, tilt_az0] + default_coeffs
        else:
            initial_params = list(initial_params)

        bounds = [
            (cx - 50, cx + 50),  # xc restringido a 100x100 alrededor del centro
            (cy - 50, cy + 50),  # yc
            (-180, 180),         # theta en grados
            (0, 10),             # tilt_angle máximo 10º
            (0, 360),            # tilt_azimuth en grados
            (0.01, 2.5),         # c1 positivo
        ] + [(-0.05, 0.05)] * (self.n_coeffs - 1)

        result = minimize(
            self._error_function,
            initial_params,
            args=(x_px, y_px, alt_true, az_true),
            method='L-BFGS-B',
            bounds=bounds
        )

        if not result.success:
            raise RuntimeError(f"La optimización del modelo no convergió: {result.message}")

        # Guardar los parámetros óptimos
        self.xc, self.yc, self.theta_deg, self.tilt_angle, self.tilt_azimuth = result.x[:5]
        self.coeffs = np.array(result.x[5:], dtype=float)
        self.is_fitted = True

        return result

def get_model(model_type: str, params: dict) -> LensModel:
    """
    Crea una instancia de LensModel, carga sus parámetros y la devuelve.
    """
    model = LensModel(model_type=model_type, max_radius_px=params.get('max_radius_px'))
    model.load(params)
    return model


def default_model_parameters(model_type: str, image_width: int, image_height: int) -> Dict:
    """Genera parámetros iniciales razonables para un modelo dado."""
    if model_type not in RADIAL_MODELS:
        raise ValueError(f"Modelo '{model_type}' no soportado")

    cx = image_width / 2
    cy = image_height / 2
    r_edge = min(image_width, image_height) / 2
    c1 = 90.0 / max(r_edge, 1.0)
    coeffs = [c1] + [0.0] * (RADIAL_MODELS[model_type].n_coeffs - 1)

    return {
        "model_type": model_type,
        "xc": cx,
        "yc": cy,
        "theta_deg": 0.0,
        "tilt_angle": 0.0,
        "tilt_azimuth": 0.0,
        "coeffs": coeffs,
        "max_radius_px": max(image_width, image_height) / 2 * np.sqrt(2),
    }
