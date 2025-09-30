"""Lens calibration models for the all-sky plate solving web application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
from scipy.optimize import minimize

# --- Funciones de Distorsión Radial ---
# Cada función devuelve la ALTITUD (en grados) dado un radio r (en píxeles)
# y sus coeficientes.


def polynomial_distortion(r_px: np.ndarray, coeffs: np.ndarray, R: float) -> np.ndarray:
    """Modelo polinómico siguiendo la fórmula: alt = π/2 * (c + b*r + a*r²)/R
    donde a, b, c son los coeficientes y R es el radio de referencia."""
    if coeffs.size < 3:
        # Rellenar con valores por defecto si no hay suficientes coeficientes
        coeffs_full = np.zeros(3)
        coeffs_full[:coeffs.size] = coeffs
        if coeffs.size == 0:
            coeffs_full = np.array([0.2, 0.8, 1.0])  # valores por defecto a, b, c
        coeffs = coeffs_full
    
    a, b, c = coeffs[0], coeffs[1], coeffs[2]
    print('a =', a, 'b =', b, 'c =', c)
    # Aplicar signos según el modelo original
    a = -a / R
    b = -b
    c = c * R
    
    alt_rad = np.pi/2 - (c + b*r_px + a*r_px**2) / R
    return np.rad2deg(alt_rad)


def exponential_distortion(r_norm: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Modelo exponencial. alt = 90 * exp(-|c1|*r_norm).
    r_norm es el radio normalizado (r_px / (W/2)) donde W es la dimensión más corta."""
    if coeffs.size == 0:
        return np.full_like(r_norm, 90.0, dtype=float)
    c1 = np.abs(coeffs[0])
    return 90.0 * np.exp(-c1 * r_norm)


@dataclass(frozen=True)
class RadialModelInfo:
    key: str
    name: str
    func: callable
    n_coeffs: int


RADIAL_MODELS: Dict[str, RadialModelInfo] = {
    "lin": RadialModelInfo("lin", "Lineal", polynomial_distortion, 1),
    "poly2": RadialModelInfo("poly2", "Polinómico (2º)", polynomial_distortion, 3),
    "poly3": RadialModelInfo("poly3", "Polinómico (3º)", polynomial_distortion, 4),
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
        self.norm_factor = 1520  # Factor de normalización W/2
        self.R = 1520  # Radio de referencia para el modelo polinómico

    def load(self, params: dict):
        """Carga los parámetros del modelo desde un diccionario."""
        self.xc = params.get('xc', 1520)
        self.yc = params.get('yc', 1520)
        self.theta_deg = params.get('theta_deg', 0.0)
        self.tilt_angle = params.get('tilt_angle', 0.0)
        self.tilt_azimuth = params.get('tilt_azimuth', 0.0)
        coeffs = params.get('coeffs', [])
        self.coeffs = np.array(coeffs, dtype=float) if coeffs is not None else np.zeros(self.n_coeffs)
        self.max_radius_px = params.get('max_radius_px', self.max_radius_px)
        self.norm_factor = params.get('norm_factor', self.norm_factor)
        self.R = params.get('R', self.R)

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
            'norm_factor': self.norm_factor,
            'R': self.R,
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


        x_centered = x - self.xc
        y_centered = y - self.yc

        # Ángulo base medido desde el eje X, corregido por la rotación
        theta_rad = np.deg2rad(self.theta_deg)

        cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
        x_rot = x_centered * cos_t - y_centered * sin_t
        y_rot = x_centered * sin_t + y_centered * cos_t

        az_cam = (np.rad2deg(np.arctan2(-x_rot, -y_rot)) + 360) % 360

        # Calcular radio y azimut en el plano de la imagen
        r_px = np.sqrt(x_rot**2 + y_rot**2)
        alt = self.radial_distortion(r_px)
        

        # Convertir de coordenadas de cámara a coordenadas celestes verdaderas (corrección de inclinación)
        alt_true, az_true = self._tilted_to_true_coords(alt, az_cam)

        return alt_true, az_true

    def altaz(self, x, y):
        """Alias para predict(x, y)."""
        return self.predict(x, y)

    def xy_inv(self, alt, az):
        """Convierte coordenadas celestes (altitud, azimut) a coordenadas de píxel (x, y)."""
        if not self.is_fitted:
            raise RuntimeError("El modelo debe ser ajustado antes de poder predecir.")

        # 1. Convertir de coordenadas celestes a coordenadas de cámara (corrección de inclinación)
        alt_cam, az_cam = self._true_to_tilted_coords(alt, az)

        # 2. Calcular la distancia radial a partir de la altitud de la cámara
        r_px = self.radial_distortion_inv(alt_cam)

        # 3. Convertir a coordenadas cartesianas, aplicando la rotación del sensor
        az_cam_rad = np.deg2rad(az_cam)
        x_rot = -r_px * np.sin(az_cam_rad)
        y_rot = -r_px * np.cos(az_cam_rad)
        
        theta_rad = np.deg2rad(self.theta_deg)
        cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
        x_centered = x_rot * cos_t + y_rot * sin_t
        y_centered = -x_rot * sin_t + y_rot * cos_t

        x = x_centered + self.xc
        y = y_centered + self.yc

        return x, y

    def radial_distortion(self, r_px):
        """Aplica el modelo de distorsión radial para obtener la altitud."""
        if self.model_type.startswith('poly'):
            return polynomial_distortion(r_px, self.coeffs, self.R)
        else:
            # Para modelos exponenciales, usar normalización
            r_norm = r_px / self.norm_factor
            return self.radial_func(r_norm, self.coeffs)

    def radial_distortion_inv(self, alt):
        """Calcula el radio en píxeles a partir de la altitud."""
        if self.model_type.startswith('poly'):
            return self._polynomial_r_inv(alt)
        else:
            # Para modelos exponenciales, usar interpolación
            alt_arr = np.atleast_1d(alt).astype(float)
            max_r_norm = self.max_radius_px / self.norm_factor
            r_norm_grid = np.linspace(0, max_r_norm, 5000)
            alt_grid = self.radial_distortion(r_norm_grid * self.norm_factor)
            sort_idx = np.argsort(alt_grid)[::-1]
            alt_sorted = alt_grid[sort_idx]
            r_norm_sorted = r_norm_grid[sort_idx]
            alt_clamped = np.clip(alt_arr, alt_sorted.min(), alt_sorted.max())
            r_norm_interp = np.interp(alt_clamped, alt_sorted, r_norm_sorted)
            return (r_norm_interp * self.norm_factor) if alt_arr.ndim > 0 else float(r_norm_interp * self.norm_factor)

    def _polynomial_r_inv(self, alt):
        """Calcula la distancia radial a partir de la altitud (inversa de `polynomial_distortion`)."""
        alt_arr = np.atleast_1d(alt).astype(float)
        
        if self.coeffs.size < 3:
            coeffs_full = np.zeros(3)
            coeffs_full[:self.coeffs.size] = self.coeffs
            if self.coeffs.size < 3:
                coeffs_full[2] = 1.0  # c es 1.0 si no se especifica
            coeffs = coeffs_full
        else:
            coeffs = self.coeffs

        r_px = np.linspace(0, 1520, 10000)  # Radio en píxeles
        alt_pred = self.radial_distortion(r_px)
        
        # Manejar múltiples valores de altitud
        result = np.zeros_like(alt_arr)
        for i, alt_val in enumerate(alt_arr):
            idx = np.argmin(np.abs(alt_pred - alt_val))
            result[i] = r_px[idx]
        
        return float(result[0]) if np.isscalar(alt) else result

    def _error_function(self, params, x_px, y_px, alt_true, az_true):
        """Función de coste a minimizar: distancia cuadrática entre puntos (x,y) detectados y esperados."""
        # Desempaquetar parámetros
        self.xc, self.yc, self.theta_deg, self.tilt_angle, self.tilt_azimuth = params[:5]
        self.coeffs = np.array(params[5:])
        
        self.is_fitted = True # Temporalmente para poder usar xy_inv

        x_expected, y_expected = self.xy_inv(alt_true, az_true)
            
        # Calcular distancia cuadrática entre puntos detectados y esperados
        return np.sum((y_px - x_expected)**2 + (x_px - y_expected)**2)

    def fit(
        self,
        x_px: np.ndarray,
        y_px: np.ndarray,
        alt_true: np.ndarray,
        az_true: np.ndarray,
        image_width: int,
        image_height: int,
        initial_params=None,
        custom_bounds=None,
    ):
        """Ajusta el modelo a partir de pares de puntos (x,y) detectados y sus coordenadas celestes (alt, az)."""
        x_px = np.asarray(x_px, dtype=float)
        y_px = np.asarray(y_px, dtype=float)
        alt_true = np.asarray(alt_true, dtype=float)
        az_true = np.asarray(az_true, dtype=float)

        if x_px.size < self.n_coeffs + 2:
            raise ValueError("Se necesitan más pares de estrellas identificadas para ajustar el modelo.")

        # Establecer factor de normalización: W/2 donde W es la dimensión más corta
        W = min(image_width, image_height)
        self.norm_factor = W / 2
        self.R = self.norm_factor  # Radio de referencia para modelo polinómico
        self.max_radius_px = max(image_width, image_height) / 2 * np.sqrt(2)

        # Valores iniciales
        cx = image_width / 2
        cy = image_height / 2
        theta0 = 0.0
        
        # Coeficientes iniciales para modelo polinómico siguiendo polymodel
        if self.model_type.startswith('poly'):
            # Valores por defecto del polymodel: a=0.2, b=0.8, c=1.0
            default_coeffs = [0.2, 0.8, 1.0][:self.n_coeffs]
            if len(default_coeffs) < self.n_coeffs:
                default_coeffs.extend([0.0] * (self.n_coeffs - len(default_coeffs)))
        else:
            c1 = 0.8
            default_coeffs = [c1] + [0.0] * (self.n_coeffs - 1)
            
        tilt_angle0 = 0.0
        tilt_az0 = 0.0

        if initial_params is None:
            initial_params = [cx, cy, theta0, tilt_angle0, tilt_az0] + default_coeffs
        else:
            initial_params = list(initial_params)

        # Default bounds
        default_bounds = [
            (cx - 50, cx + 50),  # xc con más rango
            (cy - 50, cy + 50),  # yc con más rango
            (-180, 180),         # theta en grados
            (0, 10),             # tilt_angle máximo 10º
            (0, 360),            # tilt_azimuth en grados completos
            (0.01, 3.0),         # coeffA (a)
            (0.1, 2.0),          # coeffB (b)
            (0.5, 2.0),          # coeffC (c)
        ]
        
        # Add bounds for additional coefficients if needed
        if self.n_coeffs > 3:
            default_bounds.extend([(0, 2.0)] * (self.n_coeffs - 3))
        
        # Apply custom bounds if provided
        bounds = default_bounds.copy()
        if custom_bounds:
            param_names = ['xc', 'yc', 'theta', 'tiltAngle', 'tiltAzimuth', 'coeffA'] + [f'coeff{chr(66+i)}' for i in range(self.n_coeffs-1)]
            for i, param_name in enumerate(param_names):
                if param_name in custom_bounds and i < len(bounds):
                    bounds[i] = tuple(custom_bounds[param_name])

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
    W = min(image_width, image_height)
    norm_factor = W / 2
    R = norm_factor  # Radio de referencia para modelo polinómico
    
    # Coeficientes según el tipo de modelo
    if model_type.startswith('poly'):
        # Valores por defecto del polymodel: a=0.2, b=0.8, c=1.0
        # Para poly2 (2 coeficientes), usar solo a y b, pero c debe ser 1.0 internamente
        n_coeffs = RADIAL_MODELS[model_type].n_coeffs
        if n_coeffs == 2:
            coeffs = [0.2, 0.8]  # Solo a y b, c=1.0 se manejará internamente
        else:
            coeffs = [0.2, 0.8, 1.0][:n_coeffs]
            if len(coeffs) < n_coeffs:
                coeffs.extend([0.0] * (n_coeffs - len(coeffs)))
    else:
        # Para modelos exponenciales
        c1 = 0.8
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
        "norm_factor": norm_factor,
        "R": R,
    }
