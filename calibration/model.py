import numpy as np
from scipy.optimize import minimize

# --- Funciones de Distorsión Radial ---
# Cada función devuelve la ALTITUD (en grados) dado un radio r (en píxeles)
# y sus coeficientes.

def polynomial_distortion(r, coeffs):
    """Modelo polinómico. alt = 90 - (c1*r + c2*r^2 + ...)"""
    # polyval evalúa p[0]*x**(n-1) + ... + p[n-1]
    # Queremos c1*r + c2*r^2 + ..., así que los coeficientes deben estar en orden inverso
    # y con un cero al principio para el término independiente.
    # ej: [c1, c2] -> polyval([c2, c1, 0], r)
    alt = 90.0 - np.polyval(np.concatenate((coeffs[::-1], [0])), r)
    return alt

def exponential_distortion(r, coeffs):
    """Modelo exponencial. alt = 90 * exp(-c1*r)"""
    c1, = coeffs
    # El coeficiente debe ser positivo para que la altitud disminuya con el radio
    alt = 90.0 * np.exp(-np.abs(c1) * r)
    return alt

RADIAL_MODELS = {
    'lin' : {'func': polynomial_distortion, 'n_coeffs': 1},
    'poly2': {'func': polynomial_distortion, 'n_coeffs': 2},
    'poly3': {'func': polynomial_distortion, 'n_coeffs': 3},
    'exp1': {'func': exponential_distortion, 'n_coeffs': 1},
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
    def __init__(self, model_type='poly3'):
        if model_type not in RADIAL_MODELS:
            raise ValueError(f"Modelo '{model_type}' no soportado. Opciones: {list(RADIAL_MODELS.keys())}")
        
        self.model_type = model_type
        self.radial_func = RADIAL_MODELS[model_type]['func']
        self.n_coeffs = RADIAL_MODELS[model_type]['n_coeffs']

        # Parámetros del modelo (se inicializan en fit)
        self.xc = None
        self.yc = None
        self.theta_deg = None # Rotación en grados
        self.tilt_angle = 0.0 # Inclinación de la cámara en grados
        self.tilt_azimuth = 0.0 # Dirección de la inclinación en grados
        self.coeffs = None
        self.is_fitted = False

    def load(self, params: dict):
        """Carga los parámetros del modelo desde un diccionario."""
        self.xc = params.get('xc')
        self.yc = params.get('yc')
        self.theta_deg = params.get('theta_deg')
        self.tilt_angle = params.get('tilt_angle', 0.0)
        self.tilt_azimuth = params.get('tilt_azimuth', 0.0)
        self.coeffs = np.array(params.get('coeffs', []))
        # El modelo se considera ajustado si todos los parámetros clave están presentes
        self.is_fitted = all(p is not None for p in [self.xc, self.yc, self.theta_deg, self.coeffs])

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
            'coeffs': self.coeffs.tolist() if self.coeffs is not None else []
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
        # Inversión de la función de distorsión radial
        # Para modelos polinómicos, se puede usar np.roots para encontrar las raíces
        # Sin embargo, esto puede ser inestable para polinomios de grado alto.
        # En su lugar, usamos una búsqueda lineal para encontrar el radio que produce la altitud deseada.
        r_px = np.linspace(0, 1520, 10000)  # Radio en píxeles
        alt_pred = self.radial_distortion(r_px)
        idx = np.argmin(np.abs(alt_pred - alt))
        return r_px[idx]

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

    def fit(self, x_px, y_px, alt_true, az_true, image_width, image_height):
        """Ajusta los parámetros del modelo a los datos proporcionados."""
        
        # Valores iniciales y límites para los parámetros
        initial_xc = image_width / 2
        initial_yc = image_height / 2
        initial_theta = 0.0

        # Estimar un valor inicial para el primer coeficiente (relacionado con la escala)
        # alt = 90 - c1*r  => c1 = (90 - alt) / r
        r_px_max = np.max(np.sqrt((x_px - initial_xc)**2 + (y_px - initial_yc)**2))
        alt_min = np.min(alt_true)
        initial_c1 = (90 - alt_min) / r_px_max if r_px_max > 0 else 0.1

        initial_coeffs = [initial_c1] + [0.0] * (self.n_coeffs - 1)

        initial_tilt_angle = 0.0
        initial_tilt_azimuth = 0.0

        # Los parámetros a optimizar son [xc, yc, theta_deg, tilt_angle, tilt_azimuth, ...coeffs]
        initial_params = [initial_xc, initial_yc, initial_theta, initial_tilt_angle, initial_tilt_azimuth] + initial_coeffs

        # Límites para los parámetros (xc, yc, theta, tilt_angle, tilt_azimuth, coeffs)
        bounds = [
            (image_width / 2 - 20, image_width / 2 + 20), # xc
            (image_height / 2 - 20, image_height / 2 + 20), # yc
            (-180, 180), # theta en grados
            (0, 2),      # tilt_angle en grados (límite de 2º)
            (0, 360),    # tilt_azimuth en grados
            (-1000, 1000), # Límite para c1
        ] + [(-0.1, 0.1)] * (self.n_coeffs - 1)

        # Realizar la optimización
        result = minimize(
            self._error_function,
            initial_params,
            args=(x_px, y_px, alt_true, az_true),
            method='L-BFGS-B',
            bounds=bounds
        )

        if not result.success:
            print(f"¡Advertencia! La optimización falló: {result.message}")

        # Guardar los parámetros óptimos
        self.xc, self.yc, self.theta_deg, self.tilt_angle, self.tilt_azimuth = result.x[:5]
        self.coeffs = result.x[5:]
        self.is_fitted = True
        
        print("Ajuste del modelo completado.")
        print(f"  - Cénit (xc, yc): ({self.xc:.2f}, {self.yc:.2f}) px")
        print(f"  - Rotación (theta): {self.theta_deg:.2f}º")
        print(f"  - Inclinación (ángulo, azimut): {self.tilt_angle:.2f}º, {self.tilt_azimuth:.2f}º")
        print(f"  - Coeficientes radiales: {np.round(self.coeffs, 5)}")

        return result

def get_model(model_type: str, params: dict) -> LensModel:
    """
    Crea una instancia de LensModel, carga sus parámetros y la devuelve.
    """
    model = LensModel(model_type=model_type)
    model.load(params)
    return model
