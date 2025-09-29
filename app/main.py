"""FastAPI application powering the all-sky calibration frontend."""

from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
from PIL import Image

from calibration.model import RADIAL_MODELS, LensModel, default_model_parameters, get_model
from calibration.observer import create_observer, radec_to_altaz
from calibration.stars import load_catalog, lookup_star, project_stars
from calibration.utils import run_daostar_finder


FRONTEND_DIR = Path("frontend")

app = FastAPI(title="All-Sky Plate Solve", version="0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


class Observation(BaseModel):
    latitude: float
    longitude: float
    elevation: float = 0.0
    capture_time: datetime = Field(alias="captureTime")


class ModelParameters(BaseModel):
    model_type: str = Field(alias="modelType")
    xc: float
    yc: float
    theta_deg: float = Field(alias="theta")
    tilt_angle: float = Field(alias="tiltAngle")
    tilt_azimuth: float = Field(alias="tiltAzimuth")
    coeffs: List[float]
    max_radius_px: Optional[float] = Field(default=None, alias="maxRadiusPx")

    @validator("model_type")
    def validate_model(cls, value: str) -> str:
        if value not in RADIAL_MODELS:
            raise ValueError(f"Modelo '{value}' no soportado")
        return value

    def to_model(self) -> LensModel:
        params = {
            "model_type": self.model_type,
            "xc": self.xc,
            "yc": self.yc,
            "theta_deg": self.theta_deg,
            "tilt_angle": self.tilt_angle,
            "tilt_azimuth": self.tilt_azimuth,
            "coeffs": self.coeffs,
        }
        if self.max_radius_px is not None:
            params["max_radius_px"] = self.max_radius_px
        return get_model(self.model_type, params)

    @classmethod
    def from_model(cls, model: LensModel) -> "ModelParameters":
        params = model.params()
        return cls(
            modelType=params["model_type"],
            xc=params["xc"],
            yc=params["yc"],
            theta=params["theta_deg"],
            tiltAngle=params["tilt_angle"],
            tiltAzimuth=params["tilt_azimuth"],
            coeffs=params["coeffs"],
            maxRadiusPx=params.get("max_radius_px"),
        )


class InitialModelRequest(BaseModel):
    image_width: int = Field(alias="imageWidth")
    image_height: int = Field(alias="imageHeight")
    model_type: str = Field(alias="modelType", default="poly2")


class ProjectRequest(BaseModel):
    observation: Observation
    model: ModelParameters
    image_width: int = Field(alias="imageWidth")
    image_height: int = Field(alias="imageHeight")
    min_altitude: float = Field(default=0.0, alias="minAltitude")


class DetectionRequest(BaseModel):
    image_data: str = Field(alias="imageData")
    flip_horizontal: bool = Field(alias="flipHorizontal", default=False)
    fwhm: float = 5.0
    threshold_sigma: float = Field(alias="threshold", default=5.0)
    sharplo: float = Field(default=0.2)
    sharphi: float = Field(default=1.0)
    roundlo: float = Field(default=-1.0)
    roundhi: float = Field(default=1.0)
    limit: int = Field(default=50)


class FitMatch(BaseModel):
    name: str
    x: float
    y: float


class FitRequest(BaseModel):
    observation: Observation
    matches: List[FitMatch]
    model_type: str = Field(alias="modelType", default="poly2")
    image_width: int = Field(alias="imageWidth")
    image_height: int = Field(alias="imageHeight")


def decode_image(image_data: str) -> np.ndarray:
    if "," in image_data:
        _, image_data = image_data.split(",", 1)
    binary = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(binary))
    return np.array(image)


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend no encontrado")
    return index_path.read_text(encoding="utf-8")


@app.get("/api/radial-models")
async def list_radial_models():
    return [
        {
            "key": info.key,
            "name": info.name,
            "coefficients": info.n_coeffs,
        }
        for info in RADIAL_MODELS.values()
    ]


@app.get("/api/star-names")
async def list_star_names():
    return sorted(star.name for star in load_catalog())


@app.post("/api/initial-model")
async def initial_model(payload: InitialModelRequest):
    params = default_model_parameters(payload.model_type, payload.image_width, payload.image_height)
    return params


@app.post("/api/projected-stars")
async def projected_stars(payload: ProjectRequest):
    model = payload.model.to_model()
    try:
        if not model.is_fitted:
            raise HTTPException(status_code=400, detail="El modelo no está inicializado")
        stars = project_stars(
            model,
            payload.observation.latitude,
            payload.observation.longitude,
            payload.observation.elevation,
            payload.observation.capture_time,
            payload.min_altitude,
        )
        return {
            "stars": stars,
            "model": model.params(),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/detect-stars")
async def detect_stars(payload: DetectionRequest):
    try:
        image_array = decode_image(payload.image_data)
        detections = run_daostar_finder(
            image_array,
            payload.fwhm,
            payload.threshold_sigma,
            payload.sharplo,
            payload.sharphi,
            payload.roundlo,
            payload.roundhi,
            payload.limit,
            flip_horizontal=payload.flip_horizontal,
        )
        height, width = image_array.shape[:2]
        return {
            "detections": detections,
            "imageWidth": width,
            "imageHeight": height,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/fit-model")
async def fit_model(payload: FitRequest):
    if not payload.matches:
        raise HTTPException(status_code=400, detail="Debe proporcionar al menos una estrella identificada")

    try:
        obs = payload.observation
        model_params = default_model_parameters(payload.model_type, payload.image_width, payload.image_height)
        model = get_model(payload.model_type, model_params)

        observer = create_observer(obs.latitude, obs.longitude, obs.elevation, obs.capture_time)
        x_px = []
        y_px = []
        altitudes = []
        azimuths = []
        for match in payload.matches:
            star = lookup_star(match.name)
            alt, az = radec_to_altaz(observer, star.ra_deg, star.dec_deg)
            if alt < 0:
                raise ValueError(f"La estrella {star.name} está por debajo del horizonte en la fecha indicada")
            x_px.append(match.x)
            y_px.append(match.y)
            altitudes.append(alt)
            azimuths.append(az)

        model.fit(np.array(x_px), np.array(y_px), np.array(altitudes), np.array(azimuths), payload.image_width, payload.image_height)
        stars = project_stars(
            model,
            obs.latitude,
            obs.longitude,
            obs.elevation,
            obs.capture_time,
        )

        return {
            "model": model.params(),
            "projectedStars": stars,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
