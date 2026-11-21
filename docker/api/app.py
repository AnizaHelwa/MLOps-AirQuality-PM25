"""
app.py

API inferensi PM2.5 (t+1). Model dimuat dari MLflow Model Registry
lewat alias (default: production) memakai nama layanan mlflow-server
pada jaringan Docker Compose.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, List

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
)

MODEL_NAME = os.environ.get("MODEL_NAME", "pm25-forecasting-xgboost")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "production")
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "(default)")

# Urutan kolom harus sama dengan get_feature_columns() di src/train.py
FEATURE_COLUMNS = [
    "pm2_5_lag_1", "pm2_5_lag_3", "pm2_5_lag_6", "pm2_5_lag_24",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "pm10", "ozone", "sulphur_dioxide", "nitrogen_dioxide",
    "carbon_monoxide", "temperature_2m", "relative_humidity_2m",
    "windspeed_10m", "winddirection_10m", "surface_pressure",
    "precipitation", "cloudcover",
]

state = {"model": None, "version": None, "error": None}


def load_model():
    """Memuat model beralias Production dari registry ke memori."""
    uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    try:
        version = MlflowClient().get_model_version_by_alias(
            MODEL_NAME, MODEL_ALIAS)
        state["model"] = mlflow.pyfunc.load_model(uri)
        state["version"] = str(version.version)
        state["error"] = None
        logging.info(
            f"Model dimuat dari {uri} (v{version.version}) "
            f"lewat tracking server {TRACKING_URI}")
    except Exception as exc:
        state["model"] = None
        state["error"] = f"{type(exc).__name__}: {exc}"
        logging.error(f"Gagal memuat model {uri}: {state['error']}")


@asynccontextmanager
async def lifespan(app):
    """Memuat model saat API dinyalakan."""
    load_model()
    yield


app = FastAPI(title="PM2.5 Forecast API", lifespan=lifespan)


class PredictRequest(BaseModel):
    """Daftar baris fitur; tiap baris berisi 20 fitur bernama."""

    records: List[Dict[str, float]]


def ensure_model():
    """Memastikan model siap; mencoba memuat ulang jika belum."""
    if state["model"] is None:
        load_model()
    if state["model"] is None:
        raise HTTPException(status_code=503, detail=state["error"])


@app.get("/health")
def health():
    """Status API dan model yang sedang dipakai."""
    ensure_model()
    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "model_version": state["version"],
        "tracking_uri": TRACKING_URI,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    """Memprediksi PM2.5 satu jam ke depan untuk tiap baris fitur."""
    ensure_model()
    if not req.records:
        raise HTTPException(status_code=422, detail="records kosong")
    df = pd.DataFrame(req.records)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422, detail=f"Fitur hilang: {missing}")
    df = df[FEATURE_COLUMNS].astype("float64")
    preds = state["model"].predict(df)
    return {
        "model_version": state["version"],
        "predictions_pm2_5_t_plus_1": [float(p) for p in preds],
    }
