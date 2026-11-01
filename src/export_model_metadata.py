"""
export_model_metadata.py

Menulis models/model_metadata.yaml: referensi ke model Production di
MLflow Model Registry, hash artefak model (dari file .dvc), dan hash
dataset yang menjadi asal data pelatihan (data lineage).
"""

import glob
import hashlib
import os
import subprocess

import mlflow
import yaml
from minio import Minio
from mlflow import MlflowClient

NAME = "pm25-forecasting-xgboost"
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC_DIR)
mlflow.set_tracking_uri(
    f"sqlite:///{os.path.join(SRC_DIR, 'mlflow.db')}"
)


def dvc_md5(path):
    """Membaca nilai md5 dari file penunjuk .dvc."""
    with open(path) as f:
        return yaml.safe_load(f)["outs"][0]["md5"]


def processed_md5():
    """Menghitung md5 data processed di MinIO saat ini."""
    try:
        client = Minio("localhost:9000", access_key="minioadmin",
                       secret_key="minioadmin", secure=False)
        resp = client.get_object(
            "mlops-airquality-bucket",
            "data/processed/data_processed_latest.parquet",
        )
        digest = hashlib.md5(resp.read()).hexdigest()
        resp.close()
        resp.release_conn()
        return digest
    except Exception as exc:
        return f"tidak dapat dihitung ({type(exc).__name__})"


def main():
    """Membangun dan menyimpan file metadata model."""
    client = MlflowClient()
    ver = client.get_model_version_by_alias(NAME, "production")
    run = client.get_run(ver.run_id)
    params, metrics = run.data.params, run.data.metrics

    raw_dvc = sorted(glob.glob(os.path.join(ROOT, "data/raw/*.csv.dvc")))
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()

    meta = {
        "registered_model": {
            "name": NAME,
            "version": int(ver.version),
            "alias": "production",
            "stage": ver.current_stage,
            "source_run_id": ver.run_id,
            "source_run_name": run.info.run_name,
            "model_uri": f"models:/{NAME}@production",
        },
        "hyperparameters": {
            "n_estimators": int(params["n_estimators"]),
            "max_depth": int(params["max_depth"]),
            "learning_rate": float(params["learning_rate"]),
        },
        "metrics_test_set": {
            "mae": round(metrics["mae"], 3),
            "rmse": round(metrics["rmse"], 3),
            "r2": round(metrics["r2"], 3),
            "test_rows": int(params["test_rows"]),
        },
        "model_artifact_dvc": {
            "pointer_file": "models/pm25-xgboost-production.dvc",
            "md5": dvc_md5(
                os.path.join(ROOT, "models/pm25-xgboost-production.dvc")
            ),
        },
        "data_lineage": {
            "training_data": "data/processed/data_processed_latest.parquet",
            "training_data_md5_at_export": processed_md5(),
            "dvc_tracked_raw_datasets": {
                os.path.basename(p): dvc_md5(p) for p in raw_dvc
            },
            "note": (
                "Model dilatih dari data processed (gabungan file raw "
                "hasil ingestion). Hash data processed dihitung saat "
                "metadata ini dibuat, bukan saat training."
            ),
        },
        "git_commit_at_export": git_commit,
    }

    out = os.path.join(ROOT, "models", "model_metadata.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)
    print("Tersimpan:", out)


if __name__ == "__main__":
    main()
