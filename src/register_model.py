"""
register_model.py

Mendaftarkan model dari run pelatihan terbaru ke MLflow Model Registry
dan menaikkannya ke status Staging. Dijalankan hanya jika validasi model
(validate_model.py) dinyatakan lolos.
"""

import mlflow
from mlflow import MlflowClient

EXPERIMENT = "pm25-forecasting-t1"
MODEL_NAME = "pm25-forecasting-xgboost"


def main():
    """Mendaftarkan model run terbaru dan menetapkannya ke Staging."""
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    runs = []
    if exp is not None:
        runs = client.search_runs(
            [exp.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1)
    if not runs:
        raise SystemExit(f"Tidak ada run di eksperimen '{EXPERIMENT}'")

    run = runs[0]
    m = run.data.metrics
    version = mlflow.register_model(
        f"runs:/{run.info.run_id}/model", MODEL_NAME)
    ver = version.version

    client.transition_model_version_stage(MODEL_NAME, ver, "Staging")
    client.set_registered_model_alias(MODEL_NAME, "staging", ver)
    client.set_model_version_tag(
        MODEL_NAME, ver, "validation", "passed_naive_baseline")
    client.update_model_version(
        MODEL_NAME, ver,
        description=(
            f"Run '{run.info.run_name}'. MAE={m['mae']:.3f}, "
            f"RMSE={m['rmse']:.3f}, R2={m['r2']:.3f}. Lolos validasi "
            f"otomatis (lebih baik dari baseline naif), didaftarkan "
            f"oleh pipeline."))
    print(f"Model '{MODEL_NAME}' v{ver} didaftarkan ke Staging "
          f"(run {run.info.run_name})")


if __name__ == "__main__":
    main()
