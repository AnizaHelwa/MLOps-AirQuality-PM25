"""
validate_model.py

Gerbang validasi otomatis sebelum model masuk registry. Model hasil
pelatihan terbaru dinyatakan lolos jika MAE dan RMSE-nya lebih baik
daripada baseline naif (prediksi PM2.5 jam depan = PM2.5 jam ini) pada
test set yang sama. Target perencanaan (MAE <= 10, RMSE <= 15) hanya
dicetak sebagai informasi dan tidak memblokir.

Kode keluar: 0 = lolos, 1 = gagal, 2 = run tidak ditemukan.
"""

import sys

import numpy as np
from mlflow import MlflowClient

from train import build_features, load_processed_data, split_time_series

EXPERIMENT = "pm25-forecasting-t1"
TARGET_MAE = 10.0
TARGET_RMSE = 15.0


def naive_baseline_metrics():
    """Menghitung MAE dan RMSE baseline naif pada test set."""
    data = build_features(load_processed_data())
    _, test = split_time_series(data)
    error = test["target_pm2_5_t1"] - test["pm2_5"]
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    return mae, rmse


def evaluate_gate(model_mae, model_rmse, naive_mae, naive_rmse):
    """
    Menentukan apakah model lolos validasi.

    Returns:
        Tuple (lolos, alasan): lolos True jika MAE dan RMSE model
        keduanya lebih rendah dari baseline naif.
    """
    reasons = []
    if not model_mae < naive_mae:
        reasons.append(
            f"MAE {model_mae:.3f} tidak lebih baik dari baseline "
            f"{naive_mae:.3f}")
    if not model_rmse < naive_rmse:
        reasons.append(
            f"RMSE {model_rmse:.3f} tidak lebih baik dari baseline "
            f"{naive_rmse:.3f}")
    return len(reasons) == 0, reasons


def main():
    """Membaca run terbaru, menjalankan gerbang, dan mencetak hasil."""
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    runs = []
    if exp is not None:
        runs = client.search_runs(
            [exp.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1)
    if not runs:
        print(f"Tidak ada run di eksperimen '{EXPERIMENT}'")
        sys.exit(2)

    run = runs[0]
    m = run.data.metrics
    naive_mae, naive_rmse = naive_baseline_metrics()
    passed, reasons = evaluate_gate(
        m["mae"], m["rmse"], naive_mae, naive_rmse)

    print(f"Run dievaluasi : {run.info.run_name} ({run.info.run_id[:8]})")
    print(f"MAE   model={m['mae']:.3f} | baseline={naive_mae:.3f}")
    print(f"RMSE  model={m['rmse']:.3f} | baseline={naive_rmse:.3f}")
    target_ok = m["mae"] <= TARGET_MAE and m["rmse"] <= TARGET_RMSE
    status = "tercapai" if target_ok else "belum tercapai"
    print(f"Target README (informasi, tidak memblokir): "
          f"MAE<={TARGET_MAE}, RMSE<={TARGET_RMSE} -> {status}")
    if passed:
        print("HASIL VALIDASI: LOLOS")
        sys.exit(0)
    print("HASIL VALIDASI: GAGAL")
    for r in reasons:
        print(" -", r)
    sys.exit(1)


if __name__ == "__main__":
    main()
