"""Unit test gerbang validasi model (src/validate_model.py)."""

import numpy as np
import pandas as pd
import pytest

from train import WEATHER_FEATURES
from validate_model import evaluate_gate, naive_baseline_metrics


def test_lolos_jika_lebih_baik_dari_baseline():
    lolos, alasan = evaluate_gate(10.0, 15.0, 12.0, 20.0)
    assert lolos is True
    assert alasan == []


def test_gagal_jika_mae_tidak_lebih_baik():
    lolos, alasan = evaluate_gate(13.0, 15.0, 12.0, 20.0)
    assert lolos is False
    assert len(alasan) == 1 and "MAE" in alasan[0]


def test_gagal_jika_rmse_tidak_lebih_baik():
    lolos, alasan = evaluate_gate(10.0, 21.0, 12.0, 20.0)
    assert lolos is False
    assert len(alasan) == 1 and "RMSE" in alasan[0]


def test_gagal_jika_sama_persis_dengan_baseline():
    """Harus benar-benar lebih baik, bukan sekadar sama."""
    lolos, _ = evaluate_gate(12.0, 20.0, 12.0, 20.0)
    assert lolos is False


def test_baseline_naif_pada_data_linear(tmp_path, monkeypatch):
    """pm2_5 naik 1 per jam, jadi tebakan naif selalu meleset tepat 1."""
    df = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=200, freq="h"),
        "pm2_5": np.arange(200, dtype=float),
    })
    for col in WEATHER_FEATURES:
        df[col] = 1.0
    file = tmp_path / "data.parquet"
    df.to_parquet(file)
    monkeypatch.setenv("LOCAL_DATA_PATH", str(file))

    mae, rmse = naive_baseline_metrics()
    assert mae == pytest.approx(1.0)
    assert rmse == pytest.approx(1.0)
