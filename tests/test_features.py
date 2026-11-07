"""Unit test pembentukan fitur dan pembagian data (src/train.py)."""

import numpy as np
import pandas as pd
import pytest

from train import (LAG_HOURS, WEATHER_FEATURES, build_features,
                   get_feature_columns, split_time_series)


def make_df(n_hours=60):
    """Data tiruan per jam; pm2_5 = 0, 1, 2, ... agar mudah dicek."""
    df = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=n_hours, freq="h"),
        "pm2_5": np.arange(n_hours, dtype=float),
    })
    for col in WEATHER_FEATURES:
        df[col] = 1.0
    return df


def test_jumlah_baris_setelah_feature_engineering():
    """Baris awal (lag) dan baris akhir (target) harus dibuang."""
    data = build_features(make_df(60))
    assert len(data) == 60 - max(LAG_HOURS) - 1


def test_nilai_lag_dan_target_benar():
    """Lag mengambil masa lalu, target mengambil jam berikutnya."""
    first = build_features(make_df(60)).iloc[0]
    assert first["pm2_5"] == 24
    assert first["pm2_5_lag_1"] == 23
    assert first["pm2_5_lag_24"] == 0
    assert first["target_pm2_5_t1"] == 25


def test_target_adalah_nilai_jam_berikutnya():
    """Target tidak boleh memakai nilai jam yang sama atau sebelumnya."""
    data = build_features(make_df(60))
    assert (data["target_pm2_5_t1"] == data["pm2_5"] + 1).all()


def test_tidak_ada_nan_setelah_feature_engineering():
    data = build_features(make_df(60))
    assert not data.isna().any().any()


def test_fitur_siklis_dalam_rentang():
    """Fitur sin/cos di [-1, 1]; jam 00:00 -> sin 0 dan cos 1."""
    data = build_features(make_df(60))
    for col in ("hour_sin", "hour_cos", "dow_sin", "dow_cos"):
        assert data[col].between(-1, 1).all()
    assert data.iloc[0]["hour_sin"] == pytest.approx(0.0, abs=1e-9)
    assert data.iloc[0]["hour_cos"] == pytest.approx(1.0)


def test_get_feature_columns():
    """Ada 20 fitur, semuanya ada di data, dan target bukan fitur."""
    cols = get_feature_columns()
    data = build_features(make_df(60))
    assert len(cols) == 20
    assert set(cols).issubset(data.columns)
    assert "target_pm2_5_t1" not in cols


def test_split_default_train_lebih_besar():
    """Rasio bawaan 0.85: data latih harus lebih banyak dari data uji."""
    data = build_features(make_df(200))
    train, test = split_time_series(data)
    assert len(train) == int(len(data) * 0.85)
    assert len(train) + len(test) == len(data)
    assert len(train) > len(test)


def test_split_berbasis_waktu_tanpa_kebocoran():
    """Seluruh data latih harus lebih awal dari seluruh data uji."""
    data = build_features(make_df(200))
    train, test = split_time_series(data)
    assert train["time"].max() < test["time"].min()
