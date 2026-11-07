"""Unit test pemuatan data lewat LOCAL_DATA_PATH (src/train.py)."""

import pandas as pd

from train import WEATHER_FEATURES, load_processed_data


def test_load_dari_file_lokal(tmp_path, monkeypatch):
    """Dengan LOCAL_DATA_PATH, data dibaca dari file tanpa MinIO."""
    df = pd.DataFrame({
        "time": ["2026-01-01T02:00", "2026-01-01T00:00",
                 "2026-01-01T01:00"],
        "pm2_5": [3.0, 1.0, 2.0],
    })
    for col in WEATHER_FEATURES:
        df[col] = 1.0
    file = tmp_path / "data.parquet"
    df.to_parquet(file)

    monkeypatch.setenv("LOCAL_DATA_PATH", str(file))
    out = load_processed_data()

    assert pd.api.types.is_datetime64_any_dtype(out["time"])
    assert out["time"].is_monotonic_increasing
    assert list(out["pm2_5"]) == [1.0, 2.0, 3.0]
