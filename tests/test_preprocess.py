"""Unit test pembersihan dan validasi data (src/preprocess.py)."""

import numpy as np
import pandas as pd
import pytest

from preprocess import REQUIRED_COLUMNS, clean_data, validate_schema

NUMERIC = [c for c in REQUIRED_COLUMNS if c != "time"]


def make_raw(n=6):
    """Data mentah tiruan; time berupa teks seperti hasil ingestion."""
    waktu = pd.date_range("2026-01-01", periods=n, freq="h")
    df = pd.DataFrame({"time": waktu.strftime("%Y-%m-%dT%H:%M")})
    for col in NUMERIC:
        df[col] = 10.0
    df["relative_humidity_2m"] = 70.0
    df["cloudcover"] = 40.0
    df["winddirection_10m"] = 180.0
    return df


def test_time_menjadi_datetime():
    out = clean_data(make_raw())
    assert pd.api.types.is_datetime64_any_dtype(out["time"])


def test_duplikat_dibuang_versi_terbaru_dipakai():
    """Timestamp ganda dibuang; nilai dari data yang masuk terakhir dipakai."""
    lama = make_raw(3)
    baru = make_raw(3)
    baru["pm2_5"] = 99.0
    out = clean_data(pd.concat([lama, baru], ignore_index=True))
    assert len(out) == 3
    assert out["time"].is_unique
    assert (out["pm2_5"] == 99.0).all()


def test_nan_diinterpolasi():
    raw = make_raw(6)
    raw["pm2_5"] = [10, 20, np.nan, 40, 50, 60]
    out = clean_data(raw)
    assert out["pm2_5"].isna().sum() == 0
    assert out.loc[2, "pm2_5"] == pytest.approx(30.0)


@pytest.mark.parametrize("kolom, nilai_buruk", [
    ("pm2_5", -1.0),
    ("pm10", -1.0),
    ("relative_humidity_2m", 101.0),
    ("cloudcover", 101.0),
    ("winddirection_10m", 361.0),
])
def test_baris_di_luar_batas_dibuang(kolom, nilai_buruk):
    raw = make_raw(6)
    raw.loc[3, kolom] = nilai_buruk
    out = clean_data(raw)
    assert len(out) == 5
    assert (out[kolom] == nilai_buruk).sum() == 0


def test_nilai_batas_tepat_dipertahankan():
    """0, 100, dan 360 masih valid dan tidak boleh dibuang."""
    raw = make_raw(4)
    raw.loc[0, "pm2_5"] = 0.0
    raw.loc[1, "relative_humidity_2m"] = 100.0
    raw.loc[2, "winddirection_10m"] = 360.0
    assert len(clean_data(raw)) == 4


def test_hasil_terurut_berdasarkan_waktu():
    raw = make_raw(6).sample(frac=1, random_state=0)
    out = clean_data(raw)
    assert out["time"].is_monotonic_increasing


def test_kolom_teks_dikonversi_ke_numerik():
    raw = make_raw(4)
    raw["pm10"] = raw["pm10"].astype(str)
    out = clean_data(raw)
    assert pd.api.types.is_numeric_dtype(out["pm10"])
    assert len(out) == 4


def test_validate_schema_lolos_jika_lengkap():
    assert validate_schema(make_raw(3)) is None


def test_validate_schema_gagal_jika_kolom_hilang():
    raw = make_raw(3).drop(columns=["pm2_5"])
    with pytest.raises(ValueError, match="pm2_5"):
        validate_schema(raw)
