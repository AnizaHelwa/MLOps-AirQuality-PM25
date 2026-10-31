"""
train.py

File ini digunakan untuk melatih model XGBoost guna memprediksi
konsentrasi PM2.5 satu jam ke depan (t+1), menggunakan data
historis yang telah diproses pada LK-04 dan LK-05. Setiap
eksperimen dicatat ke MLflow Tracking, mencakup parameter,
metrik evaluasi, dan artefak model.
"""

import io
import argparse
import logging

import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
from minio import Minio
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s'
)

MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
BUCKET_NAME = "mlops-airquality-bucket"
PROCESSED_PATH = "data/processed/data_processed_latest.parquet"

LAG_HOURS = [1, 3, 6, 24]
WEATHER_FEATURES = [
    "pm10", "ozone", "sulphur_dioxide", "nitrogen_dioxide",
    "carbon_monoxide", "temperature_2m", "relative_humidity_2m",
    "windspeed_10m", "winddirection_10m", "surface_pressure",
    "precipitation", "cloudcover"
]


def load_processed_data() -> pd.DataFrame:
    """
    Memuat data processed terbaru dari MinIO.

    Returns:
        DataFrame hasil preprocessing (LK-04/LK-05)
    """
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )

    try:
        response = client.get_object(BUCKET_NAME, PROCESSED_PATH)
        data = pd.read_parquet(io.BytesIO(response.read()))
        logging.info("Data processed berhasil dimuat dari MinIO.")
        return data
    except Exception as e:
        logging.error(f"Gagal memuat data processed dari MinIO: {e}")
        raise

    r = client.get_object(BUCKET_NAME, PROCESSED_PATH)
    df = pd.read_parquet(io.BytesIO(r.read()))
    r.close()
    r.release_conn()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membentuk fitur lag, fitur siklis waktu, dan target t+1 dari
    DataFrame hasil preprocessing.

    Args:
        df: DataFrame dengan kolom time, pm2_5, dan variabel cuaca.

    Returns:
        DataFrame baru berisi fitur dan target, dengan baris yang
        mengandung NaN (akibat lag/shift) sudah dibuang.
    """
    data = df.copy()

    for lag in LAG_HOURS:
        data[f"pm2_5_lag_{lag}"] = data["pm2_5"].shift(lag)

    data["hour"] = data["time"].dt.hour
    data["dow"] = data["time"].dt.dayofweek
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
    data["dow_sin"] = np.sin(2 * np.pi * data["dow"] / 7)
    data["dow_cos"] = np.cos(2 * np.pi * data["dow"] / 7)

    data["target_pm2_5_t1"] = data["pm2_5"].shift(-1)

    data = data.dropna().reset_index(drop=True)
    return data


def split_time_series(data: pd.DataFrame, train_ratio: float = -0.85):
    """
    Membagi data menjadi train dan test berdasarkan urutan waktu
    (bukan random split), untuk mencegah kebocoran informasi masa
    depan ke proses training.

    Args:
        data: DataFrame yang sudah berisi fitur dan target.
        train_ratio: Proporsi data untuk training (default 0.85).

    Returns:
        Tuple (train_df, test_df).
    """

    split_idx = int(len(data) * train_ratio)
    train_df = data.iloc[:split_idx]
    test_df = data.iloc[split_idx:]
    return train_df, test_df


def get_feature_columns():
    """Mengembalikan daftar nama kolom fitur yang dipakai model."""
    lag_cols = [f"pm2_5_lag_{lag}" for lag in LAG_HOURS]
    time_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    return lag_cols + time_cols + WEATHER_FEATURES


def train_and_log(n_estimators, max_depth, learning_rate,
                  run_name=None, register_name=None):
    """
    Melatih model XGBoost dengan parameter yang diberikan, mengevaluasi
    pada test set, dan mencatat seluruh hasilnya ke MLflow.

    Args:
        n_estimators: Jumlah pohon (trees) pada XGBoost.
        max_depth: Kedalaman maksimum tiap pohon.
        learning_rate: Laju pembelajaran.
        run_name: Nama opsional untuk run MLflow.
        register_name: Nama model di Model Registry. Jika diisi,
            model didaftarkan sebagai versi baru.
    """
    logging.info("Memuat data processed...")
    df = load_processed_data()

    logging.info("Membangun fitur dan target...")
    data = build_features(df)
    logging.info(f"Data setelah feature engineering: {data.shape}")

    feature_cols = get_feature_columns()
    target_col = "target_pm2_5_t1"

    train_df, test_df = split_time_series(data, train_ratio=0.85)
    logging.info(
        f"Train: {len(train_df)} baris, Test: {len(test_df)} baris"
    )

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("train_ratio", 0.85)
        mlflow.log_param("lag_hours", str(LAG_HOURS))
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("test_rows", len(test_df))

        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        mlflow.xgboost.log_model(
            model, "model", registered_model_name=register_name
        )

        logging.info(
            f"MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f}"
        )

    return mae, rmse, r2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--register_name", type=str, default=None)
    args = parser.parse_args()

    mlflow.set_experiment("pm25-forecasting-t1")

    train_and_log(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        run_name=args.run_name,
        register_name=args.register_name
    )
