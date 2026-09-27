"""
ingest_data.py

Skrip ingestion untuk mengambil data kualitas udara dan prakiraan cuaca
dari Open-Meteo API (wilayah DKI Jakarta), menggabungkannya berdasarkan
timestamp, dan menyimpan hasilnya sebagai file Parquet ke MinIO.

Dirancang sebagai Celery task agar dapat dipanggil secara asynchronous
dan dijadwalkan berkala (lihat bagian bawah file untuk contoh scheduling).
"""

import os
import io
import time
import logging
from datetime import datetime

import pandas as pd
import requests
from celery import Celery
from minio import Minio

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s'
)

# --- Konfigurasi Celery & Redis ---
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
app = Celery('ingestion_tasks', broker=CELERY_BROKER_URL)

# --- Konfigurasi MinIO ---
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'localhost:9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin')
BUCKET_NAME = 'mlops-airquality-bucket'

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

# --- Koordinat DKI Jakarta ---
LATITUDE = -6.2146
LONGITUDE = 106.8451

# --- Konfigurasi retry ---
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5


def fetch_with_retry(url: str, params: dict, max_retries: int = MAX_RETRIES,
                     delay: int = RETRY_DELAY_SECONDS) -> dict:
    """
    Mengambil data dari sebuah endpoint REST API dengan mekanisme retry.

    Menangani kegagalan koneksi (ConnectionError, Timeout) dan status HTTP
    error dengan mencoba ulang sebanyak `max_retries` kali, memberi jeda
    `delay` detik antar percobaan. Jika seluruh percobaan gagal, exception
    terakhir akan dilempar kembali ke pemanggil.

    Args:
        url: URL endpoint API.
        params: Query parameters untuk request.
        max_retries: Jumlah maksimum percobaan.
        delay: Jeda antar percobaan (detik).

    Returns:
        Response body yang sudah di-parse sebagai dict (JSON).

    Raises:
        requests.RequestException: jika semua percobaan gagal.
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()

        except (requests.ConnectionError, requests.Timeout) as e:
            last_error = e
            logging.warning(
                f"Percobaan {attempt}/{max_retries} ke {url} gagal (koneksi): {e}"
            )

        except requests.HTTPError as e:
            last_error = e
            logging.warning(
                f"Percobaan {attempt}/{max_retries} ke {url} gagal "
                f"(HTTP {e.response.status_code}): {e}"
            )

        if attempt < max_retries:
            time.sleep(delay)

    logging.error(f"Semua {max_retries} percobaan ke {url} gagal. Melewati proses ingestion.")
    raise last_error


@app.task(name='tasks.fetch_and_store_air_quality_data')
def fetch_and_store_air_quality_data():
    """
    Task utama ingestion: menarik data kualitas udara dan cuaca dari
    Open-Meteo, menggabungkannya, lalu menyimpannya sebagai file Parquet
    ke MinIO pada path data/raw/.

    Jika salah satu sumber data gagal diambil setelah retry, task akan
    berhenti dengan exception dan tidak menyimpan data parsial.

    Returns:
        dict: ringkasan hasil eksekusi (status, jumlah baris, path file).
    """
    logging.info("Memulai proses Ingestion Data dari Open-Meteo API...")

    # 1. Fetch Air Quality Data
    url_aq = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params_aq = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
                   "sulphur_dioxide", "ozone"],
        "timezone": "Asia/Jakarta"
    }

    try:
        data_aq = fetch_with_retry(url_aq, params_aq)
    except requests.RequestException as e:
        logging.error(f"Gagal mengambil data kualitas udara: {e}")
        return {"status": "failed", "stage": "fetch_air_quality", "error": str(e)}

    df_aq = pd.DataFrame(data_aq['hourly'])

    # 2. Fetch Weather Forecast Data
    url_weather = "https://api.open-meteo.com/v1/forecast"
    params_weather = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ["temperature_2m", "relative_humidity_2m", "windspeed_10m",
                   "winddirection_10m", "surface_pressure", "precipitation",
                   "cloudcover"],
        "timezone": "Asia/Jakarta"
    }

    try:
        data_weather = fetch_with_retry(url_weather, params_weather)
    except requests.RequestException as e:
        logging.error(f"Gagal mengambil data cuaca: {e}")
        return {"status": "failed", "stage": "fetch_weather", "error": str(e)}

    df_weather = pd.DataFrame(data_weather['hourly'])

    # 3. Merge berdasarkan timestamp
    df_merged = pd.merge(df_aq, df_weather, on="time")
    logging.info(f"Status Ingestion: Berhasil! Jumlah baris data: {len(df_merged)}")

    # 4. Simpan ke MinIO
    parquet_buffer = io.BytesIO()
    df_merged.to_parquet(parquet_buffer, index=False)
    parquet_buffer.seek(0)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    object_name = f"data/raw/raw_air_quality_weather_{timestamp}.parquet"

    try:
        if not minio_client.bucket_exists(BUCKET_NAME):
            minio_client.make_bucket(BUCKET_NAME)
            logging.info(f"Bucket '{BUCKET_NAME}' berhasil dibuat di MinIO.")

        minio_client.put_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
            data=parquet_buffer,
            length=parquet_buffer.getbuffer().nbytes,
            content_type="application/octet-stream"
        )
        logging.info(
            f"Data berhasil disimpan ke MinIO bucket '{BUCKET_NAME}' pada path: {object_name}"
        )
    except Exception as e:
        logging.error(f"Gagal menyimpan data ke MinIO: {e}")
        return {"status": "failed", "stage": "save_to_minio", "error": str(e)}

    return {"status": "success", "rows": len(df_merged), "path": object_name}


if __name__ == "__main__":
    fetch_and_store_air_quality_data()
