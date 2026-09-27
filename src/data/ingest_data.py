import os
import io
import logging
from datetime import datetime
import pandas as pd
import requests 
from celery import Celery
from minio import Minio

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s')

# Konfigurasi Celery dan Redis
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
app = Celery('ingestion_tasks', broker=CELERY_BROKER_URL)

# Konfigurasi MinIO Client
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'localhost:9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin')  
BUCKET_NAME = 'mlops-airquality-bucket'

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False  # Set to True if using HTTPS
)

# Koordinat DKI Jakarta
LATITUDE = -6.2146
LONGITUDE = 106.8451

@app.task(name='tasks.fetch_and_store_air_quality_data')
def fetch_and_store_air_quality_data():
    logging.info("Memulai proses Ingestion Data dari Open-Meteo API...")
    
    # 1. Fetch Air Quality Data from Open-Meteo API 
    url_aq = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params_aq = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ["pm10", "pm2_5", "ozone", "sulphur_dioxide", "nitrogen_dioxide", "carbon_monoxide"],
        "timezone": "Asia/Jakarta"
    }
    
    res_aq = requests.get(url_aq, params=params_aq, timeout=10)
    res_aq.raise_for_status()
    df_aq = pd.DataFrame(res_aq.json()['hourly'])
    
    # 2. Fetch Weather Forecast Data from Open-Meteo API
    url_weather = "https://api.open-meteo.com/v1/forecast"
    params_weather = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ["temperature_2m", "relative_humidity_2m", "windspeed_10m", "winddirection_10m", "surface_pressure", "precipitation", "cloudcover"],
        "timezone": "Asia/Jakarta"
    }
    res_weather = requests.get(url_weather, params=params_weather, timeout=10)
    res_weather.raise_for_status()
    df_weather = pd.DataFrame(res_weather.json()['hourly'])
    
    # 3. Merge Datasets based on Timestamp
    df_merged = pd.merge(df_aq, df_weather, on="time")
    logging.info(f"Status Ingestion: Berhasil! Jumlah baris data: {len(df_merged)}")
    
    # 4. Save to MinIO Bucket
    parquet_buffer = io.BytesIO()
    df_merged.to_parquet(parquet_buffer, index=False)
    parquet_buffer.seek(0)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    object_name = f"data/raw/raw_air_quality_weather_{timestamp}.parquet"
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
    logging.info(f"Data berhasil disimpan ke MinIO bucket '{BUCKET_NAME}' pada path: {object_name}")
    
    return {"status": "success", "rows": len(df_merged), "path": object_name}

if __name__ == "__main__":
    fetch_and_store_air_quality_data()