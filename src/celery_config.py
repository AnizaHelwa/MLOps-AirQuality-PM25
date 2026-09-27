"""
celery_config.py

Konfigurasi jadwal (Celery Beat) untuk menjalankan task-task pipeline
data secara otomatis dan berkala, terpisah dari logika task itu sendiri
agar mudah dikembangkan saat jumlah task bertambah (ingest, validate,
feature engineering, retraining, dst).
"""

from celery.schedules import crontab
from ingest_data import app

app.conf.beat_schedule = {
    'fetch-air-quality-every-6-hours': {
        'task': 'tasks.fetch_and_store_air_quality_data',
        'schedule': crontab(minute=0, hour='0,6,12,18'),
    },
}

app.conf.timezone = 'Asia/Jakarta'
