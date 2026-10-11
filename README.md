# 🌫️ MLOps Air Quality PM2.5 Forecasting

## 📌 Deskripsi Proyek

**MLOps Air Quality PM2.5 Forecasting** merupakan proyek implementasi infrastruktur MLOps end-to-end untuk melakukan **peramalan konsentrasi partikulat PM2.5 jangka pendek secara berkelanjutan di wilayah DKI Jakarta**.

Proyek ini menggunakan data time-series dinamis dari **Open-Meteo Air Quality API** dan **Open-Meteo Weather Forecast API**. Sistem dirancang untuk melakukan **multi-step forecasting**, yaitu memprediksi konsentrasi PM2.5 untuk **1 hingga 6 jam ke depan (t+1 hingga t+6)**.

Selain pembangunan model machine learning, proyek mencakup tahapan **data ingestion, data validation, feature engineering, model training, model evaluation, deployment, model serving, monitoring, dan retraining**.

---

## 🎯 Tujuan Proyek

Tujuan utama proyek ini adalah:

* Membangun sistem prediksi PM2.5 jangka pendek berbasis data dinamis.
* Melakukan prediksi PM2.5 untuk **t+1 hingga t+6 jam**.
* Menerapkan konsep **MLOps end-to-end** pada sistem machine learning.
* Menjaga kualitas dan freshness data yang digunakan dalam pipeline.
* Mendeteksi **data drift** dan penurunan performa model.
* Menerapkan mekanisme **continuous learning dan automatic retraining**.
* Menyediakan hasil prediksi melalui layanan API.

---

## 📊 Data Source

Proyek menggunakan dua sumber data dari Open-Meteo.

### Air Quality API

Data kualitas udara diperoleh dari **Open-Meteo Air Quality API** dengan sumber **CAMS Global**.

Variabel yang digunakan meliputi:

* PM2.5
* PM10
* NO₂
* SO₂
* CO
* O₃

**Target prediksi:** PM2.5

### Weather Forecast API

Data cuaca diperoleh dari **Open-Meteo Weather Forecast API**, meliputi:

* Temperature
* Relative Humidity
* Wind Speed
* Wind Direction
* Atmospheric Pressure
* Precipitation
* Cloud Cover

### 📍 Lokasi

**DKI Jakarta, Indonesia**

```text
Latitude  : -6.2146
Longitude : 106.8451
```

Data yang digunakan berupa **hourly time-series** dan digabungkan berdasarkan timestamp.

---

# 📁 Struktur Direktori

Struktur direktori utama proyek adalah sebagai berikut:

```text
MLOps-Air-Quality-PM25-Forecasting/
│
├── .devcontainer/     # Konfigurasi GitHub Codespaces
│
├── config/            # Konfigurasi pipeline dan parameter
│
├── data/              # Penyimpanan data
│   ├── raw/           # Data mentah hasil fetching API
│   └── processed/     # Data hasil preprocessing dan feature engineering
│
├── docs/              # Dokumentasi proyek dan laporan LK
│
├── models/            # Artefak model terlatih
│
├── notebooks/         # Notebook untuk EDA dan eksperimen
│
├── src/               # Source code utama
│   └── main.py        # Entry point pipeline
│
├── tests/             # Unit testing
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

> Struktur direktori dapat dikembangkan menjadi beberapa modul seperti ingestion, validation, preprocessing, training, serving, dan monitoring sesuai perkembangan implementasi proyek.

---

# 💻 Menjalankan Proyek dengan GitHub Codespaces

Proyek ini menggunakan konfigurasi **`.devcontainer/`** untuk membantu menyediakan environment pengembangan yang konsisten melalui **GitHub Codespaces**.

## 1. Membuka Repository

Buka repository proyek pada GitHub.

Kemudian klik:

```text
Code → Codespaces → Create codespace on main
```

Tunggu hingga GitHub Codespaces selesai membuat development environment.

---

## 2. Membuka Terminal

Setelah Codespaces terbuka, buka terminal melalui:

```text
Terminal → New Terminal
```

Kemudian periksa versi Python:

```bash
python --version
```

Pastikan Python yang digunakan memenuhi versi yang dibutuhkan proyek, yaitu **Python 3.10 atau lebih baru**.

---

## 3. Install Dependencies

Install seluruh library yang dibutuhkan menggunakan:

```bash
pip install -r requirements.txt
```

Dependencies utama yang digunakan dalam proyek antara lain:

* Pandas
* NumPy
* Requests
* Scikit-learn
* XGBoost
* MLflow
* FastAPI
* Uvicorn
* Great Expectations / Pydantic
* Evidently AI

---

## 4. Menjalankan Pipeline

Setelah dependencies selesai di-install, jalankan entry point proyek:

```bash
python src/main.py
```

Pipeline akan menjalankan proses sesuai implementasi yang tersedia pada source code.

---

## 5. Menjalankan Model Serving

Apabila FastAPI telah diimplementasikan, model dapat dijalankan menggunakan:

```bash
uvicorn src.main:app --reload
```

Server kemudian dapat diakses melalui environment Codespaces.

---

# 🤖 Machine Learning Task

Proyek ini menggunakan pendekatan:

**Supervised Time-Series Regression / Multi-Step Forecasting**

Target yang diprediksi:

```text
PM2.5(t+1)
PM2.5(t+2)
PM2.5(t+3)
PM2.5(t+4)
PM2.5(t+5)
PM2.5(t+6)
```

Fitur prediksi berasal dari data historis PM2.5 serta variabel kualitas udara dan meteorologi.

---

# 🔄 MLOps Pipeline

Pipeline MLOps dirancang dengan alur:

```text
Open-Meteo API
      ↓
Data Ingestion
      ↓
Data Validation
      ↓
Data Storage
      ↓
Feature Engineering
      ↓
Model Training
      ↓
Model Evaluation
      ↓
MLflow Model Registry
      ↓
Deployment
      ↓
FastAPI Model Serving
      ↓
Monitoring
      ↓
Drift & Performance Detection
      ↓
Retraining
      └──────────────→ Model Training
```

## 1. Data Ingestion

Data terbaru diperoleh melalui REST API dari Open-Meteo dengan mekanisme **pull-based**.

Pipeline dirancang berjalan setiap **6 jam** untuk memperoleh data terbaru.

### Implementasi & Cara Menjalankan

Ingestion dan preprocessing awal sudah diimplementasikan dan dapat dijalankan
secara lokal di GitHub Codespaces:

**1. Jalankan service pendukung (Redis & MinIO)**
```bash
docker compose up -d
docker compose ps
```
Pastikan `mlops_redis` dan `mlops_minio` berstatus `running`. MinIO Console
dapat diakses lewat tab **PORTS** di VS Code, pada port `9001`
(login: `minioadmin` / `minioadmin`).

**2. Jalankan ingestion**
```bash
cd src
python ingest_data.py
```
Skrip ini mengambil data dari Air Quality API dan Weather Forecast API,
menggabungkannya berdasarkan `time`, lalu menyimpan hasilnya sebagai file
Parquet baru (bertimestamp, tidak menimpa file lama) ke MinIO bucket
`mlops-airquality-bucket`, path `data/raw/`.

**3. Jalankan preprocessing**
```bash
python preprocess.py
```
Skrip ini menggabungkan **seluruh** file di `data/raw/`, membuang duplikat
timestamp (mengambil versi prakiraan paling baru), menangani missing value
dengan interpolasi linear, memvalidasi batas nilai logis (mis. `pm2_5 ≥ 0`,
kelembapan 0–100%), lalu menyimpan hasil bersih ke
`data/processed/data_processed_latest.parquet` di MinIO.

> **Catatan:** Data disimpan di MinIO (S3-compatible object storage),
> bukan langsung di filesystem/Git, agar repositori tetap ringan sesuai
> strategi *Object Storage* pada `docs/`.

### Bukti Otomasi Berkala

Pengujian otomasi dilakukan dengan menjalankan Celery Worker dan Celery
Beat secara bersamaan (interval dipercepat menjadi 1 menit untuk keperluan
demo). Hasilnya, task ingestion berhasil dijalankan otomatis oleh
scheduler tanpa intervensi manual, termasuk berhasil pulih dari kegagalan
koneksi sementara (`Read timed out`) berkat mekanisme retry:

[04:54:00] Scheduler: Sending due task fetch-air-quality-every-6-hours

[04:54:10] WARNING - Percobaan 1/3 gagal (koneksi): Read timed out

[04:54:16] INFO - Status Ingestion: Berhasil! Jumlah baris data: 120

[04:54:16] Task ... succeeded: {'status': 'success', 'rows': 120, ...}

Untuk produksi, jadwal diatur setiap 6 jam melalui `crontab(minute=0, hour='0,6,12,18')` di `src/celery_config.py`.

### Catatan Teknis: Pemilihan Celery Beat vs GitHub Actions

Pada rancangan LK-03, pemicu ingestion direncanakan menggunakan Cron Job
atau GitHub Actions Scheduled Workflow. Pada implementasi LK-04 ini,
scheduling diimplementasikan menggunakan **Celery Beat** sebagai
alternatif, dengan pertimbangan:

- GitHub Actions berjalan di runner terpisah (cloud), sehingga tidak
  dapat mengakses `localhost` MinIO/Redis yang berjalan di lingkungan
  GitHub Codespaces.
- Celery Beat memungkinkan pengujian mekanisme *asynchronous task queue*
  yaitu Redis sebagai broker, Celery Worker sebagai consumer yang secara utuh
  dalam satu lingkungan pengembangan, sesuai arsitektur yang dirancang
  di LK-03.

Untuk tahap produksi mendatang, GitHub Actions Scheduled Workflow tetap
menjadi opsi yang relevan apabila sistem sudah di-deploy ke server
permanen dan bukan Codespace dengan MinIO/storage yang dapat diakses
secara publik.

## 2. Data Validation

Data yang diperoleh akan diperiksa sebelum digunakan untuk proses machine learning.

Validasi meliputi:

* Struktur data
* Tipe data
* Missing values
* Timestamp
* Konsistensi data
* Nilai yang tidak valid

Tools yang digunakan atau direncanakan:

* Great Expectations
* Pydantic

## 3. Feature Engineering

Data time-series diolah menjadi fitur yang dapat digunakan oleh model, termasuk:

* Lag features
* Data historis PM2.5
* Variabel kualitas udara
* Variabel meteorologi
* Target multi-step forecasting

## 4. Model Training

Model regression digunakan untuk menghasilkan prediksi PM2.5.

Model kandidat dapat mencakup:

* Baseline Regression
* Scikit-learn Regression Models
* XGBoost

Model dibandingkan berdasarkan performa MAE, RMSE, dan R².

## 5. Model Registry

Model terbaik dikelola menggunakan **MLflow Model Registry** untuk mendukung pengelolaan lifecycle model.

## 6. Deployment & Serving

Model dikemas menggunakan **Docker** dan disediakan melalui REST API menggunakan:

* FastAPI
* Uvicorn

## 7. Monitoring

Monitoring dilakukan terhadap:

* Data quality
* Data freshness
* Inference latency
* Model performance
* Data drift

Evidently AI digunakan sebagai salah satu tools monitoring dan drift detection.

---

# 🗂️ Data Versioning dengan DVC

## Mengapa Data Versioning Krusial untuk Reproduktifitas

Model machine learning tidak hanya ditentukan oleh kode dan hyperparameter,
tetapi juga oleh **data yang digunakan untuk melatihnya**. Tanpa pelacakan
versi data yang jelas, performa model yang direproduksi di waktu berbeda
bisa berbeda meski kode dan konfigurasinya identik, karena data di balik
layar sudah berubah.

Dalam proyek ini, data ditarik secara berkala dari Open-Meteo API dan
bersifat *forecast* (prakiraan), yang nilainya **bisa direvisi dari waktu
ke waktu** meski nama file dan jumlah barisnya sama. Tanpa versioning,
tidak ada cara membedakan "data yang dipakai saat model A dilatih" dengan
"data terbaru yang sudah berubah nilainya" — padahal keduanya bisa memiliki
nama file yang identik.

Alasan konkret mengapa ini penting untuk proyek PM2.5 forecasting:

1. **Reproduksibilitas eksperimen**: Setiap model yang dilatih perlu bisa
   ditelusuri ke versi data persis yang digunakan. Jika suatu model
   menunjukkan performa baik di evaluasi tapi gagal direproduksi, versi
   data adalah kandidat penyebab pertama yang harus bisa dicek.
2. **Audit trail untuk continual learning**: Karena sistem dirancang untuk
   retraining otomatis berdasarkan data baru (lihat strategi Hybrid
   Retraining Trigger), setiap siklus retraining perlu tercatat memakai
   versi data yang mana, agar penurunan atau peningkatan performa model
   bisa dikaitkan dengan perubahan data yang spesifik.
3. **Efisiensi repositori**: Git dirancang untuk melacak perubahan teks,
   bukan file biner/besar. Dengan DVC, file fisik dataset tidak pernah
   masuk ke riwayat Git secara langsung — yang tersimpan di Git hanyalah
   file pointer kecil (`.dvc`) berisi hash. Ini menjaga ukuran repositori
   tetap ringan meski dataset terus bertambah dan berubah dari waktu ke
   waktu.
4. **Rollback yang aman**: Jika versi data baru ternyata mengandung
   anomali (misalnya akibat gangguan API), tim dapat kembali ke versi
   data sebelumnya yang sudah terbukti valid, tanpa kehilangan riwayat
   data yang baru.

## Arsitektur Versioning
Git (riwayat kode & pointer) DVC Cache / Remote (riwayat data fisik)
────────────────────────── 
───────────────────────────────────────
*.dvc (file pointer, berisi → .dvc/cache/ (lokal)
hash MD5 + ukuran file) → MinIO bucket: dvc-storage/ (remote)

Setiap kali `dvc add` dijalankan pada sebuah dataset, DVC:
1. Menghitung hash MD5 dari file tersebut.
2. Menyalin file fisik ke cache DVC lokal (`.dvc/cache/`).
3. Membuat/memperbarui file pointer `<namafile>.dvc` yang berisi hash
   dan metadata, lalu menambahkan nama file aslinya ke `.gitignore`
   (supaya file besar tidak ikut ter-track langsung oleh Git).

File pointer `.dvc` inilah yang di-commit ke Git — ukurannya kecil (hanya
beberapa baris YAML), sehingga riwayat Git tetap ringan walau dataset
fisiknya besar atau sering berubah.

## Remote Storage: MinIO

Cache DVC dikonfigurasi untuk melakukan sinkronisasi ke **MinIO**
(self-hosted, S3-compatible object storage yang sama dengan yang dipakai
pipeline ingestion), pada bucket `mlops-airquality-bucket`, folder
`dvc-storage/`. Dengan begini, data yang di-track DVC tidak hanya
tersimpan lokal di Codespace (yang bersifat sementara), tetapi juga
tersimpan di storage yang persisten.

Kredensial akses MinIO (access key dan secret key) disimpan secara
terpisah di `.dvc/config.local`, yang **tidak ikut ter-commit ke Git**
(otomatis di-gitignore oleh DVC), sehingga kredensial tidak terekspos di
repositori publik. Konfigurasi utama (`url`, `endpointurl`) tetap berada
di `.dvc/config` yang ter-commit, karena informasi ini tidak bersifat
rahasia.

## Alur Kerja Versioning Data

### 1. Inisialisasi (sekali di awal project)

```bash
dvc init
git add .dvc .dvcignore
git commit -m "Initialize DVC"
```

### 2. Melacak dataset untuk pertama kali

```bash
dvc add data/raw/<nama_file>.csv
git add data/raw/<nama_file>.csv.dvc data/raw/.gitignore
git commit -m "Track dataset v1 with DVC"
```

### 3. Simulasi penambahan data baru (continual learning)

Setiap kali `ingest_data.py` dijalankan ulang untuk mengambil data
terbaru, dan data tersebut digunakan untuk memperbarui file yang sudah
dilacak:

```bash
cd src
python ingest_data.py
cd ..
dvc add data/raw/<nama_file>.csv
git add data/raw/<nama_file>.csv.dvc
git commit -m "Update dataset to v2: refreshed data"
```

### 4. Audit dan membandingkan dua versi data

```bash
git diff <commit_lama> -- data/raw/<nama_file>.csv.dvc
dvc diff <commit_lama>
dvc status
```

### 5. Sinkronisasi ke remote storage (MinIO)

```bash
dvc push
dvc pull
```

## Contoh Hasil Nyata (Bukti dari Pengujian)

Pada pengujian, dataset `raw_air_quality_weather_2026-09-27_12-57-17.csv`
(840 baris) ditrack sebagai versi awal, lalu diperbarui dengan data hasil
ingestion baru (tetap 840 baris, namun nilai forecast berbeda karena
direvisi oleh sumber data). Hasil audit menunjukkan:

| | Versi v1 | Versi v2 |
|---|---|---|
| MD5 hash | `9cdbf4915e43d287eabfbb6a1be38fb8` | `26c0cf45adae14bc4c76e2bdf2652438` |
| Ukuran file | 66.737 bytes | 67.096 bytes |
| Jumlah baris | 840 | 840 |

Perintah `dvc diff` antara kedua commit tersebut mengonfirmasi:
Modified:
data/raw/raw_air_quality_weather_2026-09-27_12-57-17.csv

files summary: 1 modified

Ini membuktikan bahwa meskipun nama file dan jumlah baris identik, DVC
berhasil mendeteksi perubahan isi data melalui hash MD5 — sesuatu yang
tidak mungkin dideteksi hanya dengan melihat nama file atau ukuran baris
data saja.# 🚨 Drift & Retraining Strategy

Sistem menggunakan **Hybrid Trigger** yang terdiri dari:

```text
Performance Decay
        +
Data Drift
        +
Scheduled Retraining
```

### Performance Decay

Retraining dipicu apabila:

```text
MAE > 10 μg/m³
atau
RMSE > 15 μg/m³
```

selama **3 siklus berturut-turut**.

### Data Drift

Data drift dideteksi menggunakan **Kolmogorov-Smirnov (KS) Test**.

Trigger:

```text
p-value < 0.05
```

pada sekitar **30% fitur utama**.

### Scheduled Retraining

Retraining terjadwal dilakukan:

```text
Setiap 14 hari
```

dengan menggunakan **sliding window 60 hari terakhir**.

---

# 📈 Success Metrics

## Model Performance

| Metric |     Target |
| ------ | ---------: |
| RMSE   | ≤ 15 μg/m³ |
| MAE    | ≤ 10 μg/m³ |
| R²     |     ≥ 0.70 |

## Operational Performance

| Metric                |                        Target |
| --------------------- | ----------------------------: |
| P95 Inference Latency |                     ≤ 1 detik |
| Pipeline Execution    |              ≤ 15 menit/cycle |
| Data Freshness        | ≤ 1 jam setelah data tersedia |
| Service Availability  |            ≥ 90% uptime/bulan |

Target tersebut digunakan sebagai indikator keberhasilan sistem secara teknis dan operasional.

---

# 🛠️ Tech Stack

| Category             | Technology                   |
| -------------------- | ---------------------------- |
| Programming Language | Python 3.10+                 |
| Data Source          | Open-Meteo API               |
| Data Processing      | Pandas, NumPy                |
| Data Validation      | Great Expectations, Pydantic |
| Machine Learning     | Scikit-learn, XGBoost        |
| Experiment Tracking  | MLflow                       |
| CI/CD                | GitHub Actions               |
| Scheduling           | Cron Job                     |
| Model Serving        | FastAPI, Uvicorn             |
| Monitoring           | Evidently AI                 |
| Containerization     | Docker                       |
| Deployment           | Cloud PaaS / Render          |
| Version Control      | Git & GitHub                 |

---

# 🔁 Continuous Pipeline

Secara konseptual, pipeline berjalan secara berkala:

```text
Every 6 Hours
     ↓
Fetch New Data
     ↓
Validate Data
     ↓
Update Dataset
     ↓
Feature Engineering
     ↓
Generate Prediction
     ↓
Monitor Performance & Drift
     ↓
Check Retraining Trigger
     ↓
Retrain if Necessary
```

Frekuensi pengambilan data disesuaikan dengan pembaruan sumber data yang digunakan. CAMS Global diperbarui setiap 12 jam, sedangkan data weather forecast ECMWF tersedia dengan pembaruan yang lebih sering.

---

# 📌 Development Status

| Component                  | Status         |
| -------------------------- | -------------- |
| Project Definition         | ✅              |
| Data Source Identification | ✅              |
| MLOps Architecture         | ✅              |
| API Data Fetching PoC      | ✅              |
| Data Validation            | 🔄 Development |
| Feature Engineering        | 🔄 Development |
| Model Training             | 🔄 Development |
| MLflow Integration         | 🔄 Development |
| FastAPI Serving            | 🔄 Development |
| Dockerization              | 🔄 Development |
| Monitoring                 | 🔄 Development |
| Drift Detection            | 🔄 Development |
| Automatic Retraining       | 🔄 Development |

---

# 🔮 Future Development

Pengembangan selanjutnya meliputi:

* Otomatisasi data ingestion.
* Implementasi data validation.
* Pengembangan feature engineering untuk time-series.
* Eksperimen model machine learning.
* Integrasi MLflow Model Registry.
* Containerization menggunakan Docker.
* Deployment model serving.
* Implementasi monitoring.
* Implementasi data drift detection.
* Automatic retraining berdasarkan hybrid trigger.

---

# 📚 Documentation

Dokumentasi lengkap mengenai rancangan proyek tersedia pada folder:

```text
docs/
```

Dokumentasi mencakup:

* Problem definition
* Data source
* Machine learning task
* MLOps architecture
* Retraining strategy
* Success metrics
* Technology stack
* Proof of concept

---

## 👩‍💻 Project

**MLOps Air Quality PM2.5 Forecasting**

> Building a sustainable MLOps pipeline for short-term PM2.5 forecasting using dynamic air quality and weather data.
