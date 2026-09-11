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

# 🚨 Drift & Retraining Strategy

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
