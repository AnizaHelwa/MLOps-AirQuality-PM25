# 🌫️ MLOps Air Quality PM2.5 Forecasting

## 📌 Deskripsi Proyek

**MLOps Air Quality PM2.5 Forecasting** merupakan implementasi sistem MLOps end-to-end untuk melakukan **peramalan konsentrasi partikulat PM2.5 jangka pendek secara berkelanjutan** di wilayah DKI Jakarta.

Sistem menggunakan data time-series multivariat yang diperoleh secara dinamis melalui **Open-Meteo Air Quality API** dan **Open-Meteo Weather Forecast API**. Model dirancang untuk melakukan **multi-step forecasting**, yaitu memprediksi konsentrasi PM2.5 untuk **1 hingga 6 jam ke depan (t+1 sampai t+6)**.

Proyek ini tidak hanya berfokus pada pembangunan model machine learning, tetapi juga mencakup proses **data ingestion, data validation, feature engineering, model training, evaluation, deployment, serving, monitoring, hingga automatic retraining**.

---

## 🎯 Tujuan Proyek

Proyek ini bertujuan untuk:

* Membangun sistem prediksi PM2.5 jangka pendek berbasis data dinamis.
* Menghasilkan prediksi PM2.5 untuk horizon **t+1 hingga t+6 jam**.
* Menerapkan prinsip **MLOps end-to-end** dalam pengembangan sistem machine learning.
* Memastikan kualitas dan freshness data yang digunakan dalam pipeline.
* Mendeteksi **data drift** dan penurunan performa model.
* Mengimplementasikan mekanisme **continuous learning dan automatic retraining**.
* Menyediakan hasil prediksi melalui layanan API yang dapat digunakan oleh sistem lain.

---

## 📊 Data Source

Sistem menggunakan dua sumber data utama dari Open-Meteo.

### 1. Air Quality API

Data kualitas udara berasal dari **Open-Meteo Air Quality API** dengan sumber atmospheric composition **CAMS Global**.

Variabel kualitas udara yang digunakan antara lain:

* PM2.5
* PM10
* NO₂
* SO₂
* CO
* O₃

**Target utama:** `PM2.5`

### 2. Weather Forecast API

Data meteorologi berasal dari **Open-Meteo Weather Forecast API**.

Variabel yang digunakan meliputi:

* Temperature
* Relative Humidity
* Wind Speed
* Wind Direction
* Atmospheric Pressure
* Precipitation
* Cloud Cover

### 📍 Lokasi

**DKI Jakarta, Indonesia**

Koordinat yang digunakan:

```text
Latitude  : -6.2146
Longitude : 106.8451
```

Data berbentuk **hourly time-series** dan digabungkan berdasarkan timestamp sebelum digunakan dalam proses preprocessing dan feature engineering.

---

## 🤖 Machine Learning Task

### Task

**Supervised Time-Series Regression / Multi-Step Forecasting**

### Target

Prediksi konsentrasi PM2.5:

```text
t+1 hour
t+2 hours
t+3 hours
t+4 hours
t+5 hours
t+6 hours
```

Model memanfaatkan informasi historis PM2.5 serta variabel kualitas udara dan meteorologi sebagai fitur prediktor.

---

## 🔄 MLOps Pipeline

Sistem dirancang sebagai pipeline MLOps yang terdiri dari beberapa tahapan utama:

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
Docker Build & Test
      ↓
FastAPI Model Serving
      ↓
Monitoring
      ↓
Drift / Performance Detection
      ↓
Retraining Trigger
      └──────────────→ Model Training
```

### 1. 📥 Data Ingestion

Pipeline mengambil data terbaru dari Open-Meteo menggunakan **REST API**.

Mekanisme pengambilan data menggunakan pendekatan **pull-based**, dengan pipeline dijalankan secara berkala setiap **6 jam**.

Data mentah kemudian disimpan untuk digunakan pada tahap preprocessing dan training.

---

### 2. ✅ Data Validation

Data yang diperoleh dari API diperiksa sebelum digunakan dalam proses machine learning.

Validasi mencakup:

* Struktur data
* Tipe data
* Missing values
* Validitas timestamp
* Konsistensi variabel
* Nilai yang berada di luar batas yang tidak wajar

Tools yang direncanakan untuk digunakan:

* **Great Expectations**
* **Pydantic**

---

### 3. ⚙️ Feature Engineering

Data time-series diubah menjadi fitur yang dapat digunakan oleh model regression.

Tahapan meliputi:

* Penyusunan data berdasarkan timestamp.
* Pembuatan fitur lag.
* Penggunaan data historis PM2.5.
* Penggabungan variabel kualitas udara dan cuaca.
* Pembentukan target multi-step forecasting.

Contoh konsep fitur:

```text
PM2.5(t-1)
PM2.5(t-2)
PM2.5(t-3)
...
Weather(t)
AirQuality(t)
```

Target:

```text
PM2.5(t+1) ... PM2.5(t+6)
```

---

### 4. 🧠 Continuous Learning

Tahap training digunakan untuk menghasilkan kandidat model regression.

Model yang dapat digunakan antara lain:

* Baseline Regression
* Scikit-learn Regression Models
* XGBoost

Model kandidat dibandingkan berdasarkan performa pada data validasi menggunakan metrik:

* MAE
* RMSE
* R²

Model dengan performa terbaik kemudian dikelola menggunakan **MLflow Model Registry**.

---

### 5. 📈 Model Evaluation

Performa model dievaluasi menggunakan beberapa metrik.

| Metric |     Target |
| ------ | ---------: |
| RMSE   | ≤ 15 μg/m³ |
| MAE    | ≤ 10 μg/m³ |
| R²     |     ≥ 0.70 |

Evaluasi dilakukan untuk memastikan model memenuhi target performa sebelum digunakan pada environment serving.

---

### 6. 🚀 Continuous Deployment

Model yang telah memenuhi kriteria performa dapat masuk ke tahap deployment.

Proses deployment dirancang menggunakan:

* GitHub Actions / Cron Job
* Docker
* Cloud PaaS

Container digunakan untuk menjaga konsistensi environment antara proses development, testing, dan deployment.

---

### 7. 🌐 Model Serving

Model disediakan melalui REST API menggunakan:

* **FastAPI**
* **Uvicorn**

API memungkinkan sistem lain untuk meminta prediksi PM2.5 hingga **6 jam ke depan**.

Contoh konsep request:

```text
GET /predict
```

Output yang diharapkan berupa prediksi:

```text
t+1 → PM2.5 prediction
t+2 → PM2.5 prediction
t+3 → PM2.5 prediction
t+4 → PM2.5 prediction
t+5 → PM2.5 prediction
t+6 → PM2.5 prediction
```

---

### 8. 👀 Monitoring & Observability

Monitoring dilakukan untuk memastikan sistem tetap berjalan dengan baik setelah deployment.

Aspek yang dipantau meliputi:

#### Data Quality

Memantau kualitas data yang masuk ke pipeline.

#### Data Freshness

Memastikan data terbaru tersedia setelah sumber data melakukan update.

#### Inference Latency

Memantau waktu yang dibutuhkan API untuk menghasilkan prediksi.

#### Model Performance

Memantau perubahan performa model berdasarkan MAE dan RMSE.

#### Data Drift

Memantau perubahan distribusi data input dibandingkan dengan data referensi.

Tools yang direncanakan:

* **Evidently AI**
* Monitoring metrics pada pipeline dan API

---

## 🚨 Drift & Retraining Strategy

Sistem menggunakan pendekatan **Hybrid Trigger** untuk menentukan kapan model perlu dilatih kembali.

Hybrid trigger terdiri dari:

```text
Performance Decay
       +
Data Drift
       +
Scheduled Retraining
```

### 1. Performance Decay

Retraining dilakukan apabila performa model mengalami penurunan secara konsisten.

Trigger:

```text
MAE > 10 μg/m³
atau
RMSE > 15 μg/m³
```

selama **3 siklus berturut-turut**.

Kondisi tersebut akan memicu **automatic emergency retraining**.

---

### 2. Data Drift

Data drift dideteksi menggunakan **Kolmogorov-Smirnov (KS) Test**.

Trigger retraining:

```text
p-value < 0.05
```

pada sekitar **30% fitur utama**.

Hal ini menunjukkan adanya perubahan distribusi fitur yang cukup signifikan dibandingkan dengan data referensi.

---

### 3. Scheduled Retraining

Selain berdasarkan kondisi model dan data, retraining juga dilakukan secara berkala.

Jadwal:

```text
Setiap 14 hari
```

Training menggunakan **sliding window 60 hari terakhir** agar model tetap beradaptasi terhadap kondisi data terbaru.

---

## 📊 Success Metrics

### Model Performance

| Metric |     Target |
| ------ | ---------: |
| RMSE   | ≤ 15 μg/m³ |
| MAE    | ≤ 10 μg/m³ |
| R²     |     ≥ 0.70 |

### Operational Performance

| Metric                |                        Target |
| --------------------- | ----------------------------: |
| P95 Inference Latency |                     ≤ 1 detik |
| Pipeline Execution    |              ≤ 15 menit/cycle |
| Data Freshness        | ≤ 1 jam setelah data tersedia |
| Service Availability  |            ≥ 90% uptime/bulan |

### Business / User Requirement

Sistem diharapkan:

* Menghasilkan forecast **t+1 hingga t+6 jam** secara konsisten.
* Menyediakan informasi prediksi sebelum periode yang diprediksi.
* Mendukung penggunaan hasil forecast sebagai informasi **early warning kualitas udara**.

---

## 🏗️ System Architecture

Arsitektur sistem secara konseptual:

```text
                 ┌──────────────────────┐
                 │     Open-Meteo API   │
                 │  Air Quality +       │
                 │  Weather Forecast    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Data Ingestion     │
                 │     REST / Pull      │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │  Data Validation     │
                 │ Great Expectations   │
                 │     / Pydantic       │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Feature Engineering  │
                 │   Lag Transformation │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Model Training     │
                 │ Scikit-learn/XGBoost │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │       MLflow         │
                 │   Model Registry     │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │    Docker + CI/CD    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ FastAPI + Uvicorn    │
                 │   Model Serving      │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Monitoring & Drift   │
                 │      Evidently       │
                 └──────────┬───────────┘
                            │
                            ▼
                  Retraining Trigger
                            │
                            └──────→ Training
```

---

## 🛠️ Tech Stack

| Category             | Technology                   |
| -------------------- | ---------------------------- |
| Programming Language | Python 3.10+                 |
| Data Ingestion       | Requests, Open-Meteo API     |
| Data Processing      | Pandas, NumPy                |
| Data Validation      | Great Expectations, Pydantic |
| Machine Learning     | Scikit-learn, XGBoost        |
| Experiment Tracking  | MLflow                       |
| CI/CD                | GitHub Actions               |
| Scheduling           | Cron Job                     |
| API Serving          | FastAPI, Uvicorn             |
| Monitoring           | Evidently AI                 |
| Containerization     | Docker                       |
| Deployment           | Cloud PaaS / Render          |
| Data Format          | JSON → Tabular / Parquet     |
| Version Control      | Git & GitHub                 |

---

## 📁 Project Structure

```text
MLOps-Air-Quality-PM25/
│
├── .devcontainer/
│   └── ...
│
├── config/
│   └── ...
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   └── ...
│
├── models/
│   └── ...
│
├── notebooks/
│   └── ...
│
├── src/
│   └── main.py
│
├── tests/
│   └── ...
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

> Struktur direktori dapat dikembangkan menjadi modul terpisah untuk ingestion, validation, feature engineering, training, serving, dan monitoring seiring implementasi pipeline.

---

## 🚀 Installation

Clone repository:

```bash
git clone <repository-url>
cd MLOps-Air-Quality-PM25-Forecasting
```

Buat virtual environment:

```bash
python -m venv venv
```

Aktifkan virtual environment.

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Jalankan pipeline utama:

```bash
python src/main.py
```

Jika model serving telah tersedia menggunakan FastAPI:

```bash
uvicorn src.main:app --reload
```

API kemudian dapat digunakan untuk mengakses endpoint prediksi.

---

## 🔁 Continuous Pipeline

Pipeline dirancang untuk berjalan secara berkala:

```text
Every 6 Hours
      ↓
Fetch New Data
      ↓
Validate Data
      ↓
Update Dataset
      ↓
Generate Features
      ↓
Generate Prediction
      ↓
Monitor Performance & Drift
      ↓
Check Retraining Trigger
      ↓
Retrain if Necessary
```

Dengan pendekatan ini, sistem dapat terus beradaptasi terhadap perubahan karakteristik data PM2.5 dan kondisi meteorologi.

---

## 📌 Current Development Status

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

## 🔮 Future Development

Pengembangan selanjutnya mencakup:

* Implementasi pipeline ingestion otomatis.
* Implementasi data validation secara otomatis.
* Pengembangan fitur lag dan time-series.
* Eksperimen dan pemilihan model terbaik.
* Integrasi MLflow Model Registry.
* Containerization menggunakan Docker.
* Deployment model serving.
* Implementasi monitoring menggunakan Evidently.
* Implementasi data drift detection.
* Implementasi automatic retraining berdasarkan hybrid trigger.
* Pengembangan sistem menjadi layanan forecasting kualitas udara yang dapat digunakan secara berkelanjutan.

---

## 📚 Project Documentation

Dokumentasi lengkap mengenai inisiasi dan rancangan sistem tersedia pada:

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
