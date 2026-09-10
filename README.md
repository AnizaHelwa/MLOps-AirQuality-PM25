# MLOps Air Quality PM2.5 Forecasting

## 📌 Deskripsi Proyek
Proyek ini merupakan implementasi infrastruktur MLOps end-to-end untuk peramalan konsentrasi partikulat polusi udara **PM2.5** jangka pendek (multi-step forecasting $t+1$ hingga $t+6$ jam) di wilayah DKI Jakarta. Proyek menggunakan data dinamis *time-series* dari Open-Meteo Air Quality API dan Weather Forecast API.

---

## 📁 Struktur Direktori
Struktur proyek disusun mengacu pada standar industri (*Cookiecutter Data Science*):

```text
├── .devcontainer/     # Konfigurasi GitHub Codespaces
├── config/            # File konfigurasi pipeline & parameter
├── data/              # Storage data (raw & processed)
│   ├── raw/           # Data mentah hasil fetching API
│   └── processed/     # Data hasil cleaning & feature engineering
├── docs/              # Dokumentasi proyek & laporan LK
├── models/            # Artefak model terlatih (.pkl / .json)
├── notebooks/         # Jupyter Notebooks untuk EDA & eksperimentasi
├── src/               # Source code utama pipeline MLOps
│   └── main.py        # Entrypoint script
├── tests/             # Unit testing pipeline
├── .gitignore         # Abai file Python & data temporary
├── LICENSE            # Lisensi proyek (MIT)
└── README.md          # Dokumentasi utama proyek