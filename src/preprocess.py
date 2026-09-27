import io
import logging
import pandas as pd
from minio import Minio

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s')

MINIO_ENDPOINT = "localhost:9000"
ACCESS_KEY = "minioadmin"
SECRET_KEY = "minioadmin"
BUCKET = "mlops-airquality-bucket"

client = Minio(
    MINIO_ENDPOINT,
    access_key=ACCESS_KEY,
    secret_key=SECRET_KEY,
    secure=False)


def load_all_raw() -> pd.DataFrame:
    """
    Mengambil seluruh file Parquet di folder data/raw/ pada MinIO,
    lalu menggabungkannya menjadi satu DataFrame tunggal.

    Returns:
        DataFrame gabungan dari semua file raw yang ditemukan.

    Raises:
        FileNotFoundError: jika tidak ada file raw yang ditemukan.
    """
    files = sorted(o.object_name for o in client.list_objects(
        BUCKET, prefix="data/raw/", recursive=True))
    if not files:
        raise FileNotFoundError("Tidak ada file raw di MinIO.")

    logging.info(f"Menemukan {len(files)} file raw, menggabungkan semuanya...")
    dfs = []
    for f in files:
        r = client.get_object(BUCKET, f)
        dfs.append(pd.read_parquet(io.BytesIO(r.read())))
        r.close()
        r.release_conn()

    df_all = pd.concat(dfs, ignore_index=True)
    logging.info(f"Total baris sebelum dedup: {len(df_all)}")
    return df_all


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membersihkan data mentah: konversi tipe waktu, hapus duplikat timestamp
    (ambil versi prakiraan terbaru), interpolasi nilai kosong, dan validasi
    batas nilai logis untuk tiap variabel polutan/cuaca.
    """

    df = df.copy()

    # 0. Pastikan kolom numerik memang bertipe numerik
    expected_numeric_cols = [
        "pm10", "pm2_5", "ozone", "sulphur_dioxide", "nitrogen_dioxide",
        "carbon_monoxide", "temperature_2m", "relative_humidity_2m",
        "windspeed_10m", "winddirection_10m", "surface_pressure",
        "precipitation", "cloudcover"
    ]
    for col in expected_numeric_cols:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            logging.warning(
                f"Kolom '{col}' bukan numerik, mencoba konversi paksa...")
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 1. Pastikan time jadi datetime
    df["time"] = pd.to_datetime(df["time"])

    # 2. Buang duplikat timestamp (antar file yang tumpang tindih), simpan
    # yang paling baru
    df = df.sort_values("time")
    before = len(df)
    df = df.drop_duplicates(subset="time", keep="last")
    logging.info(f"Baris duplikat timestamp dibuang: {before - len(df)}")

    # 3. Mengurutkan data berdasarkan waktu dan reset index
    df = df.sort_values("time").reset_index(drop=True)

    # 4. Tangani missing values dengan interpolasi linear
    numeric_cols = df.select_dtypes(include="number").columns
    n_missing_before = df[numeric_cols].isna().sum().sum()
    df[numeric_cols] = df[numeric_cols].interpolate(
        method="linear", limit_direction="both")
    logging.info(f"Total NaN sebelum interpolasi: {n_missing_before}")

    # 5. Validasi batasan data
    before_valid = len(df)
    df = df[df["pm2_5"] >= 0]
    df = df[df["pm10"] >= 0]
    df = df[(df["relative_humidity_2m"] >= 0) &
            (df["relative_humidity_2m"] <= 100)]
    df = df[(df["cloudcover"] >= 0) & (df["cloudcover"] <= 100)]
    df = df[(df["winddirection_10m"] >= 0) & (df["winddirection_10m"] <= 360)]
    logging.info(
        f"Baris dibuang karena melanggar batas logis: "
        f"{before_valid - len(df)}")

    return df.reset_index(drop=True)


REQUIRED_COLUMNS = [
    "time", "pm10", "pm2_5", "ozone", "sulphur_dioxide", "nitrogen_dioxide",
    "carbon_monoxide", "temperature_2m", "relative_humidity_2m",
    "windspeed_10m", "winddirection_10m", "surface_pressure",
    "precipitation", "cloudcover"
]


def validate_schema(df: pd.DataFrame) -> None:
    """
    Memastikan seluruh kolom wajib ada sebelum data
    diproses lebih lanjut.
    """

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Data tidak memiliki kolom wajib: {missing}")


def save_processed(df: pd.DataFrame):
    """
    Menyimpan DataFrame hasil cleaning sebagai file Parquet
    ke MinIO pada path data/processed/data_processed_latest.parquet.

    Args:
        df: DataFrame yang sudah dibersihkan dan siap disimpan.
    """
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    object_name = "data/processed/data_processed_latest.parquet"
    client.put_object(
        bucket_name=BUCKET,
        object_name=object_name,
        data=buf,
        length=buf.getbuffer().nbytes,
        content_type="application/octet-stream"
    )
    logging.info(f"Data processed disimpan di: {object_name}")


def run_preprocessing():
    """
    Menjalankan alur preprocessing lengkap: memuat seluruh data raw,
    memvalidasi skema, membersihkan data, lalu menyimpan hasilnya
    sebagai data processed ke MinIO.

    Returns:
        DataFrame hasil akhir yang sudah dibersihkan.
    """
    df_raw = load_all_raw()
    validate_schema(df_raw)
    logging.info(f"Data raw setelah gabung: {df_raw.shape}")

    df_clean = clean_data(df_raw)
    logging.info(f"Data final setelah cleaning: {df_clean.shape}")
    logging.info(
        f"Rentang waktu data: {df_clean['time'].min()} "
        f"s.d. {df_clean['time'].max()}")
    logging.info(f"Total NaN tersisa: {df_clean.isna().sum().sum()}")

    save_processed(df_clean)
    return df_clean


if __name__ == "__main__":
    run_preprocessing()
