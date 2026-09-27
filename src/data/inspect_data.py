import io
import pandas as pd
from minio import Minio

BUCKET = "mlops-airquality-bucket"

c = Minio("localhost:9000", access_key="minioadmin", secret_key="minioadmin", secure=False)

# Ambil file terbaru di data/raw/
files = sorted(o.object_name for o in c.list_objects(BUCKET, prefix="data/raw/", recursive=True))
print("File tersedia:", len(files))
latest = files[-1]
print("Membaca:", latest, "\n")

r = c.get_object(BUCKET, latest)
df = pd.read_parquet(io.BytesIO(r.read()))
r.close()
r.release_conn()

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("Ukuran (baris, kolom):", df.shape)
print("\n=== Kolom & tipe data ===")
print(df.dtypes)
print("\n=== 5 baris pertama ===")
print(df.head())
print("\n=== Statistik ringkas ===")
print(df.describe().T)
print("\n=== Jumlah nilai kosong per kolom ===")
print(df.isna().sum())