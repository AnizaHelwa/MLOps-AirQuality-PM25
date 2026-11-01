"""
test_inference.py

Menguji kesiapan inferensi: memuat model berstatus Production dari
MLflow Model Registry (lewat alias) dan memprediksi beberapa baris data.
"""

import mlflow.pyfunc

from train import (load_processed_data, build_features,
                   get_feature_columns)

MODEL_URI = "models:/pm25-forecasting-xgboost@production"


def main():
    """Memuat model Production dan menampilkan contoh prediksi."""
    model = mlflow.pyfunc.load_model(MODEL_URI)
    print("Model berhasil dimuat dari registry:", MODEL_URI)

    data = build_features(load_processed_data())
    x_new = data[get_feature_columns()].tail(5)
    y_actual = data["target_pm2_5_t1"].tail(5)
    pred = model.predict(x_new)

    print("\nPrediksi PM2.5 (t+1) vs aktual, 5 baris terakhir:")
    for t, p, a in zip(data["time"].tail(5), pred, y_actual):
        print(f"  {t} | prediksi={p:.2f} | aktual={a:.2f}")


if __name__ == "__main__":
    main()
