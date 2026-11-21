"""
bootstrap.py

Menyiapkan registry pada server MLflow yang masih kosong: melatih
model dari snapshot data, memvalidasinya terhadap baseline naif, lalu
memberi alias production. Jika alias sudah ada, tidak melakukan apa
pun sehingga aman dijalankan berulang.
"""

import subprocess
import sys

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

MODEL_NAME = "pm25-forecasting-xgboost"
ALIAS = "production"
TRAIN_ARGS = [
    "--n_estimators", "150",
    "--max_depth", "5",
    "--learning_rate", "0.08",
    "--run_name", "compose_trainer",
    "--register_name", MODEL_NAME,
]


def alias_exists(client):
    """True jika alias production sudah menunjuk ke sebuah versi."""
    try:
        client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        return True
    except MlflowException as exc:
        if getattr(exc, "error_code", "") == "RESOURCE_DOES_NOT_EXIST":
            return False
        raise


def run(cmd):
    """Menjalankan perintah; berhenti dengan kode yang sama jika gagal."""
    print("$", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Langkah gagal (kode {exc.returncode}), berhenti.")
        sys.exit(exc.returncode)


def main():
    """Melatih, memvalidasi, dan memberi alias jika registry kosong."""
    client = MlflowClient()
    if alias_exists(client):
        ver = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        print(f"Alias '{ALIAS}' sudah ada (v{ver.version}), "
              "registry tidak diubah.")
        return
    run([sys.executable, "train.py"] + TRAIN_ARGS)
    run([sys.executable, "validate_model.py"])
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest = max(int(v.version) for v in versions)
    client.set_registered_model_alias(MODEL_NAME, ALIAS, str(latest))
    print(f"Model v{latest} lolos validasi dan diberi alias '{ALIAS}'.")


if __name__ == "__main__":
    main()
