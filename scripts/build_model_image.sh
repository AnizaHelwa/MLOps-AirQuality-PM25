#!/usr/bin/env bash
# Membangun image serving dari model Production di registry lokal (LK-07).
set -euo pipefail
cd "$(dirname "$0")/.."
MLFLOW_TRACKING_URI="sqlite:///$(pwd)/src/mlflow.db" \
mlflow models build-docker \
  -m "models:/pm25-forecasting-xgboost@production" \
  -n pm25-model-serve:1.0 \
  --env-manager virtualenv \
  --install-mlflow
