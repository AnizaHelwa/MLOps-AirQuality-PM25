FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    mlflow==3.17.1 xgboost==3.2.0 pandas==3.0.6 numpy==2.4.6 \
    scikit-learn==1.9.1 pyarrow==25.0.1 minio==7.2.20

WORKDIR /app
COPY src/train.py src/validate_model.py docker/trainer/bootstrap.py ./

CMD ["python", "bootstrap.py"]
