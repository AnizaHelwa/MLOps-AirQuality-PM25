FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    mlflow==3.17.1 xgboost==3.2.0 pandas==3.0.6 numpy==2.4.6 \
    scikit-learn==1.9.1 fastapi uvicorn

WORKDIR /app
COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
