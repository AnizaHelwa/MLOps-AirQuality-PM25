
import logging
import os
import time

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

logging.basicConfig(level=logging.INFO)

NGINX_URL = os.getenv("NGINX_URL", "http://model-lb:80")

app = FastAPI(title="PM2.5 Metrics Gateway")

REQUESTS = Counter(
    "inference_requests_total",
    "Total inference requests",
    ["method", "endpoint", "status"],
)

LATENCY = Histogram(
    "inference_latency_seconds",
    "Inference request latency in seconds",
    ["endpoint"],
)

PREDICTIONS = Histogram(
    "pm25_prediction_value",
    "Distribution of predicted PM2.5 values",
    buckets=(0, 5, 10, 15, 25, 35, 50, 75, 100, 150, 200, 300, 500),
)

REPLICA_REQUESTS = Counter(
    "replica_requests_total",
    "Total requests served by each replica",
    ["replica"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/invocations")
async def invocations(request: Request):
    start = time.perf_counter()
    status = "500"

    try:
        body = await request.body()

        async with httpx.AsyncClient(timeout=90.0) as client:
            upstream = await client.post(
                f"{NGINX_URL}/invocations",
                content=body,
                headers={
                    "Content-Type": request.headers.get(
                        "content-type", "application/json"
                    )
                },
            )

        status = str(upstream.status_code)

        replica = upstream.headers.get("X-Served-By", "unknown")
        REPLICA_REQUESTS.labels(replica=replica).inc()

        if upstream.status_code >= 400:
            raise HTTPException(
                status_code=upstream.status_code,
                detail=upstream.text,
            )

        result = upstream.json()

        predictions = result.get("predictions", [])

        for value in predictions:
            PREDICTIONS.observe(float(value))

        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type="application/json",
            headers={
                "X-Served-By": replica,
            },
        )

    except HTTPException:
        raise

    except Exception as exc:
        status = "502"
        logging.exception("Inference gateway error")
        raise HTTPException(status_code=502, detail=str(exc))

    finally:
        elapsed = time.perf_counter() - start
        REQUESTS.labels(
            method="POST",
            endpoint="/invocations",
            status=status,
        ).inc()

        LATENCY.labels(endpoint="/invocations").observe(elapsed)
