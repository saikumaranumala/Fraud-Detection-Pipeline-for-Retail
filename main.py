"""
src/api/main.py
----------------
FastAPI REST endpoint for real-time fraud scoring.

Endpoints:
  POST /predict        - Score a single transaction
  POST /predict/batch  - Score a batch of transactions
  GET  /health         - Health check + model version
  GET  /metrics        - Prometheus metrics exposition
"""

import time
import logging
import joblib
import os
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST,
)
from starlette.responses import Response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time e-commerce transaction fraud scoring",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

REQUEST_COUNT = Counter(
    "fraud_api_requests_total",
    "Total number of scoring requests",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "fraud_api_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
FRAUD_SCORE_DIST = Histogram(
    "fraud_score_distribution",
    "Distribution of fraud scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
FRAUD_FLAG_COUNT = Counter(
    "fraud_flags_total",
    "Total transactions flagged as fraud",
)
MODEL_VERSION = Gauge(
    "fraud_model_version",
    "Currently loaded model version",
)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class TransactionRequest(BaseModel):
    transaction_id:        str
    user_id:               str
    merchant_id:           str
    amount:                float = Field(..., gt=0)
    timestamp:             str
    merchant_category_code: str
    device_type:           str
    ip_address:            str
    billing_country:       str
    shipping_country:      str
    card_present:          bool
    review_text:           Optional[str] = ""
    transaction_note:      Optional[str] = ""

    # Pre-computed features (optional — sent from feature store for low latency)
    txn_count_1h:          Optional[float] = None
    txn_count_24h:         Optional[float] = None
    amount_zscore:         Optional[float] = None
    merchant_fraud_rate:   Optional[float] = None

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "TXN_98271834",
                "user_id": "U00421",
                "merchant_id": "M1234",
                "amount": 1250.00,
                "timestamp": "2024-06-15T02:33:00",
                "merchant_category_code": "5999",
                "device_type": "mobile",
                "ip_address": "192.168.1.1",
                "billing_country": "US",
                "shipping_country": "NG",
                "card_present": False,
                "transaction_note": "urgent transfer needed immediately",
                "review_text": "never received my item",
            }
        }


class FraudPredictionResponse(BaseModel):
    transaction_id:   str
    fraud_score:      float
    is_fraud:         int
    confidence_tier:  str
    override_fired:   bool
    signal_breakdown: dict
    latency_ms:       float
    model_version:    str


class BatchRequest(BaseModel):
    transactions: List[TransactionRequest]


class BatchResponse(BaseModel):
    results:       List[FraudPredictionResponse]
    total:         int
    flagged_count: int
    avg_latency_ms: float


class HealthResponse(BaseModel):
    status:        str
    model_version: str
    uptime_seconds: float


# ---------------------------------------------------------------------------
# Model loading (at startup)
# ---------------------------------------------------------------------------

ensemble     = None
MODEL_VER    = os.getenv("MODEL_VERSION", "1.0.0")
START_TIME   = time.time()


@app.on_event("startup")
async def load_models():
    global ensemble
    model_path = os.getenv("MODEL_PATH", "models/ensemble.joblib")
    try:
        ensemble = joblib.load(model_path)
        MODEL_VERSION.set(float(MODEL_VER.replace(".", "")))
        logger.info(f"Ensemble model loaded from {model_path} (v{MODEL_VER})")
    except FileNotFoundError:
        logger.warning(f"Model file not found at {model_path}. "
                       "Running in demo mode — predictions will be random.")


def _get_ensemble():
    if ensemble is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ensemble


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok" if ensemble else "degraded",
        model_version=MODEL_VER,
        uptime_seconds=round(time.time() - START_TIME, 2),
    )


@app.get("/metrics")
def metrics():
    """Prometheus metrics scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=FraudPredictionResponse)
def predict(request: TransactionRequest):
    t0  = time.time()
    mdl = _get_ensemble()

    try:
        txn_dict = request.dict()
        txn_dict["timestamp"] = pd.Timestamp(txn_dict["timestamp"])

        result = mdl.score_single(txn_dict)
        latency_ms = (time.time() - t0) * 1000

        # Prometheus
        REQUEST_COUNT.labels(endpoint="/predict", status="200").inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(latency_ms / 1000)
        FRAUD_SCORE_DIST.observe(result["fraud_score"])
        if result["is_fraud"]:
            FRAUD_FLAG_COUNT.inc()

        return FraudPredictionResponse(
            **result,
            latency_ms=round(latency_ms, 2),
            model_version=MODEL_VER,
        )

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict", status="500").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    t0  = time.time()
    mdl = _get_ensemble()

    try:
        rows = [t.dict() for t in request.transactions]
        for r in rows:
            r["timestamp"] = pd.Timestamp(r["timestamp"])

        df     = pd.DataFrame(rows)
        scored = mdl.score(df)

        results = []
        for _, row in scored.iterrows():
            results.append(FraudPredictionResponse(
                transaction_id  = str(row.get("transaction_id", "unknown")),
                fraud_score     = round(float(row["fraud_score"]), 4),
                is_fraud        = int(row["is_fraud_pred"]),
                confidence_tier = str(row["confidence_tier"]),
                override_fired  = bool(row["override_flag"]),
                signal_breakdown={
                    "xgboost":    round(float(row["xgb_score"]), 4),
                    "autoencoder": round(float(row["ae_score"]), 4),
                    "nlp":        round(float(row["nlp_score"]), 4),
                },
                latency_ms    = 0,
                model_version = MODEL_VER,
            ))

        total_latency_ms = (time.time() - t0) * 1000
        flagged = sum(1 for r in results if r.is_fraud)

        REQUEST_COUNT.labels(endpoint="/predict/batch", status="200").inc()
        REQUEST_LATENCY.labels(endpoint="/predict/batch").observe(total_latency_ms / 1000)

        return BatchResponse(
            results        = results,
            total          = len(results),
            flagged_count  = flagged,
            avg_latency_ms = round(total_latency_ms / max(len(results), 1), 2),
        )

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict/batch", status="500").inc()
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
