"""
Prediction API routes.

Endpoints:
    POST /api/predict       — Single text classification + risk score + explanation
    POST /api/predict/batch — Bulk classification
    POST /api/analyze       — Deep analysis with full explainability
"""

import time
from fastapi import APIRouter, HTTPException

from src.api.schemas import PredictRequest, BatchPredictRequest, PredictionResponse
from src.inference.predict import predict
from src.inference.risk_scorer import calculate_risk_score
from src.inference.explainability import MentalHealthExplainer
from src.api.crisis_alert import check_crisis_risk

router = APIRouter(prefix="/api", tags=["Prediction"])

# These will be set by main.py during startup
app_state = {}


def set_app_state(state: dict):
    global app_state
    app_state = state


def _ensure_model():
    if "model" not in app_state:
        raise HTTPException(status_code=503, detail="Model not loaded. Train a model first.")


@router.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictRequest):
    """Classify a single text for mental health indicators with risk score and explainability."""
    _ensure_model()

    t0 = time.time()

    # Get prediction
    results = predict(
        [request.text],
        app_state["model"],
        app_state["tokenizer"],
        app_state["label_encoder"],
    )
    result = results[0]

    # Risk score
    risk = calculate_risk_score(
        text=request.text,
        prediction=result["prediction"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
    )

    # Explainability
    try:
        explainer = MentalHealthExplainer(
            model=app_state["model"],
            tokenizer=app_state["tokenizer"],
            label_encoder=app_state["label_encoder"],
        )
        explanation = explainer.explain(request.text, top_n_words=8)
        explanations = explanation.get("explanations", [])
    except Exception:
        explanations = []

    latency_ms = (time.time() - t0) * 1000

    # Log to DB
    if "db_log" in app_state:
        app_state["db_log"](
            username="api_user",
            text=request.text,
            prediction=result["prediction"],
            confidence=result["confidence"],
            risk_flag=risk["is_crisis"],
            latency_ms=latency_ms,
        )

    return PredictionResponse(
        text=request.text,
        prediction=result["prediction"],
        confidence=round(result["confidence"], 4),
        risk_score=risk["risk_score"],
        severity=risk["severity"],
        is_crisis=risk["is_crisis"],
        probabilities={k: round(v, 4) for k, v in result["probabilities"].items()},
        explanations=explanations,
        recommendations=risk["recommendations"],
        latency_ms=round(latency_ms, 2),
    )


@router.post("/predict/batch")
async def predict_batch(request: BatchPredictRequest):
    """Classify multiple texts at once."""
    _ensure_model()

    if len(request.texts) > 500:
        raise HTTPException(status_code=400, detail="Batch limit is 500 texts.")

    t0 = time.time()
    results = predict(
        request.texts,
        app_state["model"],
        app_state["tokenizer"],
        app_state["label_encoder"],
    )
    total_ms = (time.time() - t0) * 1000

    output = []
    for r in results:
        risk = calculate_risk_score(
            text=r["text"],
            prediction=r["prediction"],
            confidence=r["confidence"],
            probabilities=r["probabilities"],
        )
        output.append({
            "text": r["text"][:100],
            "prediction": r["prediction"],
            "confidence": round(r["confidence"], 4),
            "risk_score": risk["risk_score"],
            "severity": risk["severity"],
            "is_crisis": risk["is_crisis"],
        })

    return {
        "count": len(output),
        "total_ms": round(total_ms, 2),
        "predictions": output,
    }


@router.post("/analyze")
async def analyze_deep(request: PredictRequest):
    """Deep analysis — prediction + full risk breakdown + explanations."""
    _ensure_model()

    t0 = time.time()

    results = predict(
        [request.text],
        app_state["model"],
        app_state["tokenizer"],
        app_state["label_encoder"],
    )
    result = results[0]

    risk = calculate_risk_score(
        text=request.text,
        prediction=result["prediction"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
    )

    try:
        explainer = MentalHealthExplainer(
            model=app_state["model"],
            tokenizer=app_state["tokenizer"],
            label_encoder=app_state["label_encoder"],
        )
        explanation = explainer.explain(request.text, top_n_words=12)
    except Exception:
        explanation = {"explanations": [], "method": "unavailable"}

    latency_ms = (time.time() - t0) * 1000

    return {
        "text": request.text,
        "prediction": result["prediction"],
        "confidence": round(result["confidence"], 4),
        "probabilities": {k: round(v, 4) for k, v in result["probabilities"].items()},
        "risk": risk,
        "explanation": explanation,
        "latency_ms": round(latency_ms, 2),
        "disclaimer": "This is an AI-based screening tool, NOT a medical diagnosis.",
    }
