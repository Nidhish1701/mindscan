"""
FastAPI REST API for the Mental Health Early Warning System.

Endpoints:
    POST /auth/login           — get JWT token
    POST /api/predict          — classify a single comment
    POST /api/predict/batch    — classify multiple comments
    GET  /api/history          — get prediction history
    GET  /api/stats            — model performance stats
    GET  /docs                 — Swagger UI (auto-generated)

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import time
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional

import torch
import pickle
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from src.predict import load_model, predict
from api.crisis_alert import check_crisis_risk

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

SECRET_KEY = os.getenv("SECRET_KEY", "change-this-in-production-please")
ALGORITHM  = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
MODEL_DIR  = os.getenv("MODEL_DIR", "models/distilbert/best_model")
DB_PATH    = os.getenv("DB_PATH", "data/predictions.db")

# --------------------------------------------------
# AUTH
# --------------------------------------------------

pwd_context   = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# Demo users — replace with a real DB in production
USERS_DB = {
    "admin":    {"password": pwd_context.hash("admin123"),  "role": "admin"},
    "clinician": {"password": pwd_context.hash("clinic123"), "role": "analyst"},
    "viewer":   {"password": pwd_context.hash("view123"),   "role": "viewer"},
}

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode["exp"] = expire
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None or username not in USERS_DB:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return {"username": username, **USERS_DB[username]}

# --------------------------------------------------
# DATABASE
# --------------------------------------------------

def init_db():
    """Create predictions table if it doesn't exist."""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            username    TEXT    NOT NULL,
            text        TEXT    NOT NULL,
            prediction  TEXT    NOT NULL,
            confidence  REAL    NOT NULL,
            risk_flag   INTEGER NOT NULL DEFAULT 0,
            latency_ms  REAL
        )
    """)
    conn.commit()
    conn.close()

def log_prediction(username: str, text: str, prediction: str,
                   confidence: float, risk_flag: bool, latency_ms: float):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions (timestamp, username, text, prediction, confidence, risk_flag, latency_ms) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), username, text, prediction,
         round(confidence, 4), int(risk_flag), round(latency_ms, 2))
    )
    conn.commit()
    conn.close()

# --------------------------------------------------
# APP STATE (model loaded once at startup)
# --------------------------------------------------

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup, release at shutdown."""
    print(f"[Startup] Loading model from {MODEL_DIR} ...")
    try:
        model, tokenizer, label_encoder = load_model(MODEL_DIR)
        app_state["model"]         = model
        app_state["tokenizer"]     = tokenizer
        app_state["label_encoder"] = label_encoder
        print(f"[Startup] Model loaded. Classes: {list(label_encoder.classes_)}")
    except Exception as e:
        print(f"[Startup] WARNING: Could not load model — {e}")
        print("[Startup] API will start but /predict will return 503 until model is available.")
    init_db()
    yield
    app_state.clear()
    print("[Shutdown] Model unloaded.")

# --------------------------------------------------
# APP
# --------------------------------------------------

app = FastAPI(
    title="Mental Health Early Warning System",
    description="ML-powered API for detecting mental health signals in Reddit comments.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# SCHEMAS
# --------------------------------------------------

class Token(BaseModel):
    access_token: str
    token_type: str

class PredictRequest(BaseModel):
    text: str
    class Config:
        json_schema_extra = {"example": {"text": "I have been feeling really hopeless lately"}}

class BatchPredictRequest(BaseModel):
    texts: list[str]

class PredictionResponse(BaseModel):
    text: str
    prediction: str
    confidence: float
    risk_flag: bool
    probabilities: dict[str, float]
    latency_ms: float

# --------------------------------------------------
# AUTH ROUTES
# --------------------------------------------------

@app.post("/auth/login", response_model=Token, tags=["Auth"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Get a JWT access token. Demo credentials: admin/admin123"""
    user = USERS_DB.get(form_data.username)
    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    token = create_access_token(
        data={"sub": form_data.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": token, "token_type": "bearer"}

# --------------------------------------------------
# PREDICT ROUTES
# --------------------------------------------------

@app.post("/api/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(
    request: PredictRequest,
    current_user: dict = Depends(get_current_user)
):
    """Classify a single Reddit comment for mental health signals."""
    if "model" not in app_state:
        raise HTTPException(status_code=503, detail="Model not loaded. Train a model first.")

    t0 = time.time()
    results = predict(
        [request.text],
        app_state["model"],
        app_state["tokenizer"],
        app_state["label_encoder"],
    )
    latency_ms = (time.time() - t0) * 1000
    result = results[0]

    risk_flag = check_crisis_risk(result["prediction"], result["confidence"])

    log_prediction(
        username   = current_user["username"],
        text       = request.text,
        prediction = result["prediction"],
        confidence = result["confidence"],
        risk_flag  = risk_flag,
        latency_ms = latency_ms,
    )

    return PredictionResponse(
        text          = request.text,
        prediction    = result["prediction"],
        confidence    = round(result["confidence"], 4),
        risk_flag     = risk_flag,
        probabilities = {k: round(v, 4) for k, v in result["probabilities"].items()},
        latency_ms    = round(latency_ms, 2),
    )


@app.post("/api/predict/batch", tags=["Prediction"])
async def predict_batch(
    request: BatchPredictRequest,
    current_user: dict = Depends(get_current_user)
):
    """Classify multiple comments at once."""
    if "model" not in app_state:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if len(request.texts) > 500:
        raise HTTPException(status_code=400, detail="Batch size limit is 500 texts.")

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
        risk = check_crisis_risk(r["prediction"], r["confidence"])
        output.append({
            "text":          r["text"],
            "prediction":    r["prediction"],
            "confidence":    round(r["confidence"], 4),
            "risk_flag":     risk,
            "probabilities": {k: round(v, 4) for k, v in r["probabilities"].items()},
        })
        log_prediction(current_user["username"], r["text"], r["prediction"],
                       r["confidence"], risk, total_ms / len(results))

    return {
        "count":       len(output),
        "total_ms":    round(total_ms, 2),
        "predictions": output,
    }

# --------------------------------------------------
# HISTORY + STATS ROUTES
# --------------------------------------------------

@app.get("/api/history", tags=["Analytics"])
async def get_history(
    limit: int = 50,
    current_user: dict = Depends(get_current_user)
):
    """Get recent prediction history."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT timestamp, username, text, prediction, confidence, risk_flag, latency_ms "
        "FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [
        {
            "timestamp":  r[0], "username": r[1], "text": r[2][:80] + "..." if len(r[2]) > 80 else r[2],
            "prediction": r[3], "confidence": r[4], "risk_flag": bool(r[5]), "latency_ms": r[6]
        }
        for r in rows
    ]


@app.get("/api/stats", tags=["Analytics"])
async def get_stats(current_user: dict = Depends(get_current_user)):
    """Get aggregate prediction statistics."""
    conn = sqlite3.connect(DB_PATH)
    total      = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    risk_count = conn.execute("SELECT COUNT(*) FROM predictions WHERE risk_flag=1").fetchone()[0]
    avg_conf   = conn.execute("SELECT AVG(confidence) FROM predictions").fetchone()[0]
    avg_lat    = conn.execute("SELECT AVG(latency_ms) FROM predictions").fetchone()[0]
    dist       = conn.execute(
        "SELECT prediction, COUNT(*) FROM predictions GROUP BY prediction"
    ).fetchall()
    conn.close()
    return {
        "total_predictions": total,
        "crisis_flags":      risk_count,
        "crisis_rate_pct":   round((risk_count / total * 100) if total else 0, 2),
        "avg_confidence":    round(avg_conf or 0, 4),
        "avg_latency_ms":    round(avg_lat or 0, 2),
        "label_distribution": {r[0]: r[1] for r in dist},
    }


@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "healthy",
        "model_loaded": "model" in app_state,
        "timestamp": datetime.utcnow().isoformat(),
    }
