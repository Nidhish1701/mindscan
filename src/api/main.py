"""
FastAPI Main Application — Mental Health Early Warning System.

Serves:
    - REST API for predictions, analytics, and chatbot
    - Static frontend files

Run:
    python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from src.api.routes_chatbot import router as chatbot_router, set_chatbot_model
from src.api.routes_dashboard import router as dashboard_router
from src.api.routes_predict import router as predict_router, set_app_state
import os
import sys
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

MODEL_DIR = os.getenv("MODEL_DIR", "models")
DB_PATH = os.getenv("DB_PATH", "data/predictions.db")

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


def log_prediction(username, text, prediction, confidence, risk_flag, latency_ms):
    """Log a prediction to the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO predictions (timestamp, username, text, prediction, confidence, risk_flag, latency_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (datetime.utcnow().isoformat(), username, text, prediction,
             round(confidence, 4), int(risk_flag), round(latency_ms, 2))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB] Warning: Could not log prediction: {e}")


# --------------------------------------------------
# APP STATE
# --------------------------------------------------

app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, release at shutdown."""
    print(f"[Startup] Loading model from {MODEL_DIR} ...")
    try:
        from src.inference.predict import load_model
        model, tokenizer, label_encoder = load_model(MODEL_DIR)
        app_state["model"] = model
        app_state["tokenizer"] = tokenizer
        app_state["label_encoder"] = label_encoder
        app_state["db_log"] = log_prediction
        set_app_state(app_state)
        set_chatbot_model(app_state)
        print(
            f"[Startup] Model loaded. Classes: {list(label_encoder.classes_)}")
    except Exception as e:
        print(f"[Startup] WARNING: Could not load model — {e}")
        print("[Startup] API will start but /predict will return 503.")
        app_state["db_log"] = log_prediction
        set_app_state(app_state)

    init_db()
    yield
    app_state.clear()
    print("[Shutdown] Model unloaded.")


# --------------------------------------------------
# APP
# --------------------------------------------------

app = FastAPI(
    title="Mental Health Early Warning System",
    description="ML-powered API for detecting mental health signals in text using NLP and Deep Learning.",
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

# Include routers
app.include_router(predict_router)
app.include_router(dashboard_router)
app.include_router(chatbot_router)


# --------------------------------------------------
# FRONTEND SERVING
# --------------------------------------------------

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'frontend')


@app.get("/", tags=["Frontend"])
async def serve_frontend():
    """Serve the main dashboard page."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Mental Health Early Warning System API", "docs": "/docs"}


# Mount static files (CSS, JS)
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "healthy",
        "model_loaded": "model" in app_state,
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
    }
