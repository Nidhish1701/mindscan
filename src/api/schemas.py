"""
Pydantic schemas for the Mental Health API.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=3, description="Text to analyze")

    class Config:
        json_schema_extra = {"example": {"text": "I have been feeling really hopeless lately"}}


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=500)


class ExplanationItem(BaseModel):
    word: str
    importance: float
    shap_value: float
    impact: str


class RiskComponents(BaseModel):
    category_base: float
    confidence_boost: float
    keyword_severity: float
    multi_category: float


class RiskResult(BaseModel):
    risk_score: int
    severity: str
    description: str
    is_crisis: bool
    components: RiskComponents
    recommendations: List[str]


class PredictionResponse(BaseModel):
    text: str
    prediction: str
    confidence: float
    risk_score: int
    severity: str
    is_crisis: bool
    probabilities: Dict[str, float]
    explanations: List[ExplanationItem]
    recommendations: List[str]
    latency_ms: float
    disclaimer: str = "⚠️ This is an AI-based screening tool, NOT a medical diagnosis. Please consult a qualified mental health professional for clinical assessment."


# --------------------------------------------------
# CHATBOT SCHEMAS
# --------------------------------------------------

class MentalHealthAssessment(BaseModel):
    """ML-powered mental health assessment from chatbot."""
    detected_condition: str = Field(..., description="Detected condition: one of Anxiety, BPD, Bipolar, Depression, Mental Illness, Schizophrenia, Normal, or Unknown")
    confidence: Optional[float] = Field(None, description="Detection probability 0-1")
    is_normal: bool = Field(False, description="True if user is in a normal/healthy mental state")
    ml_prediction: Optional[str] = Field(None, description="Raw ML model prediction (may differ from detected_condition for Normal states)")
    ml_confidence: Optional[float] = Field(None, description="Raw ML model confidence")
    note: Optional[str] = Field(None, description="Brief note about the assessment")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    mood_detected: Optional[str] = None
    suggestions: List[str] = []
    resources: Optional[Dict] = None
    mental_health_assessment: Optional[MentalHealthAssessment] = None


class DashboardData(BaseModel):
    total_predictions: int
    crisis_flags: int
    crisis_rate_pct: float
    avg_confidence: float
    avg_latency_ms: float
    label_distribution: Dict[str, int]
    recent_predictions: List[Dict]
    hourly_trend: List[Dict]
