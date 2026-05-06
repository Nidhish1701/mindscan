"""
Chatbot API route — with ML model integration.

Endpoint:
    POST /api/chatbot — Mental health support chatbot with ML classification
"""

from fastapi import APIRouter
from src.api.schemas import ChatRequest, ChatResponse
from src.chatbot.mental_health_bot import MentalHealthChatbot

router = APIRouter(prefix="/api", tags=["Chatbot"])

chatbot = MentalHealthChatbot()

# Will be called from main.py after model loads
_app_state = {}


def set_chatbot_model(state: dict):
    """Inject ML model into chatbot for classification."""
    global _app_state
    _app_state = state
    if "model" in state and "tokenizer" in state and "label_encoder" in state:
        chatbot.set_ml_model(
            model=state["model"],
            tokenizer=state["tokenizer"],
            label_encoder=state["label_encoder"],
        )
        print("[Chatbot] ML model connected — chatbot now uses real classification.")
    else:
        print("[Chatbot] ML model not available — chatbot using pattern matching only.")


@router.post("/chatbot", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Get a supportive response from the mental health chatbot with ML-powered assessment."""
    result = chatbot.respond(request.message)
    return ChatResponse(**result)
