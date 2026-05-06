"""
Crisis Alert System for the Mental Health Early Warning System.

Flags predictions that may indicate immediate crisis risk.
High-risk categories trigger alert logging and can be extended
to send notifications (email, Slack, webhook).
"""

import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.WARNING)
crisis_logger = logging.getLogger("crisis_alert")

# Subreddits/labels considered high-risk
HIGH_RISK_LABELS = {
    "suicidewatch", "suicide", "selfharm", "crisis",
    "depression",   # flagged when confidence is very high
}

# Labels that are high-risk at ANY confidence level
ALWAYS_ALERT = {"suicidewatch", "suicide", "selfharm", "crisis"}

# Confidence threshold for moderate-risk labels to become alerts
HIGH_CONFIDENCE_THRESHOLD = 0.85


def check_crisis_risk(prediction: str, confidence: float) -> bool:
    """
    Returns True if the prediction should be flagged as a crisis risk.

    Rules:
    - Any prediction in ALWAYS_ALERT -> True regardless of confidence
    - 'depression' with confidence > 85% -> True
    - Everything else -> False
    """
    label_lower = prediction.lower()

    if label_lower in ALWAYS_ALERT:
        crisis_logger.warning(
            f"[CRISIS ALERT] High-risk label detected: '{prediction}' "
            f"(confidence: {confidence:.1%}) at {datetime.utcnow().isoformat()}"
        )
        return True

    if label_lower in HIGH_RISK_LABELS and confidence >= HIGH_CONFIDENCE_THRESHOLD:
        crisis_logger.warning(
            f"[CRISIS FLAG] Elevated risk: '{prediction}' "
            f"(confidence: {confidence:.1%}) — threshold exceeded"
        )
        return True

    return False


def get_crisis_resources() -> dict:
    """Return crisis support resources to include in high-risk API responses."""
    return {
        "message": "This prediction has been flagged for clinical review.",
        "resources": {
            "iCall (India)": "9152987821",
            "Vandrevala Foundation": "1860-2662-345",
            "iCall WhatsApp": "+91 9152987821",
            "International Crisis Lines": "https://findahelpline.com",
        },
        "note": "If you or someone you know is in immediate danger, please contact emergency services."
    }
