"""
Crisis Alert System for the Mental Health Early Warning System.

Flags predictions that may indicate immediate crisis risk.
"""

import logging
from datetime import datetime

logging.basicConfig(level=logging.WARNING)
crisis_logger = logging.getLogger("crisis_alert")

HIGH_RISK_LABELS = {"suicidewatch", "suicide", "selfharm", "crisis", "depression"}
ALWAYS_ALERT = {"suicidewatch", "suicide", "selfharm", "crisis"}
HIGH_CONFIDENCE_THRESHOLD = 0.85


def check_crisis_risk(prediction: str, confidence: float) -> bool:
    """Returns True if the prediction should be flagged as a crisis risk."""
    label_lower = prediction.lower()

    if label_lower in ALWAYS_ALERT:
        crisis_logger.warning(
            f"[CRISIS ALERT] High-risk label: '{prediction}' "
            f"(confidence: {confidence:.1%}) at {datetime.utcnow().isoformat()}"
        )
        return True

    if label_lower in HIGH_RISK_LABELS and confidence >= HIGH_CONFIDENCE_THRESHOLD:
        crisis_logger.warning(
            f"[CRISIS FLAG] Elevated risk: '{prediction}' "
            f"(confidence: {confidence:.1%})"
        )
        return True

    return False


def get_crisis_resources() -> dict:
    """Return crisis support resources."""
    return {
        "message": "This prediction has been flagged for clinical review.",
        "resources": {
            "iCall (India)": "9152987821",
            "Vandrevala Foundation": "1860-2662-345",
            "International Crisis Lines": "https://findahelpline.com",
        },
        "note": "If you or someone you know is in immediate danger, please contact emergency services."
    }
