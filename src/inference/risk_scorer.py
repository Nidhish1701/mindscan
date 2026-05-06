"""
Mental Health Risk Score Calculator (0-100).

This is a UNIQUE feature that combines:
    1. Model prediction confidence
    2. Keyword severity analysis (crisis/distress words)
    3. Text emotional intensity
    4. Multi-category risk weighting

Outputs a single 0-100 risk score with severity level.
"""

import re
from typing import Dict, Tuple

# Crisis-level keywords (highest severity)
CRISIS_KEYWORDS = {
    'suicide', 'kill myself', 'end my life', 'want to die', 'suicidal',
    'self-harm', 'self harm', 'cutting', 'overdose', 'no reason to live',
    'better off dead', 'end it all', 'goodbye forever', 'final note',
    'not worth living', 'can\'t go on', 'no way out',
}

# High distress keywords
DISTRESS_KEYWORDS = {
    'hopeless', 'worthless', 'helpless', 'empty', 'numb', 'alone',
    'isolated', 'broken', 'exhausted', 'overwhelmed', 'suffocating',
    'drowning', 'trapped', 'unbearable', 'agonizing', 'miserable',
    'devastated', 'shattered', 'crushed', 'despair', 'tormented',
    'can\'t cope', 'falling apart', 'giving up', 'lost everything',
}

# Moderate concern keywords
CONCERN_KEYWORDS = {
    'anxious', 'worried', 'stressed', 'depressed', 'sad', 'scared',
    'panicking', 'panic', 'crying', 'insomnia', 'nightmares',
    'can\'t sleep', 'can\'t eat', 'no motivation', 'no energy',
    'struggling', 'suffering', 'hurting', 'afraid', 'nervous',
}

# Category risk weights (some disorders inherently higher risk)
CATEGORY_RISK_WEIGHTS = {
    'Depression':      0.75,
    'Anxiety':         0.55,
    'BPD':             0.70,
    'Bipolar':         0.65,
    'Schizophrenia':   0.70,
    'Mental Illness':  0.50,
    'Normal':          0.0,
}

# Severity levels
SEVERITY_LEVELS = [
    (0,  20, 'Low',      'No significant mental health concerns detected.'),
    (20, 40, 'Mild',     'Some indicators of emotional distress present.'),
    (40, 60, 'Moderate', 'Notable mental health indicators detected. Consider seeking support.'),
    (60, 80, 'High',     'Significant mental health risk indicators. Professional support recommended.'),
    (80, 100, 'Critical', 'Critical risk level detected. Immediate professional support strongly recommended.'),
]


def calculate_risk_score(
    text: str,
    prediction: str,
    confidence: float,
    probabilities: Dict[str, float],
) -> Dict:
    """
    Calculate a comprehensive Mental Health Risk Score (0-100).

    Components:
        1. Base score from prediction category (0-25)
        2. Confidence amplifier (0-25)
        3. Keyword severity score (0-30)
        4. Multi-category risk (0-20)

    Returns:
        Dict with score, severity, components, and recommendations
    """
    text_lower = text.lower()

    # ---- Component 1: Category Base Score (0-25) ----
    category_weight = CATEGORY_RISK_WEIGHTS.get(prediction, 0.5)
    base_score = category_weight * 25

    # ---- Component 2: Confidence Amplifier (0-25) ----
    # If the prediction is Normal, high confidence means lower risk, so we don't add to the score
    confidence_score = 0.0 if prediction == 'Normal' else confidence * 25

    # ---- Component 3: Keyword Severity (0-30) ----
    crisis_count  = sum(1 for kw in CRISIS_KEYWORDS  if kw in text_lower)
    distress_count = sum(1 for kw in DISTRESS_KEYWORDS if kw in text_lower)
    concern_count = sum(1 for kw in CONCERN_KEYWORDS  if kw in text_lower)

    keyword_score = min(30, (
        crisis_count * 15 +
        distress_count * 5 +
        concern_count * 2
    ))

    # ---- Component 4: Multi-Category Risk (0-20) ----
    # Higher score if multiple high-risk categories have significant probability
    high_risk_cats = ['Depression', 'BPD', 'Schizophrenia', 'Bipolar']
    multi_risk = sum(
        probabilities.get(cat, 0) * CATEGORY_RISK_WEIGHTS.get(cat, 0.5)
        for cat in high_risk_cats
    )
    multi_score = min(20, multi_risk * 25)

    # ---- Total Risk Score ----
    raw_score = base_score + confidence_score + keyword_score + multi_score
    risk_score = min(100, max(0, round(raw_score)))

    # ---- Determine Severity ----
    severity = 'Low'
    description = ''
    for low, high, level, desc in SEVERITY_LEVELS:
        if low <= risk_score < high or (high == 100 and risk_score == 100):
            severity = level
            description = desc
            break

    # ---- Crisis Flag ----
    is_crisis = crisis_count > 0 or risk_score >= 80

    # ---- Recommendations ----
    recommendations = _get_recommendations(severity, prediction, is_crisis)

    return {
        'risk_score':   risk_score,
        'severity':     severity,
        'description':  description,
        'is_crisis':    is_crisis,
        'components': {
            'category_base':    round(base_score, 1),
            'confidence_boost': round(confidence_score, 1),
            'keyword_severity': round(keyword_score, 1),
            'multi_category':   round(multi_score, 1),
        },
        'keywords_detected': {
            'crisis':  crisis_count,
            'distress': distress_count,
            'concern': concern_count,
        },
        'recommendations': recommendations,
    }


def _get_recommendations(severity: str, prediction: str, is_crisis: bool) -> list:
    """Generate appropriate recommendations based on severity level."""
    recs = []

    if is_crisis:
        recs.append("🚨 If you or someone you know is in immediate danger, please contact emergency services.")
        recs.append("📞 Crisis Helplines: iCall (9152987821) | Vandrevala Foundation (1860-2662-345)")
        recs.append("🌐 International: https://findahelpline.com")

    if severity in ('Critical', 'High'):
        recs.append("💬 Please consider reaching out to a licensed mental health professional.")
        recs.append("🧘 Practice grounding: name 5 things you see, 4 you can touch, 3 you hear.")
        recs.append("📝 Consider maintaining a daily mood journal to track patterns.")

    elif severity == 'Moderate':
        recs.append("💭 Consider speaking to a counselor or trusted person about how you're feeling.")
        recs.append("🏃 Regular physical activity can significantly improve mental wellbeing.")
        recs.append("😴 Prioritize sleep hygiene — aim for 7-9 hours of quality sleep.")

    elif severity == 'Mild':
        recs.append("🌱 Practice daily mindfulness or meditation (even 5 minutes helps).")
        recs.append("👥 Stay connected with friends and family.")
        recs.append("📖 Consider self-help resources or mental health apps.")

    else:
        recs.append("✅ No significant risk indicators detected.")
        recs.append("🌟 Continue maintaining healthy habits and social connections.")

    recs.append("⚠️ This is an AI-based screening tool, NOT a medical diagnosis.")

    return recs
