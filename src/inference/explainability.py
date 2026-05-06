"""
Explainability module using SHAP for the Mental Health Early Warning System.

Provides word-level explanations for model predictions — shows WHICH words
drove the classification. Critical for clinical trust and interpretability.

Falls back to a keyword-based explainer if SHAP is not installed,
ensuring the system always works.

Usage:
    from src.inference.explainability import MentalHealthExplainer
    explainer = MentalHealthExplainer("models/distilbert/best_model")
    result = explainer.explain("I have been feeling hopeless and empty")
"""

import os
import re
import numpy as np
import torch
from typing import Dict, List

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Keyword importance for fallback explainer
KEYWORD_WEIGHTS = {
    # Depression indicators
    'hopeless': 0.9, 'worthless': 0.88, 'empty': 0.85, 'numb': 0.82,
    'depressed': 0.95, 'suicidal': 0.99, 'crying': 0.7, 'alone': 0.75,
    'exhausted': 0.65, 'guilt': 0.7, 'failure': 0.72, 'sad': 0.6,
    'miserable': 0.8, 'despair': 0.88, 'darkness': 0.7, 'suffering': 0.75,

    # Anxiety indicators
    'anxious': 0.9, 'panic': 0.88, 'worried': 0.7, 'fear': 0.75,
    'terrified': 0.85, 'overthinking': 0.72, 'racing': 0.65, 'nervous': 0.68,
    'dread': 0.78, 'phobia': 0.7, 'restless': 0.6, 'palpitations': 0.65,

    # General distress
    'struggling': 0.6, 'overwhelmed': 0.72, 'trapped': 0.78, 'broken': 0.75,
    'help': 0.5, 'can\'t': 0.45, 'never': 0.4, 'always': 0.35,
    'hate': 0.55, 'angry': 0.5, 'rage': 0.6, 'frustrated': 0.5,

    # Positive indicators (negative SHAP = argues against disorder)
    'happy': -0.5, 'grateful': -0.55, 'better': -0.4, 'improving': -0.45,
    'hopeful': -0.6, 'recovered': -0.65, 'healthy': -0.5, 'good': -0.3,
}


class MentalHealthExplainer:
    """
    Provides word-level explanations for mental health predictions.

    Uses SHAP when available, falls back to keyword importance analysis.
    """

    def __init__(self, model_dir: str = None, model=None, tokenizer=None, label_encoder=None, max_length: int = 128):
        if model_dir and model is None:
            from src.inference.predict import load_model
            self.model, self.tokenizer, self.label_encoder = load_model(model_dir)
        else:
            self.model = model
            self.tokenizer = tokenizer
            self.label_encoder = label_encoder

        self.max_length = max_length
        self.classes_ = list(self.label_encoder.classes_) if self.label_encoder else []

    def explain(self, text: str, top_n_words: int = 10) -> Dict:
        """
        Generate explanation for a prediction.

        If SHAP is available, uses true SHAP values.
        Otherwise, uses keyword-based importance (still useful for presentation).
        """
        if SHAP_AVAILABLE and self.model is not None:
            try:
                return self._explain_shap(text, top_n_words)
            except Exception as e:
                print(f"[Explainer] SHAP failed ({e}), using keyword fallback")

        return self._explain_keywords(text, top_n_words)

    def _explain_shap(self, text: str, top_n_words: int) -> Dict:
        """Full SHAP explanation."""
        predict_fn = self._get_prediction_fn()

        probs = predict_fn([text])[0]
        top_idx = int(np.argmax(probs))
        prediction = self.classes_[top_idx]
        confidence = float(probs[top_idx])

        masker = shap.maskers.Text(self.tokenizer)
        explainer = shap.Explainer(predict_fn, masker, output_names=self.classes_)
        shap_values = explainer([text])

        class_shap = shap_values.values[0, :, top_idx]
        words = shap_values.data[0]

        word_shap_pairs = sorted(
            zip(words, class_shap),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_n_words]

        explanations = [
            {
                'word': word,
                'importance': round(float(abs(shap_val)), 4),
                'shap_value': round(float(shap_val), 4),
                'impact': 'supports' if shap_val > 0 else 'opposes',
            }
            for word, shap_val in word_shap_pairs
            if word.strip()
        ]

        return {
            'text': text,
            'prediction': prediction,
            'confidence': round(confidence, 4),
            'method': 'SHAP',
            'explanations': explanations,
        }

    def _explain_keywords(self, text: str, top_n_words: int) -> Dict:
        """Keyword-based importance fallback (always works, no SHAP needed)."""
        # Get prediction first
        if self.model and self.tokenizer:
            from src.inference.predict import predict
            results = predict([text], self.model, self.tokenizer, self.label_encoder)
            result = results[0]
            prediction = result['prediction']
            confidence = result['confidence']
        else:
            prediction = 'Unknown'
            confidence = 0.0

        words = re.findall(r'\b\w+\b', text.lower())
        word_scores = []

        for word in set(words):
            if word in KEYWORD_WEIGHTS:
                score = KEYWORD_WEIGHTS[word]
                word_scores.append({
                    'word': word,
                    'importance': round(abs(score), 4),
                    'shap_value': round(score, 4),
                    'impact': 'supports' if score > 0 else 'opposes',
                })

        # Sort by absolute importance
        word_scores.sort(key=lambda x: x['importance'], reverse=True)

        return {
            'text': text,
            'prediction': prediction,
            'confidence': round(confidence, 4),
            'method': 'Keyword Analysis',
            'explanations': word_scores[:top_n_words],
        }

    def _get_prediction_fn(self):
        model = self.model
        tokenizer = self.tokenizer
        device = DEVICE
        max_len = self.max_length

        def predict_fn(texts):
            enc = tokenizer(
                list(texts), truncation=True, padding=True,
                max_length=max_len, return_tensors='pt',
            )
            with torch.no_grad():
                logits = model(
                    input_ids=enc['input_ids'].to(device),
                    attention_mask=enc['attention_mask'].to(device),
                ).logits
            return torch.softmax(logits, dim=1).cpu().numpy()

        return predict_fn
