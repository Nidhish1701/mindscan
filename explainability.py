"""
Explainability module using SHAP for the Mental Health Early Warning System.

Provides word-level explanations for model predictions — shows WHICH words
drove the classification decision. Critical for clinical trust and interpretability.

Usage:
    python -m src.explainability \
        --model models/distilbert/best_model \
        --text "I have been feeling hopeless and empty for weeks"

Requirements:
    pip install shap transformers
"""

import argparse
import os
import pickle

import numpy as np
import torch

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[Warning] shap not installed. Run: pip install shap")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# EXPLAINER
# --------------------------------------------------

class MentalHealthExplainer:
    """
    Provides word-level SHAP explanations for mental health predictions.

    For each prediction, returns a list of (word, shap_value) pairs showing
    which words pushed the model toward or away from the predicted class.
    """

    def __init__(self, model_dir: str, max_length: int = 128):
        from src.predict import load_model

        self.model, self.tokenizer, self.label_encoder = load_model(model_dir)
        self.max_length = max_length
        self.classes_   = list(self.label_encoder.classes_)
        self._explainer = None

    def _get_prediction_fn(self):
        """
        Returns a function that takes raw text strings and outputs
        a numpy probability matrix — required by shap.Explainer.
        """
        model     = self.model
        tokenizer = self.tokenizer
        device    = DEVICE
        max_len   = self.max_length

        def predict_fn(texts):
            enc = tokenizer(
                list(texts),
                truncation=True,
                padding=True,
                max_length=max_len,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = model(
                    input_ids=enc["input_ids"].to(device),
                    attention_mask=enc["attention_mask"].to(device),
                ).logits
            return torch.softmax(logits, dim=1).cpu().numpy()

        return predict_fn

    def explain(self, text: str, top_n_words: int = 10) -> dict:
        """
        Generate a SHAP explanation for a single text prediction.

        Returns:
            {
                "prediction": str,
                "confidence": float,
                "explanations": [{"word": str, "shap_value": float, "impact": "positive"|"negative"}, ...]
            }
        """
        if not SHAP_AVAILABLE:
            return {"error": "shap library not installed. Run: pip install shap"}

        predict_fn = self._get_prediction_fn()

        # Get the prediction first
        probs      = predict_fn([text])[0]
        top_idx    = int(np.argmax(probs))
        prediction = self.classes_[top_idx]
        confidence = float(probs[top_idx])

        # Build SHAP explainer (text masker handles tokenization)
        masker   = shap.maskers.Text(self.tokenizer)
        explainer = shap.Explainer(predict_fn, masker, output_names=self.classes_)

        shap_values = explainer([text])

        # Extract SHAP values for the predicted class
        class_shap = shap_values.values[0, :, top_idx]
        words      = shap_values.data[0]

        # Sort by absolute impact
        word_shap_pairs = sorted(
            zip(words, class_shap),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_n_words]

        explanations = [
            {
                "word":       word,
                "shap_value": round(float(shap_val), 4),
                "impact":     "positive" if shap_val > 0 else "negative",
            }
            for word, shap_val in word_shap_pairs
            if word.strip()
        ]

        return {
            "text":         text,
            "prediction":   prediction,
            "confidence":   round(confidence, 4),
            "all_probs":    {cls: round(float(p), 4) for cls, p in zip(self.classes_, probs)},
            "explanations": explanations,
            "note":         f"Positive SHAP = word supports '{prediction}'; "
                            f"Negative SHAP = word argues against it.",
        }

    def print_explanation(self, result: dict) -> None:
        """Pretty-print the explanation to the terminal."""
        if "error" in result:
            print(f"[Error] {result['error']}")
            return

        print(f"\n{'─'*56}")
        print(f"  Text        : {result['text'][:60]}...")
        print(f"  Prediction  : {result['prediction'].upper()}")
        print(f"  Confidence  : {result['confidence']*100:.1f}%")
        print(f"\n  Key words driving the prediction:")
        for e in result["explanations"]:
            sign = "+" if e["impact"] == "positive" else "-"
            bar  = "█" * min(int(abs(e["shap_value"]) * 50), 20)
            print(f"  {sign} {e['word']:<18} {e['shap_value']:+.4f}  {bar}")
        print(f"{'─'*56}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate SHAP explanations for mental health predictions"
    )
    parser.add_argument("--model", required=True, help="Path to trained model directory")
    parser.add_argument("--text",  required=True, help="Text to explain")
    parser.add_argument("--top-words", type=int, default=10,
                        help="Number of top words to show (default: 10)")
    args = parser.parse_args()

    explainer = MentalHealthExplainer(args.model)
    result    = explainer.explain(args.text, top_n_words=args.top_words)
    explainer.print_explanation(result)


if __name__ == "__main__":
    main()
