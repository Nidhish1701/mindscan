"""
Ensemble inference — combines Transformer + XGBoost predictions.

Weighted voting:
    - Transformer (DistilBERT): 60% weight
    - XGBoost (TF-IDF):        40% weight

Usage:
    from src.inference.ensemble import EnsemblePredictor
    ens = EnsemblePredictor("models/distilbert/best_model", "models/xgb_model.pkl")
    results = ens.predict(["I feel hopeless"])
"""

import os
import pickle
import numpy as np
import torch
from typing import List, Dict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EnsemblePredictor:
    """Weighted ensemble of transformer + XGBoost."""

    def __init__(
        self,
        transformer_dir: str,
        xgb_path: str,
        transformer_weight: float = 0.6,
    ):
        from src.inference.predict import load_model

        print("[Ensemble] Loading transformer ...")
        self.model, self.tokenizer, self.label_encoder = load_model(transformer_dir)

        print("[Ensemble] Loading XGBoost ...")
        with open(xgb_path, 'rb') as f:
            xgb_bundle = pickle.load(f)
        self.xgb = xgb_bundle['xgb']
        self.tfidf = xgb_bundle['tfidf']

        self.transformer_weight = transformer_weight
        self.xgb_weight = 1.0 - transformer_weight
        self.classes_ = list(self.label_encoder.classes_)
        print(f"[Ensemble] Ready. Weights: transformer={transformer_weight:.0%}, xgb={self.xgb_weight:.0%}")

    def predict(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        # Transformer probabilities
        trans_probs = self._transformer_probs(texts, batch_size)

        # XGBoost probabilities
        xgb_vec = self.tfidf.transform(texts)
        xgb_probs = self.xgb.predict_proba(xgb_vec)

        # Align XGBoost classes
        xgb_classes = [str(c) for c in self.xgb.classes_]
        aligned_xgb = np.zeros((len(texts), len(self.classes_)))
        for i, cls in enumerate(self.classes_):
            if str(cls) in xgb_classes:
                idx = xgb_classes.index(str(cls))
                aligned_xgb[:, i] = xgb_probs[:, idx]

        # Weighted average
        ensemble_probs = self.transformer_weight * trans_probs + self.xgb_weight * aligned_xgb

        results = []
        for i, text in enumerate(texts):
            top_idx = int(np.argmax(ensemble_probs[i]))
            results.append({
                'text': text,
                'prediction': str(self.classes_[top_idx]),
                'confidence': float(ensemble_probs[i][top_idx]),
                'probabilities': {
                    str(self.classes_[j]): float(ensemble_probs[i][j])
                    for j in range(len(self.classes_))
                },
                'source': 'ensemble',
            })

        return results

    def _transformer_probs(self, texts, batch_size):
        self.model.eval()
        all_probs = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = self.tokenizer(
                batch, truncation=True, padding=True,
                max_length=128, return_tensors='pt'
            )
            with torch.no_grad():
                logits = self.model(
                    input_ids=enc['input_ids'].to(DEVICE),
                    attention_mask=enc['attention_mask'].to(DEVICE),
                ).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
        return np.vstack(all_probs)
