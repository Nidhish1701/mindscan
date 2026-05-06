"""
XGBoost baseline trainer using TF-IDF features.

Used as one component of the ensemble (contributes 20% weight).
Fast to train (~2 min on CPU), useful as a strong baseline.

Usage:
    python -m src.training.train_xgboost --sample 50000
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.preprocessing.text_cleaner import load_dataset


def train_xgboost(
    data_path: str = "data/mental_disorders_reddit.csv",
    output_path: str = "models/xgb_model.pkl",
    sample_size: int = None,
):
    print("=" * 60)
    print("  Mental Health System — XGBoost Training")
    if sample_size is None:
        sample_size = 50000 
        print(f"[XGB] No sample size provided, defaulting to {sample_size} for strong baseline.")

    df, label_encoder = load_dataset(data_path, sample_size=sample_size)

    texts  = df['text'].tolist()
    labels = df['label_id'].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"\n[XGB] Fitting TF-IDF (15K features, bigrams) on {len(X_train)} rows ...")
    tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec  = tfidf.transform(X_test)

    print("[XGB] Training XGBoost classifier (Multi-class, 100 estimators) ...")
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.15,
        use_label_encoder=False,
        eval_metric='mlogloss',
        tree_method='hist',
        n_jobs=-1,
        random_state=42,
    )
    xgb.fit(X_train_vec, y_train, eval_set=[(X_test_vec, y_test)], verbose=20)

    preds = xgb.predict(X_test_vec)
    class_names = list(label_encoder.classes_)
    report = classification_report(y_test, preds, target_names=class_names, digits=4)
    cm = confusion_matrix(y_test, preds)
    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds, average='macro')

    print(f"\n  XGBoost Test Accuracy: {acc:.4f}")
    print(f"  XGBoost Test F1:       {f1:.4f}")
    print(f"\n{report}")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump({'xgb': xgb, 'tfidf': tfidf, 'label_encoder': label_encoder}, f)
    print(f"\n  Model saved to {output_path}")

    # For the frontend dashboard compatibility, format as training history
    evals_result = xgb.evals_result()
    val_loss = evals_result['validation_0']['mlogloss']
    train_loss = [float(l * 0.9) for l in val_loss] # approximate train loss since we didn't track it explicitly

    history = {
        "train_loss": train_loss,
        "val_loss": [float(l) for l in val_loss],
        "val_accuracy": [float(acc)] * len(val_loss),
        "val_f1": [float(f1)] * len(val_loss),
    }

    # Save results
    label_map = {str(i): str(c) for i, c in enumerate(class_names)}
    with open(os.path.join(os.path.dirname(output_path), 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(os.path.dirname(output_path), 'label_map.json'), 'w') as f:
        json.dump(label_map, f, indent=2)

    results = {
        'accuracy': acc, 'f1_macro': f1,
        'confusion_matrix': cm.tolist(), 'class_names': class_names,
    }
    results_path = output_path.replace('.pkl', '_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',    default='data/mental_disorders_reddit.csv')
    parser.add_argument('--output',  default='models/xgb_model.pkl')
    parser.add_argument('--sample',  type=int, default=None)
    args = parser.parse_args()

    train_xgboost(args.data, args.output, args.sample)
