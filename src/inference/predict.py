import os
import pickle
import numpy as np

def load_model(model_dir: str):
    """
    Load the trained XGBoost model for 6-class mental health detection.
    """
    xgb_path = os.path.join("models", "xgb_model.pkl")
    print(f"[Predict] Loading High-Performance XGBoost model from {xgb_path} ...")
    
    if not os.path.exists(xgb_path):
        raise FileNotFoundError(f"Model file not found at {xgb_path}. Run training first.")
        
    with open(xgb_path, "rb") as f:
        data = pickle.load(f)
        
    xgb = data['xgb']
    tfidf = data['tfidf']
    label_encoder = data['label_encoder']

    print(f"[Predict] Model ready. Supporting 6 mental health classes.")
    return xgb, tfidf, label_encoder

from src.chatbot.mental_health_bot import _is_normal_state

def predict(texts: list, model, tokenizer, label_encoder, batch_size: int = 32, max_length: int = 128) -> list:
    """
    Mental Health classification engine (multi-category).
    """
    xgb = model
    tfidf = tokenizer # the vectorizer acts as our tokenizer
    
    # Vectorize and predict
    X_vec = tfidf.transform(texts)
    probs = xgb.predict_proba(X_vec)
    
    all_results = []
    classes = list(label_encoder.classes_)
    
    for i, text in enumerate(texts):
        # Intercept normal/positive states
        if _is_normal_state(text.lower()):
            probs_dict = {str(classes[j]): 0.0 for j in range(len(classes))}
            probs_dict['Normal'] = 1.0
            all_results.append({
                'text':          text,
                'prediction':    'Normal',
                'confidence':    1.0,
                'probabilities': probs_dict,
            })
            continue

        top_idx = int(np.argmax(probs[i]))
        all_results.append({
            'text':          text,
            'prediction':    str(classes[top_idx]),
            'confidence':    float(probs[i][top_idx]),
            'probabilities': {str(classes[j]): float(probs[i][j]) for j in range(len(classes))},
        })

    return all_results