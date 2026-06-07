# Task 6 - ASL Letter Classification
# Loads a trained classifier and predicts the ASL letter (A-Z)
# from the 63-dimensional MediaPipe landmark feature vector.

import joblib
import numpy as np
from pathlib import Path

_MODEL_PATH = Path("model/classifier.pkl")
_model = None


def _get_model():
    global _model
    # Load and cache the model on first call; return None if file is missing
    if _model is None:
        if not _MODEL_PATH.exists():
            return None
        _model = joblib.load(_MODEL_PATH)
    return _model


def predict_letter(landmarks_63):
    model = _get_model()
    if model is None:
        return None

    # Reshape to (1, 63) for sklearn and run inference
    features = np.array(landmarks_63, dtype=np.float32).reshape(1, -1)
    letter = model.predict(features)[0]

    # Return the top class probability as the confidence score
    confidence = float(model.predict_proba(features).max())

    return letter, confidence
