import pickle
import numpy as np
from config.constants import MODELS_DIR
from xgboost import XGBClassifier  # Required for pickle deserialization

_MODEL = None

def load_model(model_path=None):
    global _MODEL
    if model_path is None:
        model_path = f"{MODELS_DIR}/model.pkl"
    with open(model_path, 'rb') as f:
        _MODEL = pickle.load(f)
    return _MODEL

def predict(features):
    """
    features: numpy array of shape (63,) or (126,).
    Returns: predicted label index (int).
    """
    global _MODEL
    if _MODEL is None:
        load_model()
    # Ensure 2D input for XGBoost
    if features.ndim == 1:
        features = features.reshape(1, -1)
    return int(_MODEL.predict(features)[0])