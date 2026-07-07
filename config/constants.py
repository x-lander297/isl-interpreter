"""
Module: config.constants

Purpose:
    Central, read-only configuration for the ISL Interpreter project.
    Every other module (data loading, feature extraction, training,
    inference) must import its constants from here.

    This is a MERGED version:
        - P1's simple os.path style preserved for compatibility.
        - P3's additional constants (audio, camera, smoothing, dynamic)
          included so their files work without modification.

Owner: P1 + P3 (shared ownership)
Dependencies: None (standard library only)
"""

import os


# ---------------------------------------------------------------------------
# 0. Root directory
# ---------------------------------------------------------------------------
ROOT_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# 1. Landmark / feature dimensions
# ---------------------------------------------------------------------------
NUM_HAND_LANDMARKS: int = 21
COORDS_PER_LANDMARK: int = 3
STATIC_FEATURE_DIM: int = NUM_HAND_LANDMARKS * COORDS_PER_LANDMARK  # 63
DYNAMIC_FEATURE_DIM: int = STATIC_FEATURE_DIM * 2                   # 126
INPUT_DIM: int = DYNAMIC_FEATURE_DIM                                # 126

# Alias for P3's files (same as INPUT_DIM)
FEATURE_VECTOR_SIZE: int = INPUT_DIM


# ---------------------------------------------------------------------------
# 2. Class labels
# ---------------------------------------------------------------------------
STATIC_LABELS: list = [chr(ord('A') + i) for i in range(26)] + [str(i) for i in range(1, 10)]
NUM_STATIC_CLASSES: int = len(STATIC_LABELS)   # 35
TOTAL_CLASSES: int = NUM_STATIC_CLASSES

LABEL_INDEX_TO_NAME: dict = {i: name for i, name in enumerate(STATIC_LABELS)}
LABEL_NAME_TO_INDEX: dict = {name: i for i, name in LABEL_INDEX_TO_NAME.items()}


# ---------------------------------------------------------------------------
# 3. XGBoost hyperparameters (P1's version, used by train.py)
# ---------------------------------------------------------------------------
XGB_PARAMS: dict = {
    'n_estimators': 150,
    'max_depth': 8,
    'learning_rate': 0.1,
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': 'mlogloss',
    'objective': 'multi:softprob',
    'num_class': TOTAL_CLASSES,
    'n_jobs': -1,
}

# P3's DEFAULT_XGBOOST_PARAMS alias for compatibility
DEFAULT_XGBOOST_PARAMS: dict = XGB_PARAMS


# ---------------------------------------------------------------------------
# 4. File system paths
# ---------------------------------------------------------------------------
DATA_RAW_STATIC: str = os.path.join(ROOT_DIR, 'data', 'raw', 'static')
DATA_RAW_DYNAMIC: str = os.path.join(ROOT_DIR, 'data', 'raw', 'dynamic')
DATA_PROCESSED: str = os.path.join(ROOT_DIR, 'data', 'processed')
MODELS_DIR: str = os.path.join(ROOT_DIR, 'models')

# Processed files (used by loader, extractor, static_pipeline, merge_data)
STATIC_LANDMARKS_PATH: str = os.path.join(DATA_PROCESSED, 'static_landmarks.npy')
STATIC_LABELS_PATH: str = os.path.join(DATA_PROCESSED, 'static_labels.npy')
X_COMBINED_PATH: str = os.path.join(DATA_PROCESSED, 'X_combined.npy')
Y_COMBINED_PATH: str = os.path.join(DATA_PROCESSED, 'y_combined.npy')

# Aliases for P3's files (they expect these names)
COMBINED_DATASET_PATH: str = X_COMBINED_PATH   # for merge_data.py
LABELS_PATH: str = STATIC_LABELS_PATH          # for loader

# Model paths (for train.py and predict.py)
XGBOOST_MODEL_PATH: str = os.path.join(MODELS_DIR, 'xgb_model.pkl')
LABEL_ENCODER_PATH: str = os.path.join(MODELS_DIR, 'label_encoder.pkl')


# ---------------------------------------------------------------------------
# 5. Training settings
# ---------------------------------------------------------------------------
TEST_SPLIT_RATIO: float = 0.2
RANDOM_SEED: int = 42


# ---------------------------------------------------------------------------
# 6. Audio / speech feedback settings (for P3's audio.py)
# ---------------------------------------------------------------------------
SPEECH_COOLDOWN_SECONDS: float = 2.0
MUTE_DEFAULT: bool = False


# ---------------------------------------------------------------------------
# 7. Camera / capture settings (for P2's camera.py and processor.py)
# ---------------------------------------------------------------------------
CAMERA_INDEX: int = 0
FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 480
MAX_READ_RETRIES: int = 3
PREDICTION_BUFFER_SIZE: int = 5   # P3's smoothing buffer size

# Aliases for P3's files (same as PREDICTION_BUFFER_SIZE)
BUFFER_SIZE: int = PREDICTION_BUFFER_SIZE


# ---------------------------------------------------------------------------
# 8. Real-time inference tuning
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD: float = 0.7
COOLDOWN_FRAMES: int = 30


# ---------------------------------------------------------------------------
# 9. Future dynamic gesture settings (placeholder)
# ---------------------------------------------------------------------------
DYNAMIC_GESTURE_SEQUENCE_LENGTH: int = 30
DYNAMIC_GESTURE_FEATURE_VECTOR_SIZE: int = FEATURE_VECTOR_SIZE
DYNAMIC_GESTURES_ENABLED: bool = False
MODEL_VERSION: str = "v1"
MODEL_REGISTRY_DIR: str = os.path.join(MODELS_DIR, 'registry')