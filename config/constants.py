"""
Module: config.constants

Purpose:
    Central, read-only configuration for the ISL Interpreter project.
    This is the SINGLE SOURCE OF TRUTH for all constants.

    This is a FULLY MERGED version:
        - Uses pathlib.Path for all paths (compatible with os.path functions).
        - Includes every constant name from P1's and P3's files.
        - Provides aliases so no one needs to change their code.

Owner: P1 + P3 (shared ownership)
Dependencies: pathlib (standard library)
"""

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# 0. Root directory (as Path)
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

# Alias for P3's files
PROJECT_ROOT: Path = ROOT_DIR


# ---------------------------------------------------------------------------
# 1. Data directories (all as Path objects)
# ---------------------------------------------------------------------------
DATA_DIR: Path = ROOT_DIR / "data"
DATA_RAW_STATIC: Path = DATA_DIR / "raw" / "static"
DATA_RAW_DYNAMIC: Path = DATA_DIR / "raw" / "dynamic"
DATA_PROCESSED: Path = DATA_DIR / "processed"


# ---------------------------------------------------------------------------
# 2. Processed file paths (used by loader, extractor, merge_data, train)
# ---------------------------------------------------------------------------
STATIC_LANDMARKS_PATH: Path = DATA_PROCESSED / "static_landmarks.npy"
STATIC_LABELS_PATH: Path = DATA_PROCESSED / "static_labels.npy"
X_COMBINED_PATH: Path = DATA_PROCESSED / "X_combined.npy"
Y_COMBINED_PATH: Path = DATA_PROCESSED / "y_combined.npy"

# Aliases for P3's files
COMBINED_DATASET_PATH: Path = X_COMBINED_PATH
LABELS_PATH: Path = STATIC_LABELS_PATH


# ---------------------------------------------------------------------------
# 3. Model directories and paths
# ---------------------------------------------------------------------------
MODELS_DIR: Path = ROOT_DIR / "models"
XGBOOST_MODEL_PATH: Path = MODELS_DIR / "xgb_model.pkl"
LABEL_ENCODER_PATH: Path = MODELS_DIR / "label_encoder.pkl"

# For P3's future versioning
MODEL_REGISTRY_DIR: Path = MODELS_DIR / "registry"
MODEL_VERSION: str = "v1"


# ---------------------------------------------------------------------------
# 4. Landmark / feature dimensions
# ---------------------------------------------------------------------------
NUM_HAND_LANDMARKS: int = 21
COORDS_PER_LANDMARK: int = 3

# P3's alias (same value)
MEDIAPIPE_LANDMARK_COUNT: int = NUM_HAND_LANDMARKS

# Alias for extractor (which uses COORDINATES_PER_LANDMARK)
COORDINATES_PER_LANDMARK: int = COORDS_PER_LANDMARK

STATIC_FEATURE_DIM: int = NUM_HAND_LANDMARKS * COORDS_PER_LANDMARK  # 63
DYNAMIC_FEATURE_DIM: int = STATIC_FEATURE_DIM * 2                   # 126
INPUT_DIM: int = DYNAMIC_FEATURE_DIM                                # 126

# P3's alias
FEATURE_VECTOR_SIZE: int = INPUT_DIM


# ---------------------------------------------------------------------------
# 5. Class labels (A–Z, 1–9)
# ---------------------------------------------------------------------------
STATIC_LABELS: list = [chr(ord('A') + i) for i in range(26)] + [str(i) for i in range(1, 10)]
NUM_STATIC_CLASSES: int = len(STATIC_LABELS)   # 35
TOTAL_CLASSES: int = NUM_STATIC_CLASSES

LABEL_INDEX_TO_NAME: dict = {i: name for i, name in enumerate(STATIC_LABELS)}
LABEL_NAME_TO_INDEX: dict = {name: i for i, name in LABEL_INDEX_TO_NAME.items()}


# ---------------------------------------------------------------------------
# 6. XGBoost hyperparameters
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

# P3's alias
DEFAULT_XGBOOST_PARAMS: dict = XGB_PARAMS


# ---------------------------------------------------------------------------
# 7. Training settings
# ---------------------------------------------------------------------------
TEST_SPLIT_RATIO: float = 0.2
RANDOM_SEED: int = 42


# ---------------------------------------------------------------------------
# 8. Real-time inference tuning
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD: float = 0.7
COOLDOWN_FRAMES: int = 30
BUFFER_SIZE: int = 5                 # smoothing buffer size

# P3's alias
PREDICTION_BUFFER_SIZE: int = BUFFER_SIZE


# ---------------------------------------------------------------------------
# 9. Audio feedback settings
# ---------------------------------------------------------------------------
SPEECH_COOLDOWN_SECONDS: float = 2.0
MUTE_DEFAULT: bool = False


# ---------------------------------------------------------------------------
# 10. Camera settings
# ---------------------------------------------------------------------------
CAMERA_INDEX: int = 0
FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 480
MAX_READ_RETRIES: int = 3


# ---------------------------------------------------------------------------
# 11. Future dynamic gesture placeholders
# ---------------------------------------------------------------------------
DYNAMIC_GESTURE_SEQUENCE_LENGTH: int = 30
DYNAMIC_GESTURE_FEATURE_VECTOR_SIZE: int = FEATURE_VECTOR_SIZE
DYNAMIC_GESTURES_ENABLED: bool = False


# ---------------------------------------------------------------------------
# 12. Backward compatibility for os.path users
# ---------------------------------------------------------------------------
# Convert Path objects to strings for files that still use os.path.join.
# These are provided as an escape hatch, but all new code should use Path.
ROOT_DIR_STR: str = str(ROOT_DIR)
DATA_DIR_STR: str = str(DATA_DIR)
DATA_PROCESSED_STR: str = str(DATA_PROCESSED)
MODELS_DIR_STR: str = str(MODELS_DIR)