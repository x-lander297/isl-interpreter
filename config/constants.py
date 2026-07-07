"""
config/constants.py
====================

Central configuration module for the Indian Sign Language (ISL)
Interpreter project.

This module defines static, immutable configuration values consumed by
other parts of the system (dataset preparation, model training,
real-time inference, and audio feedback). It intentionally contains
**no business logic, no file I/O, no data processing, and no model
training** -- it only declares configuration constants and lightweight
path/parameter groupings.

Configuration is organized into clearly delimited sections:

1. Dataset paths
2. Model artifact paths
3. Feature-engineering settings (MediaPipe landmarks -> feature vector)
4. Training settings (XGBoost hyperparameters, split ratio, seed)
5. Audio/speech feedback settings
6. Camera/capture settings (consumed by ``src.inference.camera``)
7. Future-extension settings (dynamic gestures, model versioning)

Paths are resolved relative to the project root (the parent directory
of this ``config`` package) using :mod:`pathlib`, so the project can be
run from any working directory.

Notes
-----
Only a module-level logger is configured here (attached to a
``NullHandler``, per standard library convention for importable
modules). No logging *calls* are made at import time beyond an optional
debug-level confirmation, since a pure configuration module has no
runtime behavior worth narrating beyond "constants were loaded."
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "STATIC_LANDMARKS_PATH",
    "LABELS_PATH",
    "COMBINED_DATASET_PATH",
    "MODEL_DIR",
    "XGBOOST_MODEL_PATH",
    "LABEL_ENCODER_PATH",
    "MEDIAPIPE_LANDMARK_COUNT",
    "COORDINATES_PER_LANDMARK",
    "SINGLE_HAND_VECTOR_SIZE",
    "NUM_HAND_SLOTS",
    "FEATURE_VECTOR_SIZE",
    "TEST_SPLIT_RATIO",
    "RANDOM_SEED",
    "DEFAULT_XGBOOST_PARAMS",
    "SPEECH_COOLDOWN_SECONDS",
    "MUTE_DEFAULT",
    "CAMERA_INDEX",
    "FRAME_WIDTH",
    "FRAME_HEIGHT",
    "MAX_READ_RETRIES",
    "PREDICTION_BUFFER_SIZE",
    "DYNAMIC_GESTURE_SEQUENCE_LENGTH",
    "DYNAMIC_GESTURE_FEATURE_VECTOR_SIZE",
    "DYNAMIC_GESTURES_ENABLED",
    "MODEL_VERSION",
    "MODEL_REGISTRY_DIR",
]

# ---------------------------------------------------------------------------
# Module-level logger (NullHandler: this is a library/config module, not an
# application entry point, so it must not configure handlers itself).
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


# =============================================================================
# 0. Project root resolution
# =============================================================================
# This file lives at <project_root>/config/constants.py, so the project
# root is two levels up from this file's resolved, absolute location.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent


# =============================================================================
# 1. Dataset paths
# =============================================================================
# Root directory for all dataset artifacts (raw and processed).
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"

# CSV/NPY file holding extracted static-hand-sign landmark feature rows
# (pre-flattening or post-flattening, depending on the extraction script),
# one row per captured sample.
STATIC_LANDMARKS_PATH: Final[Path] = DATA_DIR / "static" / "landmarks.csv"

# CSV/NPY file holding the ground-truth class labels aligned by row index
# with STATIC_LANDMARKS_PATH.
LABELS_PATH: Final[Path] = DATA_DIR / "static" / "labels.csv"

# Combined dataset (features + labels merged into a single file), used as
# the direct input to model training.
COMBINED_DATASET_PATH: Final[Path] = DATA_DIR / "processed" / "combined_dataset.csv"


# =============================================================================
# 2. Model artifact paths
# =============================================================================
# Root directory for all trained model artifacts.
MODEL_DIR: Final[Path] = PROJECT_ROOT / "models"

# Serialized, trained XGBoost classifier (e.g. saved via `booster.save_model`
# or `joblib.dump` on an XGBClassifier), consumed read-only by inference code.
XGBOOST_MODEL_PATH: Final[Path] = MODEL_DIR / "xgboost_isl_classifier.json"

# Serialized `sklearn.preprocessing.LabelEncoder` (or equivalent) mapping
# model output indices back to human-readable ISL class labels.
LABEL_ENCODER_PATH: Final[Path] = MODEL_DIR / "label_encoder.pkl"


# =============================================================================
# 3. Feature settings (MediaPipe landmarks -> feature vector)
# =============================================================================
# Number of hand landmarks produced by MediaPipe Hands per detected hand.
MEDIAPIPE_LANDMARK_COUNT: Final[int] = 21

# Coordinates captured per landmark (x, y, z).
COORDINATES_PER_LANDMARK: Final[int] = 3

# Flattened feature-vector size for a single detected hand
# (21 landmarks * 3 coordinates = 63).
SINGLE_HAND_VECTOR_SIZE: Final[int] = MEDIAPIPE_LANDMARK_COUNT * COORDINATES_PER_LANDMARK

# Number of hand "slots" the current feature representation reserves.
# The static ISL alphabet/number model is trained on a fixed two-slot
# layout: one real hand's landmarks plus one zero-padded slot for a
# second (currently unused) hand, preserving forward compatibility with
# two-handed signs without retraining the input layer.
NUM_HAND_SLOTS: Final[int] = 2

# Final input dimension consumed by the XGBoost classifier:
# 63 real landmark values + 63 zero-padding values = 126.
FEATURE_VECTOR_SIZE: Final[int] = SINGLE_HAND_VECTOR_SIZE * NUM_HAND_SLOTS


# =============================================================================
# 4. Training settings
# =============================================================================
# Proportion of the combined dataset held out for evaluation.
TEST_SPLIT_RATIO: Final[float] = 0.2

# Fixed seed for all stochastic operations (train/test split, XGBoost's
# internal subsampling, etc.) to ensure reproducible training runs.
RANDOM_SEED: Final[int] = 42

# Default hyperparameters for the XGBoost classifier used on the static
# ISL alphabet/number recognition task. These are sensible, non-placeholder
# defaults for a multi-class tabular classification problem of this size;
# they may be overridden by a hyperparameter search in the training script.
DEFAULT_XGBOOST_PARAMS: Final[dict[str, object]] = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.0,
    "min_child_weight": 1,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "tree_method": "hist",
}


# =============================================================================
# 5. Audio / speech feedback settings
# =============================================================================
# Minimum time, in seconds, that must elapse between two consecutive
# spoken (text-to-speech) announcements of a recognized sign, to avoid
# repeatedly re-announcing a stable/held gesture every frame.
SPEECH_COOLDOWN_SECONDS: Final[float] = 2.0

# Default state of the "mute" toggle for speech output when the
# application starts. False = audio feedback enabled by default.
MUTE_DEFAULT: Final[bool] = False


# =============================================================================
# 6. Camera / capture settings
# =============================================================================
# Index of the default webcam device passed to the video capture backend.
CAMERA_INDEX: Final[int] = 0

# Target frame dimensions (pixels) that captured frames are resized to
# before landmark extraction and inference.
FRAME_WIDTH: Final[int] = 640
FRAME_HEIGHT: Final[int] = 480

# Number of consecutive failed frame-read attempts tolerated before the
# capture layer raises an error.
MAX_READ_RETRIES: Final[int] = 3

# Number of recent raw predictions retained for majority-vote temporal
# smoothing during real-time inference.
PREDICTION_BUFFER_SIZE: Final[int] = 5


# =============================================================================
# 7. Future extension settings
# =============================================================================
# --- Dynamic gesture recognition (planned) ---------------------------------
# Number of consecutive frames that will comprise a single dynamic-gesture
# sequence sample once temporal (video-based) gesture recognition is added.
DYNAMIC_GESTURE_SEQUENCE_LENGTH: Final[int] = 30

# Per-frame feature-vector size for dynamic gestures. Reuses the same
# 126-dimensional static representation as the base per-frame feature,
# so a dynamic-gesture model's input is naturally
# (DYNAMIC_GESTURE_SEQUENCE_LENGTH, DYNAMIC_GESTURE_FEATURE_VECTOR_SIZE).
DYNAMIC_GESTURE_FEATURE_VECTOR_SIZE: Final[int] = FEATURE_VECTOR_SIZE

# Feature flag gating dynamic-gesture inference paths in the wider
# application. Disabled by default since only static recognition is
# implemented today; future modules can flip this on once trained.
DYNAMIC_GESTURES_ENABLED: Final[bool] = False

# --- Multiple ML model support / versioning --------------------------------
# Directory reserved for versioned model artifacts (e.g.
# MODEL_REGISTRY_DIR / "v1" / "xgboost_isl_classifier.json"), enabling
# future support for multiple concurrently available model versions
# without changing the single "current" paths defined in section 2.
MODEL_REGISTRY_DIR: Final[Path] = MODEL_DIR / "registry"

# Identifier for the currently active model version. Used to select a
# subdirectory under MODEL_REGISTRY_DIR once model versioning is wired
# into the training/inference pipeline.
MODEL_VERSION: Final[str] = "v1"


logger.debug(
    "config.constants loaded: PROJECT_ROOT=%s, FEATURE_VECTOR_SIZE=%d, "
    "MODEL_VERSION=%s.",
    PROJECT_ROOT,
    FEATURE_VECTOR_SIZE,
    MODEL_VERSION,
)