#!/usr/bin/env python3
"""
src/models/train.py
====================

Training pipeline for the Indian Sign Language (ISL) Interpreter's
static-gesture XGBoost classifier.

This module loads the previously combined, padded feature/label dataset
(``X_combined.npy`` / ``y_combined.npy``, produced by
``scripts/merge_data.py``), validates it, encodes string labels if
necessary, performs a stratified train/test split, trains an
``xgboost.XGBClassifier`` with overfitting-resistant defaults, evaluates
it with standard multiclass metrics, and persists the trained model and
label encoder to disk.

This module performs **no webcam, MediaPipe, inference, audio, or raw
data-collection logic** -- it strictly consumes an already-prepared
combined dataset and produces trained-model artifacts.

Usage
-----
    python -m src.models.train
    python src/models/train.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from config import constants

__all__ = [
    "TrainingError",
    "load_dataset",
    "validate_dataset",
    "encode_labels",
    "split_dataset",
    "train_model",
    "evaluate_model",
    "save_artifacts",
    "main",
]

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure a basic, human-readable logging setup for CLI execution.

    Only installs handlers if the root logger has none yet, so importing
    this module (e.g. from a test) does not clobber a host application's
    logging configuration.
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )


# ---------------------------------------------------------------------------
# Configuration resolution (defensive against config/constants.py not
# defining every combined-dataset / model-artifact attribute name)
# ---------------------------------------------------------------------------
_DEFAULT_PROCESSED_DIR: Final[Path] = constants.DATA_DIR / "processed"

X_COMBINED_PATH: Final[Path] = Path(
    getattr(constants, "X_COMBINED_PATH", _DEFAULT_PROCESSED_DIR / "X_combined.npy")
)
Y_COMBINED_PATH: Final[Path] = Path(
    getattr(constants, "Y_COMBINED_PATH", _DEFAULT_PROCESSED_DIR / "y_combined.npy")
)

# The task explicitly requires models/xgb_model.pkl via joblib, which takes
# precedence over the .json-oriented XGBOOST_MODEL_PATH already defined in
# config.constants (a different artifact format for a different use case).
XGB_MODEL_PKL_PATH: Final[Path] = Path(
    getattr(constants, "XGB_MODEL_PKL_PATH", constants.MODEL_DIR / "xgb_model.pkl")
)
LABEL_ENCODER_PATH: Final[Path] = Path(constants.LABEL_ENCODER_PATH)

EXPECTED_FEATURE_DIM: Final[int] = int(constants.FEATURE_VECTOR_SIZE)
TEST_SPLIT_RATIO: Final[float] = float(constants.TEST_SPLIT_RATIO)
RANDOM_SEED: Final[int] = int(constants.RANDOM_SEED)
BASE_XGB_PARAMS: Final[dict[str, Any]] = dict(constants.DEFAULT_XGBOOST_PARAMS)

# Rounds of no improvement on the held-out set before stopping early, as a
# light guard against overfitting during boosting.
EARLY_STOPPING_ROUNDS: Final[int] = 20


class TrainingError(RuntimeError):
    """Raised for any unrecoverable failure during dataset loading,
    validation, label encoding, splitting, training, evaluation, or
    artifact persistence.
    """


def load_dataset(x_path: Path, y_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the combined feature matrix and label array from disk.

    Parameters
    ----------
    x_path:
        Path to ``X_combined.npy``.
    y_path:
        Path to ``y_combined.npy``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The ``(X, y)`` arrays, loaded as-is from disk.

    Raises
    ------
    TrainingError
        If either file does not exist, is not a file, or cannot be
        parsed as a NumPy array.
    """
    for label, path in (("features", x_path), ("labels", y_path)):
        if not path.exists():
            logger.error("Required %s file does not exist: %s", label, path)
            raise TrainingError(f"Required {label} file does not exist: {path}")
        if not path.is_file():
            logger.error("Expected a file for %s but found: %s", label, path)
            raise TrainingError(f"Expected a file for {label} but found: {path}")

    try:
        logger.info("Loading feature matrix from: %s", x_path)
        X = np.load(x_path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load features file %s: %s", x_path, exc)
        raise TrainingError(f"Failed to load features file: {x_path}") from exc

    try:
        logger.info("Loading label array from: %s", y_path)
        y = np.load(y_path, allow_pickle=True)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load labels file %s: %s", y_path, exc)
        raise TrainingError(f"Failed to load labels file: {y_path}") from exc

    logger.info(
        "Loaded raw dataset: X.shape=%s (dtype=%s), y.shape=%s (dtype=%s).",
        X.shape,
        X.dtype,
        y.shape,
        y.dtype,
    )
    return X, y


def validate_dataset(
    X: np.ndarray, y: np.ndarray, expected_feature_dim: int = EXPECTED_FEATURE_DIM
) -> None:
    """Validate structural integrity of the loaded feature/label arrays.

    Parameters
    ----------
    X:
        The feature matrix.
    y:
        The label array.
    expected_feature_dim:
        The required number of feature columns (defaults to
        ``config.constants.FEATURE_VECTOR_SIZE``).

    Raises
    ------
    TrainingError
        If either array is empty, ``X`` is not 2-D, ``y`` is not 1-D,
        the sample counts differ, or ``X``'s feature dimension does not
        equal ``expected_feature_dim``.
    """
    if X.size == 0:
        logger.error("Feature matrix is empty.")
        raise TrainingError("Feature matrix is empty; nothing to train on.")
    if y.size == 0:
        logger.error("Label array is empty.")
        raise TrainingError("Label array is empty; nothing to train on.")

    if X.ndim != 2:
        logger.error("Feature matrix has unexpected ndim=%d.", X.ndim)
        raise TrainingError(
            f"Feature matrix must be 2-D (num_samples, feature_dim); "
            f"got ndim={X.ndim} with shape={X.shape}."
        )
    if y.ndim != 1:
        logger.error("Label array has unexpected ndim=%d.", y.ndim)
        raise TrainingError(
            f"Label array must be 1-D (num_samples,); "
            f"got ndim={y.ndim} with shape={y.shape}."
        )

    if X.shape[0] != y.shape[0]:
        logger.error(
            "Sample count mismatch: X has %d samples, y has %d samples.",
            X.shape[0],
            y.shape[0],
        )
        raise TrainingError(
            f"Number of samples does not match between features "
            f"({X.shape[0]}) and labels ({y.shape[0]})."
        )

    if X.shape[1] != expected_feature_dim:
        logger.error(
            "Feature dimension mismatch: got %d, expected %d.",
            X.shape[1],
            expected_feature_dim,
        )
        raise TrainingError(
            f"Feature dimension ({X.shape[1]}) does not match the expected "
            f"model input dimension ({expected_feature_dim})."
        )

    logger.info(
        "Validation passed: %d samples, feature_dim=%d, %d unique labels.",
        X.shape[0],
        X.shape[1],
        len(np.unique(y)),
    )


def encode_labels(y: np.ndarray) -> tuple[np.ndarray, LabelEncoder | None]:
    """Encode labels to contiguous integers if they are not already numeric.

    Parameters
    ----------
    y:
        The raw label array, either string/object-typed or
        numeric-typed.

    Returns
    -------
    tuple[numpy.ndarray, Optional[LabelEncoder]]
        A ``(y_encoded, encoder)`` pair. ``y_encoded`` is an
        ``int64`` array of contiguous, 0-indexed class labels suitable
        for XGBoost's multiclass objective. ``encoder`` is the fitted
        :class:`~sklearn.preprocessing.LabelEncoder` if one was used, or
        ``None`` if the input labels were already numeric.

    Raises
    ------
    TrainingError
        If numeric labels are provided but are not whole numbers (and
        therefore cannot be safely interpreted as class indices).
    """
    if np.issubdtype(y.dtype, np.number):
        logger.info("Labels are numeric (dtype=%s); using directly without encoding.", y.dtype)
        if np.issubdtype(y.dtype, np.floating):
            rounded = np.round(y)
            if not np.allclose(y, rounded):
                logger.error("Numeric labels contain non-integer values.")
                raise TrainingError(
                    "Numeric label array contains non-integer values; "
                    "XGBoost's multiclass objective requires integer class "
                    "indices."
                )
            y = rounded.astype(np.int64)
        else:
            y = y.astype(np.int64)
        return y, None

    logger.info("Labels are non-numeric (dtype=%s); fitting LabelEncoder.", y.dtype)
    try:
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y).astype(np.int64)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to fit LabelEncoder on labels: %s", exc)
        raise TrainingError("Failed to fit LabelEncoder on string labels.") from exc

    logger.info(
        "LabelEncoder fitted: %d classes -> %s",
        len(encoder.classes_),
        list(encoder.classes_),
    )
    return y_encoded, encoder


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = TEST_SPLIT_RATIO,
    random_seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the dataset into training and test sets.

    Attempts a stratified split (preserving class proportions), falling
    back to a plain random split if stratification is not possible
    (e.g. a class has fewer than 2 samples).

    Parameters
    ----------
    X:
        Feature matrix.
    y:
        Encoded, integer-typed label array.
    test_size:
        Fraction of samples held out for testing.
    random_seed:
        Seed for reproducible splitting.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        ``(X_train, X_test, y_train, y_test)``.

    Raises
    ------
    TrainingError
        If the split otherwise fails unexpectedly.
    """
    _, class_counts = np.unique(y, return_counts=True)
    can_stratify = bool(np.all(class_counts >= 2))

    if not can_stratify:
        logger.warning(
            "At least one class has fewer than 2 samples; falling back to "
            "a non-stratified train/test split."
        )

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_seed,
            stratify=y if can_stratify else None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Train/test split failed: %s", exc)
        raise TrainingError("Failed to split dataset into train/test sets.") from exc

    logger.info(
        "Split dataset: X_train=%s, X_test=%s, y_train=%s, y_test=%s "
        "(test_size=%.2f, stratified=%s).",
        X_train.shape,
        X_test.shape,
        y_train.shape,
        y_test.shape,
        test_size,
        can_stratify,
    )
    return X_train, X_test, y_train, y_test


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    base_params: dict[str, Any] | None = None,
) -> XGBClassifier:
    """Train an XGBoost classifier with overfitting-resistant defaults.

    Parameters
    ----------
    X_train, y_train:
        Training features and labels.
    X_test, y_test:
        Held-out features and labels, used as an early-stopping
        evaluation set to guard against overfitting.
    num_classes:
        Number of distinct target classes.
    base_params:
        Base hyperparameters (defaults to
        ``config.constants.DEFAULT_XGBOOST_PARAMS``); ``num_class`` and
        early-stopping configuration are added/overridden as needed.

    Returns
    -------
    XGBClassifier
        The fitted classifier.

    Raises
    ------
    TrainingError
        If model construction or fitting fails.
    """
    params = dict(base_params if base_params is not None else BASE_XGB_PARAMS)
    params["num_class"] = num_classes
    params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS

    logger.info("Training XGBoost classifier with parameters: %s", params)

    try:
        model = XGBClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Model training failed: %s", exc)
        raise TrainingError("XGBoost model training failed.") from exc

    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None:
        logger.info(
            "Training complete. Best iteration (early stopping): %d.", best_iteration
        )
    else:
        logger.info("Training complete.")

    return model


def evaluate_model(
    model: XGBClassifier, X_test: np.ndarray, y_test: np.ndarray
) -> dict[str, Any]:
    """Evaluate a trained classifier on a held-out test set.

    Computes accuracy, weighted precision/recall/F1, a full per-class
    classification report, and a confusion matrix.

    Parameters
    ----------
    model:
        The fitted classifier.
    X_test, y_test:
        Held-out features and labels.

    Returns
    -------
    dict[str, Any]
        A dictionary with keys: ``"accuracy"``, ``"precision"``,
        ``"recall"``, ``"f1_score"``, ``"confusion_matrix"``, and
        ``"classification_report"``.

    Raises
    ------
    TrainingError
        If prediction or metric computation fails.
    """
    try:
        y_pred = model.predict(X_test)
    except Exception as exc:  # noqa: BLE001
        logger.error("Prediction on test set failed: %s", exc)
        raise TrainingError("Failed to generate predictions on the test set.") from exc

    try:
        metrics: dict[str, Any] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(
                precision_score(y_test, y_pred, average="weighted", zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, y_pred, average="weighted", zero_division=0)
            ),
            "f1_score": float(
                f1_score(y_test, y_pred, average="weighted", zero_division=0)
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(
                y_test, y_pred, zero_division=0
            ),
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("Metric computation failed: %s", exc)
        raise TrainingError("Failed to compute evaluation metrics.") from exc

    logger.info(
        "Evaluation results: accuracy=%.4f, precision=%.4f, recall=%.4f, f1=%.4f.",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
    )
    logger.info("Confusion matrix:\n%s", metrics["confusion_matrix"])
    logger.info("Per-class classification report:\n%s", metrics["classification_report"])

    return metrics


def save_artifacts(
    model: XGBClassifier,
    encoder: LabelEncoder | None,
    model_path: Path,
    encoder_path: Path,
) -> None:
    """Persist the trained model and (optionally) the label encoder.

    Parameters
    ----------
    model:
        The fitted classifier to persist.
    encoder:
        The fitted label encoder, or ``None`` if labels were already
        numeric (in which case no encoder file is written).
    model_path:
        Destination path for the serialized model (``.pkl``, via
        joblib).
    encoder_path:
        Destination path for the serialized label encoder (``.pkl``,
        via joblib).

    Raises
    ------
    TrainingError
        If the output directories cannot be created, or artifacts
        cannot be written to disk.
    """
    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error("Failed to create model output directory: %s", exc)
        raise TrainingError("Failed to create model output directory.") from exc

    try:
        joblib.dump(model, model_path)
        logger.info("Saved trained model to: %s", model_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to save model to %s: %s", model_path, exc)
        raise TrainingError(f"Failed to save trained model to: {model_path}") from exc

    if encoder is not None:
        try:
            encoder_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(encoder, encoder_path)
            logger.info("Saved label encoder to: %s", encoder_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to save label encoder to %s: %s", encoder_path, exc)
            raise TrainingError(
                f"Failed to save label encoder to: {encoder_path}"
            ) from exc
    else:
        logger.info(
            "Labels were already numeric; no label encoder was fitted, so "
            "none is saved."
        )


def main() -> int:
    """Entry point: load, validate, encode, split, train, evaluate, and
    save the ISL static-gesture XGBoost classifier.

    Returns
    -------
    int
        Process exit code: ``0`` on success, ``1`` on failure.
    """
    _configure_logging()

    logger.info("=== ISL XGBoost training pipeline: starting ===")
    logger.info("Feature dataset path: %s", X_COMBINED_PATH)
    logger.info("Label dataset path:   %s", Y_COMBINED_PATH)
    logger.info("Model output path:    %s", XGB_MODEL_PKL_PATH)
    logger.info("Encoder output path:  %s", LABEL_ENCODER_PATH)
    logger.info("Expected feature dim: %d", EXPECTED_FEATURE_DIM)
    logger.info("Test split ratio:     %.2f", TEST_SPLIT_RATIO)
    logger.info("Random seed:          %d", RANDOM_SEED)

    try:
        X, y_raw = load_dataset(X_COMBINED_PATH, Y_COMBINED_PATH)
        validate_dataset(X, y_raw, expected_feature_dim=EXPECTED_FEATURE_DIM)

        y_encoded, encoder = encode_labels(y_raw)
        num_classes = int(len(np.unique(y_encoded)))
        logger.info("Number of classes for training: %d", num_classes)

        X_train, X_test, y_train, y_test = split_dataset(
            X, y_encoded, test_size=TEST_SPLIT_RATIO, random_seed=RANDOM_SEED
        )

        model = train_model(
            X_train, y_train, X_test, y_test, num_classes=num_classes
        )

        evaluate_model(model, X_test, y_test)

        save_artifacts(
            model,
            encoder,
            model_path=XGB_MODEL_PKL_PATH,
            encoder_path=LABEL_ENCODER_PATH,
        )
    except TrainingError as exc:
        logger.error("Training pipeline failed: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001 - final safety net for CLI use
        logger.exception("Unexpected error during training pipeline: %s", exc)
        return 1

    logger.info("=== ISL XGBoost training pipeline: completed successfully ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())