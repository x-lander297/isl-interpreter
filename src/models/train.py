#!/usr/bin/env python3
"""
src/models/train.py
====================

Training pipeline for the Indian Sign Language (ISL) Interpreter's
static-gesture XGBoost classifier.

This module loads the combined, padded feature/label dataset
(``X_combined.npy`` / ``y_combined.npy``, produced by
``scripts/merge_data.py``), splits it into train/test sets, optionally
performs grid-search hyperparameter tuning with cross-validation,
trains an ``xgboost.XGBClassifier``, evaluates it with standard
multiclass metrics, renders and saves a confusion-matrix plot, and
persists the trained model and label encoder to disk via ``joblib``.

This module performs **no webcam, MediaPipe, inference, or audio
logic** -- it strictly consumes an already-prepared combined dataset
and produces trained-model artifacts.

Usage
-----
    python -m src.models.train
    python src/models/train.py --tune
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend: safe for headless/CI training runs.
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder

from config import constants

__all__ = [
    "TrainingError",
    "load_data",
    "split_data",
    "train_xgboost",
    "tune_hyperparameters",
    "evaluate_model",
    "plot_confusion_matrix",
    "save_model",
    "train_xgboost_model",
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
# defining the exact attribute names referenced in this file's spec)
# ---------------------------------------------------------------------------
INPUT_DIM: int = int(getattr(constants, "INPUT_DIM", constants.FEATURE_VECTOR_SIZE))

XGB_PARAMS: Dict[str, Any] = dict(
    getattr(constants, "XGB_PARAMS", constants.DEFAULT_XGBOOST_PARAMS)
)

MODEL_PATH: Path = Path(
    getattr(
        constants,
        "MODEL_PATH",
        getattr(constants, "XGB_MODEL_PKL_PATH", constants.MODELS_DIR / "xgb_model.pkl"),
    )
)
ENCODER_PATH: Path = Path(
    getattr(constants, "ENCODER_PATH", constants.LABEL_ENCODER_PATH)
)
DATA_PATH: Path = Path(
    getattr(constants, "DATA_PATH", constants.DATA_DIR / "processed")
)

X_COMBINED_PATH: Path = Path(
    getattr(constants, "X_COMBINED_PATH", DATA_PATH / "X_combined.npy")
)
Y_COMBINED_PATH: Path = Path(
    getattr(constants, "Y_COMBINED_PATH", DATA_PATH / "y_combined.npy")
)

TEST_SPLIT_RATIO: float = float(getattr(constants, "TEST_SPLIT_RATIO", 0.2))
RANDOM_SEED: int = int(getattr(constants, "RANDOM_SEED", 42))

CONFUSION_MATRIX_PATH: Path = Path(
    getattr(constants, "CONFUSION_MATRIX_PATH", constants.MODELS_DIR / "confusion_matrix.png")
)

# Grid-search space for optional hyperparameter tuning.
_TUNING_PARAM_GRID: Dict[str, List[Any]] = {
    "n_estimators": [50, 100, 200],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.01, 0.1, 0.3],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}
_TUNING_CV_FOLDS: int = 5


class TrainingError(RuntimeError):
    """Raised for any unrecoverable failure during data loading,
    validation, splitting, tuning, training, evaluation, plotting, or
    artifact persistence.
    """


def load_data(processed_dir: str = str(DATA_PATH)) -> Tuple[np.ndarray, np.ndarray]:
    """Load ``X_combined.npy`` and ``y_combined.npy`` from a processed data directory.

    Parameters
    ----------
    processed_dir:
        Directory expected to contain ``X_combined.npy`` and
        ``y_combined.npy``. Defaults to the resolved
        ``config.constants`` processed-data directory.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        The ``(X, y)`` arrays, loaded as-is from disk.

    Raises
    ------
    FileNotFoundError
        If either expected ``.npy`` file does not exist.
    TrainingError
        If a file exists but cannot be parsed as a NumPy array, or if
        the loaded arrays are empty / structurally invalid.
    """
    directory = Path(processed_dir)
    x_path = directory / "X_combined.npy"
    y_path = directory / "y_combined.npy"

    for label, path in (("features", x_path), ("labels", y_path)):
        if not path.exists():
            logger.error("Required %s file does not exist: %s", label, path)
            raise FileNotFoundError(f"Required {label} file does not exist: {path}")
        if not path.is_file():
            logger.error("Expected a file for %s but found: %s", label, path)
            raise FileNotFoundError(f"Expected a file for {label} but found: {path}")

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

    if X.size == 0 or y.size == 0:
        logger.error("Loaded dataset is empty: X.size=%d, y.size=%d.", X.size, y.size)
        raise TrainingError("Loaded feature/label arrays must not be empty.")
    if X.ndim != 2:
        logger.error("Feature matrix has unexpected ndim=%d.", X.ndim)
        raise TrainingError(f"Feature matrix must be 2-D; got shape={X.shape}.")
    if X.shape[1] != INPUT_DIM:
        logger.error(
            "Feature dimension mismatch: got %d, expected %d.", X.shape[1], INPUT_DIM
        )
        raise TrainingError(
            f"Feature dimension ({X.shape[1]}) does not match expected "
            f"INPUT_DIM ({INPUT_DIM})."
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

    logger.info(
        "Loaded dataset: X.shape=%s (dtype=%s), y.shape=%s (dtype=%s), "
        "%d unique labels.",
        X.shape,
        X.dtype,
        y.shape,
        y.dtype,
        len(np.unique(y)),
    )
    return X, y


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = TEST_SPLIT_RATIO,
    random_state: int = RANDOM_SEED,
) -> Dict[str, np.ndarray]:
    """Split features and labels into stratified train/test sets.

    Parameters
    ----------
    X:
        Feature matrix of shape ``(num_samples, INPUT_DIM)``.
    y:
        Label array of shape ``(num_samples,)`` (raw, pre-encoding).
    test_size:
        Fraction of samples held out for testing.
    random_state:
        Seed for reproducible splitting.

    Returns
    -------
    Dict[str, numpy.ndarray]
        A dictionary with keys ``"X_train"``, ``"X_test"``,
        ``"y_train"``, ``"y_test"``.

    Raises
    ------
    TrainingError
        If the split fails (e.g. too few samples for the requested
        split configuration).
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
            random_state=random_state,
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
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


def tune_hyperparameters(
    X_train: np.ndarray, y_train: np.ndarray, base_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Tune XGBoost hyperparameters via grid search with cross-validation.

    Searches over ``n_estimators``, ``max_depth``, ``learning_rate``,
    ``subsample``, and ``colsample_bytree`` using 5-fold stratified
    cross-validation, scored on accuracy.

    Parameters
    ----------
    X_train:
        Training feature matrix.
    y_train:
        Encoded (integer) training labels.
    base_params:
        Base parameters (e.g. ``objective``, ``eval_metric``,
        ``random_state``, ``n_jobs``) held fixed across the search;
        merged with each candidate combination from the tuning grid.

    Returns
    -------
    Dict[str, Any]
        The best parameter combination found, merged with
        ``base_params``.

    Raises
    ------
    TrainingError
        If the grid search fails to complete.
    """
    logger.info(
        "Starting hyperparameter tuning: grid=%s, cv_folds=%d.",
        _TUNING_PARAM_GRID,
        _TUNING_CV_FOLDS,
    )

    fixed_params = {
        key: value
        for key, value in base_params.items()
        if key not in _TUNING_PARAM_GRID
    }

    try:
        estimator = xgb.XGBClassifier(**fixed_params)
        search = GridSearchCV(
            estimator=estimator,
            param_grid=_TUNING_PARAM_GRID,
            cv=_TUNING_CV_FOLDS,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train, y_train)
    except Exception as exc:  # noqa: BLE001
        logger.error("Hyperparameter tuning failed: %s", exc)
        raise TrainingError("Hyperparameter tuning via GridSearchCV failed.") from exc

    best_params = {**fixed_params, **search.best_params_}
    logger.info(
        "Hyperparameter tuning complete. Best CV accuracy=%.4f, best_params=%s.",
        search.best_score_,
        search.best_params_,
    )
    return best_params


def train_xgboost(
    X_train: np.ndarray, y_train: np.ndarray, params: Optional[Dict[str, Any]] = None
) -> xgb.XGBClassifier:
    """Train an ``xgboost.XGBClassifier`` on the given training data.

    Parameters
    ----------
    X_train:
        Training feature matrix.
    y_train:
        Encoded (integer) training labels.
    params:
        Hyperparameters to construct the classifier with. Defaults to
        ``config.constants``' resolved ``XGB_PARAMS``.

    Returns
    -------
    xgboost.XGBClassifier
        The fitted classifier.

    Raises
    ------
    TrainingError
        If model construction or fitting fails.
    """
    effective_params = dict(params) if params is not None else dict(XGB_PARAMS)
    effective_params.setdefault("random_state", RANDOM_SEED)

    logger.info("Training XGBoost classifier with parameters: %s", effective_params)

    try:
        model = xgb.XGBClassifier(**effective_params)
        model.fit(X_train, y_train)
    except Exception as exc:  # noqa: BLE001
        logger.error("Model training failed: %s", exc)
        raise TrainingError("XGBoost model training failed.") from exc

    logger.info("Training complete: %d estimators fitted.", model.n_estimators)
    return model


def evaluate_model(
    model: xgb.XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
) -> Dict[str, Any]:
    """Evaluate a trained classifier on a held-out test set.

    Computes accuracy and weighted precision/recall/F1, and produces a
    confusion matrix over the encoder's known class labels.

    Parameters
    ----------
    model:
        The fitted classifier.
    X_test:
        Held-out test feature matrix.
    y_test:
        Encoded (integer) test labels.
    label_encoder:
        The fitted label encoder, used to resolve human-readable class
        names for logging and plotting.

    Returns
    -------
    Dict[str, Any]
        A dictionary with keys: ``"accuracy"``, ``"precision"``,
        ``"recall"``, ``"f1_score"``, ``"confusion_matrix"``,
        ``"y_pred"``, and ``"class_names"``.

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
        accuracy = float(accuracy_score(y_test, y_pred))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred)
        class_names = [str(c) for c in label_encoder.classes_]
    except Exception as exc:  # noqa: BLE001
        logger.error("Metric computation failed: %s", exc)
        raise TrainingError("Failed to compute evaluation metrics.") from exc

    metrics: Dict[str, Any] = {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "class_names": class_names,
    }

    logger.info(
        "Evaluation results: accuracy=%.4f, precision=%.4f, recall=%.4f, f1=%.4f.",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
    )
    logger.debug("Confusion matrix:\n%s", cm)

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: str = str(CONFUSION_MATRIX_PATH),
) -> None:
    """Render and save a confusion-matrix heatmap.

    Parameters
    ----------
    y_true:
        Ground-truth encoded (integer) labels.
    y_pred:
        Predicted encoded (integer) labels.
    class_names:
        Human-readable class names, ordered by encoded index, used for
        axis tick labels.
    save_path:
        Destination file path for the rendered plot (PNG).

    Raises
    ------
    TrainingError
        If plot generation or saving fails.
    """
    output_path = Path(save_path)

    try:
        cm = confusion_matrix(y_true, y_pred)

        figure_size = max(8, len(class_names) * 0.4)
        fig, ax = plt.subplots(figsize=(figure_size, figure_size))
        sns.heatmap(
            cm,
            annot=len(class_names) <= 40,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar=True,
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("ISL Static Gesture Classifier — Confusion Matrix")
        fig.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to generate/save confusion matrix plot: %s", exc)
        raise TrainingError(
            f"Failed to generate or save confusion matrix plot to: {output_path}"
        ) from exc

    logger.info("Confusion matrix plot saved to: %s", output_path)


def save_model(
    model: xgb.XGBClassifier,
    encoder: LabelEncoder,
    model_path: str = str(MODEL_PATH),
    encoder_path: str = str(ENCODER_PATH),
) -> None:
    """Persist the trained model and label encoder to disk via joblib.

    Parameters
    ----------
    model:
        The fitted classifier to persist.
    encoder:
        The fitted label encoder to persist.
    model_path:
        Destination path for the serialized model.
    encoder_path:
        Destination path for the serialized label encoder.

    Raises
    ------
    TrainingError
        If either artifact cannot be written to disk.
    """
    model_out = Path(model_path)
    encoder_out = Path(encoder_path)

    try:
        model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_out)
        logger.info("Saved trained model to: %s", model_out)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to save model to %s: %s", model_out, exc)
        raise TrainingError(f"Failed to save trained model to: {model_out}") from exc

    try:
        encoder_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(encoder, encoder_out)
        logger.info("Saved label encoder to: %s", encoder_out)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to save label encoder to %s: %s", encoder_out, exc)
        raise TrainingError(
            f"Failed to save label encoder to: {encoder_out}"
        ) from exc

'''
def train_xgboost_model(
    processed_dir: str = str(DATA_PATH),
    model_path: str = str(MODEL_PATH),
    encoder_path: str = str(ENCODER_PATH),
    tune: bool = False,
    plot_path: str = str(CONFUSION_MATRIX_PATH),
) -> Dict[str, Any]:
    """Run the full ISL static-gesture XGBoost training pipeline.

    Loads the combined dataset, encodes labels, splits into train/test
    sets, optionally tunes hyperparameters via grid search, trains the
    classifier, evaluates it, plots and saves a confusion matrix, and
    persists the model and encoder to disk.

    Parameters
    ----------
    processed_dir:
        Directory containing ``X_combined.npy`` / ``y_combined.npy``.
    model_path:
        Destination path for the saved model.
    encoder_path:
        Destination path for the saved label encoder.
    tune:
        If ``True``, run :func:`tune_hyperparameters` (5-fold grid
        search) before final training; otherwise train directly with
        ``config.constants``' resolved ``XGB_PARAMS``. Grid search is
        computationally expensive and is opt-in.
    plot_path:
        Destination path for the confusion-matrix plot.

    Returns
    -------
    Dict[str, Any]
        A dictionary with keys ``"model"``, ``"encoder"``, and
        ``"metrics"`` (the dict returned by :func:`evaluate_model`).

    Raises
    ------
    TrainingError
        If any pipeline stage fails.
    """
    logger.info("=== ISL XGBoost training pipeline: starting ===")
    logger.info("Processed data directory: %s", processed_dir)
    logger.info("Model output path:        %s", model_path)
    logger.info("Encoder output path:      %s", encoder_path)
    logger.info("Confusion matrix path:    %s", plot_path)
    logger.info("Hyperparameter tuning:    %s", tune)

    X, y_raw = load_data(processed_dir)

    try:
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y_raw).astype(np.int64)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to fit LabelEncoder on labels: %s", exc)
        raise TrainingError("Failed to fit LabelEncoder on target labels.") from exc

    logger.info(
        "LabelEncoder fitted: %d classes -> %s",
        len(encoder.classes_),
        list(encoder.classes_),
    )

    splits = split_data(X, y_encoded, test_size=TEST_SPLIT_RATIO, random_state=RANDOM_SEED)
    X_train, X_test = splits["X_train"], splits["X_test"]
    y_train, y_test = splits["y_train"], splits["y_test"]

    if tune:
        best_params = tune_hyperparameters(X_train, y_train, base_params=dict(XGB_PARAMS))
        model = train_xgboost(X_train, y_train, params=best_params)
    else:
        model = train_xgboost(X_train, y_train, params=dict(XGB_PARAMS))

    metrics = evaluate_model(model, X_test, y_test, label_encoder=encoder)

    plot_confusion_matrix(
        y_test, metrics["y_pred"], class_names=metrics["class_names"], save_path=plot_path
    )

    save_model(model, encoder, model_path=model_path, encoder_path=encoder_path)

    logger.info("=== ISL XGBoost training pipeline: completed successfully ===")
    return {"model": model, "encoder": encoder, "metrics": metrics}

'''
def train_xgboost_model(
    processed_dir: str = str(DATA_PATH),
    model_path: str = str(MODEL_PATH),
    encoder_path: str = str(ENCODER_PATH),
    tune: bool = False,
    plot_path: str = str(CONFUSION_MATRIX_PATH),
) -> Dict[str, Any]:
    """Run the full ISL static-gesture XGBoost training pipeline..."""
    logger.info("=== ISL XGBoost training pipeline: starting ===")
    logger.info("Processed data directory: %s", processed_dir)
    logger.info("Model output path:        %s", model_path)
    logger.info("Encoder output path:      %s", encoder_path)
    logger.info("Confusion matrix path:    %s", plot_path)
    logger.info("Hyperparameter tuning:    %s", tune)

    X, y_raw = load_data(processed_dir)

    # --- FIX: Fit encoder on class names from constants ---
    try:
        encoder = LabelEncoder()
        encoder.fit(constants.STATIC_LABELS)   # <-- THIS IS THE CHANGE
        y_encoded = y_raw.astype(np.int64)     # y_raw is already integer indices
    except Exception as exc:
        logger.error("Failed to fit LabelEncoder on labels: %s", exc)
        raise TrainingError("Failed to fit LabelEncoder on target labels.") from exc

    logger.info(
        "LabelEncoder fitted: %d classes -> %s",
        len(encoder.classes_),
        list(encoder.classes_),
    )

    splits = split_data(X, y_encoded, test_size=TEST_SPLIT_RATIO, random_state=RANDOM_SEED)
    X_train, X_test = splits["X_train"], splits["X_test"]
    y_train, y_test = splits["y_train"], splits["y_test"]

    if tune:
        best_params = tune_hyperparameters(X_train, y_train, base_params=dict(XGB_PARAMS))
        model = train_xgboost(X_train, y_train, params=best_params)
    else:
        model = train_xgboost(X_train, y_train, params=dict(XGB_PARAMS))

    metrics = evaluate_model(model, X_test, y_test, label_encoder=encoder)

    plot_confusion_matrix(
        y_test, metrics["y_pred"], class_names=metrics["class_names"], save_path=plot_path
    )

    save_model(model, encoder, model_path=model_path, encoder_path=encoder_path)

    logger.info("=== ISL XGBoost training pipeline: completed successfully ===")
    return {"model": model, "encoder": encoder, "metrics": metrics}

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the training script.

    Parameters
    ----------
    argv:
        Argument list to parse (defaults to ``sys.argv[1:]``).

    Returns
    -------
    argparse.Namespace
        Parsed arguments: ``tune`` (bool), ``processed_dir`` (str),
        ``model_path`` (str), ``encoder_path`` (str), ``plot_path``
        (str).
    """
    parser = argparse.ArgumentParser(
        description="Train the ISL static-gesture XGBoost classifier."
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run grid-search hyperparameter tuning before final training.",
    )
    parser.add_argument("--processed-dir", type=str, default=str(DATA_PATH))
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH))
    parser.add_argument("--encoder-path", type=str, default=str(ENCODER_PATH))
    parser.add_argument("--plot-path", type=str, default=str(CONFUSION_MATRIX_PATH))
    return parser.parse_args(argv)


def main() -> int:
    """CLI entry point for the ISL XGBoost training pipeline.

    Returns
    -------
    int
        Process exit code: ``0`` on success, ``1`` on failure.
    """
    _configure_logging()
    args = _parse_args()

    try:
        train_xgboost_model(
            processed_dir=args.processed_dir,
            model_path=args.model_path,
            encoder_path=args.encoder_path,
            tune=args.tune,
            plot_path=args.plot_path,
        )
    except (TrainingError, FileNotFoundError) as exc:
        logger.error("Training pipeline failed: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001 - final safety net for CLI use
        logger.exception("Unexpected error during training pipeline: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())