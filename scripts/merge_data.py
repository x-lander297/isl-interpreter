#!/usr/bin/env python3
"""
scripts/merge_data.py
======================

Data-merging step of the Indian Sign Language Interpreter's ML pipeline:

    MediaPipe landmarks -> Feature preparation -> XGBoost classifier

This script combines previously extracted static-gesture landmark
features (``static_landmarks.npy``) with their corresponding labels
(``static_labels.npy``) into a single, XGBoost-ready dataset:

    X_combined.npy  (shape: [num_samples, 126])
    y_combined.npy  (shape: [num_samples])

Responsibilities
-----------------
1. Load the landmark-feature array and the label array from disk.
2. Validate that both files exist, are non-empty, and have a matching
   number of samples.
3. Pad the feature vectors up to the 126-dimensional input expected by
   the downstream XGBoost classifier (63 real landmark values + 63
   zero-padding values), raising a clear error if the input dimension
   is larger than 126 (i.e. cannot be padded, would require
   truncation).
4. Save the combined ``X_combined.npy`` / ``y_combined.npy`` arrays.

This script performs **no model training, no prediction, and no label
encoding** -- it is strictly a data-preparation step.

Usage
-----
    python scripts/merge_data.py
    ./scripts/merge_data.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Final

import numpy as np

from config import constants

__all__ = [
    "MergeDataError",
    "load_landmarks_and_labels",
    "validate_dataset",
    "pad_feature_vectors",
    "save_combined_dataset",
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
# Configuration resolution (defensive against config/constants.py not yet
# defining .npy-specific attribute names)
# ---------------------------------------------------------------------------
_DEFAULT_PROCESSED_DIR: Final[Path] = constants.DATA_DIR / "processed"

STATIC_LANDMARKS_NPY_PATH: Final[Path] = Path(
    getattr(
        constants,
        "STATIC_LANDMARKS_NPY_PATH",
        constants.DATA_DIR / "static" / "static_landmarks.npy",
    )
)
STATIC_LABELS_NPY_PATH: Final[Path] = Path(
    getattr(
        constants,
        "STATIC_LABELS_NPY_PATH",
        constants.DATA_DIR / "static" / "static_labels.npy",
    )
)
X_COMBINED_PATH: Final[Path] = Path(
    getattr(constants, "X_COMBINED_PATH", _DEFAULT_PROCESSED_DIR / "X_combined.npy")
)
Y_COMBINED_PATH: Final[Path] = Path(
    getattr(constants, "Y_COMBINED_PATH", _DEFAULT_PROCESSED_DIR / "y_combined.npy")
)

# Target feature-vector dimension expected by the XGBoost classifier.
# Falls back to 126 if config.constants does not (yet) define it.
TARGET_FEATURE_DIM: Final[int] = int(getattr(constants, "FEATURE_VECTOR_SIZE", 126))


class MergeDataError(RuntimeError):
    """Raised for any unrecoverable failure while merging the static
    gesture dataset (missing files, empty arrays, sample-count mismatch,
    or an incompatible feature dimension).
    """


def load_landmarks_and_labels(
    landmarks_path: Path, labels_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Load the raw landmark-feature array and label array from disk.

    Parameters
    ----------
    landmarks_path:
        Path to the ``static_landmarks.npy`` file.
    labels_path:
        Path to the ``static_labels.npy`` file.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The ``(landmarks, labels)`` arrays, loaded as-is from disk.

    Raises
    ------
    MergeDataError
        If either file does not exist, or cannot be loaded/parsed as a
        NumPy array.
    """
    for label, path in (("landmarks", landmarks_path), ("labels", labels_path)):
        if not path.exists():
            logger.error("Required %s file does not exist: %s", label, path)
            raise MergeDataError(f"Required {label} file does not exist: {path}")
        if not path.is_file():
            logger.error("Expected a file for %s but found a directory: %s", label, path)
            raise MergeDataError(f"Expected a file for {label} but found: {path}")

    try:
        logger.info("Loading landmarks from: %s", landmarks_path)
        landmarks = np.load(landmarks_path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load landmarks file %s: %s", landmarks_path, exc)
        raise MergeDataError(f"Failed to load landmarks file: {landmarks_path}") from exc

    try:
        logger.info("Loading labels from: %s", labels_path)
        labels = np.load(labels_path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load labels file %s: %s", labels_path, exc)
        raise MergeDataError(f"Failed to load labels file: {labels_path}") from exc

    logger.info(
        "Loaded raw arrays: landmarks.shape=%s (dtype=%s), labels.shape=%s (dtype=%s).",
        landmarks.shape,
        landmarks.dtype,
        labels.shape,
        labels.dtype,
    )
    return landmarks, labels


def validate_dataset(landmarks: np.ndarray, labels: np.ndarray) -> None:
    """Validate structural integrity of the loaded landmarks/labels arrays.

    Parameters
    ----------
    landmarks:
        The raw landmark-feature array.
    labels:
        The raw label array.

    Raises
    ------
    MergeDataError
        If either array is empty, ``landmarks`` is not 2-D, ``labels``
        is not 1-D, or the number of samples does not match between the
        two arrays.
    """
    if landmarks.size == 0:
        logger.error("Landmarks array is empty.")
        raise MergeDataError("Landmarks array is empty; nothing to merge.")
    if labels.size == 0:
        logger.error("Labels array is empty.")
        raise MergeDataError("Labels array is empty; nothing to merge.")

    if landmarks.ndim != 2:
        logger.error("Landmarks array has unexpected ndim=%d.", landmarks.ndim)
        raise MergeDataError(
            f"Landmarks array must be 2-D (num_samples, feature_dim); "
            f"got ndim={landmarks.ndim} with shape={landmarks.shape}."
        )
    if labels.ndim != 1:
        logger.error("Labels array has unexpected ndim=%d.", labels.ndim)
        raise MergeDataError(
            f"Labels array must be 1-D (num_samples,); "
            f"got ndim={labels.ndim} with shape={labels.shape}."
        )

    num_landmark_samples = landmarks.shape[0]
    num_label_samples = labels.shape[0]
    if num_landmark_samples != num_label_samples:
        logger.error(
            "Sample count mismatch: landmarks has %d samples, labels has %d samples.",
            num_landmark_samples,
            num_label_samples,
        )
        raise MergeDataError(
            "Number of samples does not match between landmarks "
            f"({num_landmark_samples}) and labels ({num_label_samples})."
        )

    logger.info(
        "Validation passed: %d samples, feature_dim=%d.",
        num_landmark_samples,
        landmarks.shape[1],
    )


def pad_feature_vectors(
    landmarks: np.ndarray, target_dim: int = TARGET_FEATURE_DIM
) -> np.ndarray:
    """Zero-pad landmark feature vectors up to ``target_dim`` columns.

    Parameters
    ----------
    landmarks:
        A 2-D array of shape ``(num_samples, current_feature_dim)``.
    target_dim:
        The required final feature dimension (defaults to the project's
        configured ``FEATURE_VECTOR_SIZE``, typically 126).

    Returns
    -------
    numpy.ndarray
        A 2-D array of shape ``(num_samples, target_dim)``. If
        ``current_feature_dim == target_dim``, the array is returned
        unchanged (as a ``float32`` copy). Otherwise, columns
        ``[current_feature_dim, target_dim)`` are filled with zeros.

    Raises
    ------
    MergeDataError
        If ``current_feature_dim > target_dim`` (the feature vector is
        already too large and cannot be safely padded/truncated), or if
        ``target_dim`` is not a positive integer.
    """
    if target_dim <= 0:
        logger.error("Invalid target_dim=%d; must be positive.", target_dim)
        raise MergeDataError(f"'target_dim' must be a positive integer; got {target_dim}.")

    current_dim = landmarks.shape[1]

    if current_dim > target_dim:
        logger.error(
            "Feature dimension %d exceeds target dimension %d; cannot pad.",
            current_dim,
            target_dim,
        )
        raise MergeDataError(
            f"Landmark feature dimension ({current_dim}) exceeds the "
            f"required target dimension ({target_dim}). This indicates "
            "an incompatible or corrupted feature-extraction output; "
            "truncation is not performed automatically."
        )

    landmarks_f32 = landmarks.astype(np.float32, copy=False)

    if current_dim == target_dim:
        logger.info(
            "Feature dimension already matches target (%d); no padding applied.",
            target_dim,
        )
        return landmarks_f32.copy()

    num_samples = landmarks_f32.shape[0]
    padding_width = target_dim - current_dim
    padding = np.zeros((num_samples, padding_width), dtype=np.float32)
    padded = np.concatenate([landmarks_f32, padding], axis=1)

    logger.info(
        "Padded feature vectors from dimension %d to %d (added %d zero columns).",
        current_dim,
        target_dim,
        padding_width,
    )
    return padded


def save_combined_dataset(
    X_combined: np.ndarray,
    y_combined: np.ndarray,
    x_output_path: Path,
    y_output_path: Path,
) -> None:
    """Persist the combined feature/label arrays to disk as ``.npy`` files.

    Parameters
    ----------
    X_combined:
        The final, padded feature matrix.
    y_combined:
        The label array (passed through unchanged).
    x_output_path:
        Destination path for the combined feature matrix.
    y_output_path:
        Destination path for the combined label array.

    Raises
    ------
    MergeDataError
        If the output directories cannot be created, or the arrays
        cannot be written to disk.
    """
    try:
        x_output_path.parent.mkdir(parents=True, exist_ok=True)
        y_output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error("Failed to create output directories: %s", exc)
        raise MergeDataError("Failed to create output directories for combined dataset.") from exc

    try:
        np.save(x_output_path, X_combined)
        logger.info(
            "Saved combined feature matrix: shape=%s -> %s",
            X_combined.shape,
            x_output_path,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to save %s: %s", x_output_path, exc)
        raise MergeDataError(f"Failed to save combined features to: {x_output_path}") from exc

    try:
        np.save(y_output_path, y_combined)
        logger.info(
            "Saved combined label array: shape=%s -> %s",
            y_combined.shape,
            y_output_path,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to save %s: %s", y_output_path, exc)
        raise MergeDataError(f"Failed to save combined labels to: {y_output_path}") from exc


def main() -> int:
    """Entry point: load, validate, pad, and save the combined dataset.

    Returns
    -------
    int
        Process exit code: ``0`` on success, ``1`` on failure.
    """
    _configure_logging()

    logger.info("=== ISL static dataset merge: starting ===")
    logger.info("Landmarks input path: %s", STATIC_LANDMARKS_NPY_PATH)
    logger.info("Labels input path:    %s", STATIC_LABELS_NPY_PATH)
    logger.info("Feature output path:  %s", X_COMBINED_PATH)
    logger.info("Label output path:    %s", Y_COMBINED_PATH)
    logger.info("Target feature dim:   %d", TARGET_FEATURE_DIM)

    try:
        landmarks, labels = load_landmarks_and_labels(
            STATIC_LANDMARKS_NPY_PATH, STATIC_LABELS_NPY_PATH
        )
        validate_dataset(landmarks, labels)
        X_combined = pad_feature_vectors(landmarks, target_dim=TARGET_FEATURE_DIM)
        y_combined = labels

        logger.info(
            "Final combined dataset shapes: X_combined=%s, y_combined=%s.",
            X_combined.shape,
            y_combined.shape,
        )

        save_combined_dataset(
            X_combined,
            y_combined,
            x_output_path=X_COMBINED_PATH,
            y_output_path=Y_COMBINED_PATH,
        )
    except MergeDataError as exc:
        logger.error("Dataset merge failed: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001 - final safety net for CLI use
        logger.exception("Unexpected error during dataset merge: %s", exc)
        return 1

    logger.info("=== ISL static dataset merge: completed successfully ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())