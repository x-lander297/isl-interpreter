"""
Module: src.features.static_pipeline

Purpose: Process all static ISL images (A-Z, 1-9) to extract 63-dimensional
    hand landmark features and save them as .npy files for training.

Owner: P1
Dependencies: numpy, tqdm, logging, config.constants, src.data.loader, src.features.extractor
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from config import constants
from src.data.loader import load_static_images
from src.features.extractor import get_landmarks_from_image

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.propagate = False


def extract_static_features(data_dir: Optional[Path] = None) -> None:
    """Extract 63-dim hand landmarks from all static images and save as .npy.

    Args:
        data_dir (Optional[Path]): Path to the root folder containing class
            subfolders (e.g., 'A', 'B', ..., '1', '2', ...). If None,
            defaults to `constants.DATA_DIR / "raw" / "static"`.

    Raises:
        ValueError: If no images were loaded.
        RuntimeError: If no hand landmarks were detected in any image.
    """
    logger.info("Loading static images...")
    X_imgs, y_labels = load_static_images(data_dir)
    logger.info("Loaded %d images.", len(X_imgs))

    X_features = []
    y_filtered = []

    logger.info("Extracting hand landmarks from images...")
    for idx, img in tqdm(enumerate(X_imgs), total=len(X_imgs), desc="Extracting"):
        feat = get_landmarks_from_image(img)
        if feat is not None:
            X_features.append(feat)
            y_filtered.append(y_labels[idx])
        else:
            logger.debug("No hand detected in image %d", idx)

    if len(X_features) == 0:
        raise RuntimeError(
            "No hand landmarks detected in any image. "
            "Check that images contain visible hands and that MediaPipe is working."
        )

    X_features = np.array(X_features, dtype=np.float32)
    y_filtered = np.array(y_filtered, dtype=np.int32)

    # Ensure output directory exists
    output_dir: Path = constants.DATA_DIR / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save using constants paths
    np.save(constants.STATIC_LANDMARKS_PATH, X_features)
    np.save(constants.LABELS_PATH, y_filtered)

    logger.info(
        "Extracted landmarks for %d samples (skipped %d).",
        len(X_features),
        len(X_imgs) - len(X_features),
    )
    logger.info("Feature shape: %s", X_features.shape)
    logger.info("Saved to: %s", constants.STATIC_LANDMARKS_PATH)
    logger.info("Saved labels to: %s", constants.LABELS_PATH)

    # Class balance summary
    unique, counts = np.unique(y_filtered, return_counts=True)
    for label_idx, count in zip(unique, counts):
        label_name = constants.LABEL_INDEX_TO_NAME.get(label_idx, "Unknown")
        logger.info("Class %s (%d): %d samples", label_name, label_idx, count)