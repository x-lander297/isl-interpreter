"""
Module: src.data.loader

Purpose: Load raw static ISL images (A-Z, 1-9) from disk into memory as
    uniformly-sized RGB NumPy arrays, paired with their integer class labels.

Owner: P1
Dependencies: cv2, numpy, tqdm, config.constants
"""

import logging
import os
from typing import Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from config import constants


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

_VALID_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg")
_TARGET_SIZE: Tuple[int, int] = (224, 224)


def load_static_images(data_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load all static ISL images from class-labeled subfolders into memory.

    Scans `data_dir` for subfolders named after known class labels (as
    defined in `constants.LABEL_NAME_TO_INDEX`), loads every valid image
    file within each subfolder, converts it to RGB, resizes it to 224x224,
    and returns the full dataset as parallel NumPy arrays.

    Args:
        data_dir (Optional[str]): Path to the root folder containing class
            subfolders (e.g., 'A', 'B', ..., '1', '2', ...). If None,
            defaults to `constants.DATA_RAW_STATIC`.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - X_images: Array of shape (n_samples, 224, 224, 3), dtype=uint8.
              RGB images resized to a uniform shape.
            - y_labels: Array of shape (n_samples,), dtype=int32. Integer
              class labels corresponding to `constants.LABEL_NAME_TO_INDEX`.

    Raises:
        FileNotFoundError: If `data_dir` does not exist.
        ValueError: If no valid images were found across all class folders.
    """
    root_dir = data_dir if data_dir is not None else constants.DATA_RAW_STATIC

    if not os.path.isdir(root_dir):
        raise FileNotFoundError(
            f"Static dataset directory not found: '{root_dir}'. "
            f"Expected subfolders named after class labels "
            f"(e.g., 'A', 'B', ..., '1', '2', ...)."
        )

    images: list = []
    labels: list = []
    classes_found: int = 0

    sorted_labels = sorted(
        constants.LABEL_NAME_TO_INDEX.items(), key=lambda item: item[1]
    )

    for label_name, label_idx in sorted_labels:
        class_dir = os.path.join(root_dir, label_name)

        if not os.path.isdir(class_dir):
            logger.warning(
                "Class folder missing, skipping: '%s' (label='%s', index=%d)",
                class_dir,
                label_name,
                label_idx,
            )
            continue

        all_files = os.listdir(class_dir)
        image_files = [
            f for f in all_files if f.lower().endswith(_VALID_EXTENSIONS)
        ]

        if not image_files:
            logger.warning(
                "No valid image files found in class folder: '%s'", class_dir
            )
            continue

        classes_found += 1

        for filename in tqdm(image_files, desc=f"Loading {label_name}"):
            file_path = os.path.join(class_dir, filename)
            img = cv2.imread(file_path)

            if img is None:
                logger.warning(
                    "Failed to read image (corrupted or unsupported), "
                    "skipping: '%s'",
                    file_path,
                )
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(
                img_rgb, _TARGET_SIZE, interpolation=cv2.INTER_LINEAR
            )

            images.append(img_resized)
            labels.append(label_idx)

    if len(images) == 0:
        raise ValueError(
            f"No valid images found in '{root_dir}'. Verify that class "
            f"subfolders exist and contain readable .png/.jpg/.jpeg files."
        )

    X_imgs = np.array(images, dtype=np.uint8)
    y_labels = np.array(labels, dtype=np.int32)

    if len(X_imgs) != len(y_labels):
        raise ValueError(
            f"Mismatch between number of images ({len(X_imgs)}) and "
            f"number of labels ({len(y_labels)})."
        )

    logger.info(
        "Loaded %d images across %d classes from '%s'.",
        len(X_imgs),
        classes_found,
        root_dir,
    )

    return X_imgs, y_labels