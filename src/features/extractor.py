"""
Module: src.features.extractor

Purpose: Wraps MediaPipe Hands to extract 63-dimensional hand landmark
    feature vectors (21 landmarks x 3 coordinates) from a single RGB image
    or BGR video frame.

Owner: P1
Dependencies: mediapipe, opencv-python, numpy, config.constants
"""

import logging
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp

from config import constants


logger = logging.getLogger(__name__)

_EXPECTED_FEATURE_DIM: int = (
    constants.NUM_HAND_LANDMARKS * constants.COORDS_PER_LANDMARK
)

try:
    _mp_hands_solution = mp.solutions.hands
    _hands = _mp_hands_solution.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=constants.CONFIDENCE_THRESHOLD,
    )
    logger.info(
        "MediaPipe Hands initialized (static_image_mode=True, "
        "max_num_hands=1, min_detection_confidence=%.2f).",
        constants.CONFIDENCE_THRESHOLD,
    )
except Exception as exc:
    logger.error("Failed to initialize MediaPipe Hands: %s", exc)
    raise RuntimeError(
        f"MediaPipe Hands initialization failed: {exc}"
    ) from exc


def get_landmarks_from_image(img_rgb: np.ndarray) -> Optional[np.ndarray]:
    """Extract a 63-dimensional hand landmark feature vector from an RGB image.

    Runs MediaPipe Hands on a single RGB image and flattens the first
    detected hand's 21 landmarks (x, y, z each) into a 1D feature vector.

    Args:
        img_rgb (np.ndarray): RGB image of shape (H, W, 3), dtype uint8.

    Returns:
        Optional[np.ndarray]: Array of shape (63,), dtype float32, containing
            [x1, y1, z1, x2, y2, z2, ..., x21, y21, z21]. Returns None if no
            hand is detected in the image.

    Raises:
        TypeError: If `img_rgb` is not a NumPy array.
        ValueError: If `img_rgb` does not have shape (H, W, 3).
    """
    if not isinstance(img_rgb, np.ndarray):
        raise TypeError(
            f"img_rgb must be a numpy.ndarray, got {type(img_rgb).__name__}."
        )

    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(
            f"img_rgb must have shape (H, W, 3), got shape {img_rgb.shape}."
        )

    results = _hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        logger.debug("No hand detected in image.")
        return None

    hand_landmarks = results.multi_hand_landmarks[0]

    features = np.empty(_EXPECTED_FEATURE_DIM, dtype=np.float32)
    for idx, landmark in enumerate(hand_landmarks.landmark):
        offset = idx * constants.COORDS_PER_LANDMARK
        features[offset] = landmark.x
        features[offset + 1] = landmark.y
        features[offset + 2] = landmark.z

    return features


def get_landmarks_from_frame(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Extract a 63-dimensional hand landmark feature vector from a BGR frame.

    Convenience wrapper for callers working with OpenCV-native BGR frames
    (e.g., webcam captures). Converts to RGB internally before delegating to
    `get_landmarks_from_image`.

    Args:
        frame_bgr (np.ndarray): BGR image of shape (H, W, 3), dtype uint8,
            as returned by `cv2.VideoCapture.read()` or `cv2.imread()`.

    Returns:
        Optional[np.ndarray]: Array of shape (63,), dtype float32, or None
            if no hand is detected.

    Raises:
        TypeError: If `frame_bgr` is not a NumPy array.
        ValueError: If `frame_bgr` does not have shape (H, W, 3).
    """
    if not isinstance(frame_bgr, np.ndarray):
        raise TypeError(
            f"frame_bgr must be a numpy.ndarray, got {type(frame_bgr).__name__}."
        )

    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(
            f"frame_bgr must have shape (H, W, 3), got shape {frame_bgr.shape}."
        )

    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return get_landmarks_from_image(img_rgb)