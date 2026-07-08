"""
processor.py

Frame processing pipeline for the Indian Sign Language interpreter
project. Wraps MediaPipe Hands to extract hand landmarks from BGR
frames (automatically converted to RGB), maintains a rolling buffer of
raw predictions, and produces a temporally-smoothed "stable" prediction
via majority voting.

Note: This module does not perform actual sign classification. The
`raw_prediction` value is a placeholder (always 0) until a trained
model is integrated in a later stage of the pipeline.
"""

import logging
from collections import deque
from typing import Any, Dict, Optional

import cv2
import mediapipe as mp
import numpy as np

from src.utils.smoothing import majority_vote

logger = logging.getLogger(__name__)

# Number of (x, y, z) landmarks per hand as produced by MediaPipe Hands.
NUM_LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3
LANDMARKS_PER_HAND_FLAT = NUM_LANDMARKS_PER_HAND * COORDS_PER_LANDMARK  # 63


class FrameProcessor:
    """
    Processes video frames to detect hand landmarks and produce a
    temporally-smoothed prediction using a rolling buffer.

    This class wraps MediaPipe's Hands solution to detect and extract
    hand landmarks from individual frames. Raw (per-frame) predictions
    are placeholders in this stage of the pipeline and are smoothed
    over a sliding window using majority voting to produce a stable
    prediction.

    Attributes
    ----------
    buffer_size : int
        Maximum number of recent raw predictions to retain.
    min_predictions : int
        Minimum number of buffered predictions required before a
        stable prediction can be computed.
    max_hands : int
        Maximum number of hands MediaPipe should detect per frame.
    min_detection_confidence : float
        Minimum confidence threshold for hand detection.
    min_tracking_confidence : float
        Minimum confidence threshold for hand landmark tracking.
    """

    def __init__(
        self,
        buffer_size: int = 5,
        min_predictions: int = 3,
        max_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """
        Initialize the MediaPipe Hands solution and internal state.

        Parameters
        ----------
        buffer_size : int, optional
            Maximum length of the rolling prediction buffer. Defaults
            to 5.
        min_predictions : int, optional
            Minimum number of predictions required in the buffer
            before a stable prediction is computed. Defaults to 3.
        max_hands : int, optional
            Maximum number of hands to detect. Defaults to 2.
        min_detection_confidence : float, optional
            Minimum confidence for initial hand detection. Defaults
            to 0.5.
        min_tracking_confidence : float, optional
            Minimum confidence for hand landmark tracking across
            frames. Defaults to 0.5.
        """
        self.min_predictions = min_predictions
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.buffer: deque = deque(maxlen=buffer_size)
        self.frame_counter: int = 0

        self._mp_hands_module = mp.solutions.hands
        try:
            self.hands = self._mp_hands_module.Hands(
                static_image_mode=False,
                max_num_hands=self.max_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to initialize MediaPipe Hands: %s", exc)
            raise

        logger.info(
            "FrameProcessor initialized (buffer_size=%d, "
            "min_predictions=%d, max_hands=%d)",
            buffer_size,
            self.min_predictions,
            self.max_hands,
        )

    def process_frame(self, frame: np.ndarray, flip: bool = True) -> Optional[Dict[str, Any]]:
        """
        Process a single BGR frame through the hand detection pipeline.

        Runs MediaPipe hand detection on the frame, extracts hand
        landmarks (if any), updates the internal prediction buffer
        with a placeholder raw prediction, and computes a stable,
        temporally-smoothed prediction.

        Parameters
        ----------
        frame : np.ndarray
            A BGR frame of shape (height, width, 3) with dtype uint8
            (as returned by OpenCV).
        flip : bool, optional
            Whether to mirror the frame horizontally for a natural
            view. Defaults to True.

        Returns
        -------
        Optional[Dict[str, Any]]
            A dictionary containing:
                - "landmarks": Optional[np.ndarray], flattened hand
                  landmark coordinates.
                - "raw_prediction": Any, the current frame's raw
                  (placeholder) prediction.
                - "stable_prediction": Optional[Any], the smoothed
                  prediction from the buffer, or None if insufficient
                  data.
                - "num_hands_detected": int, number of hands detected
                  in the frame.
                - "confidence": Optional[float], placeholder for
                  future model confidence score.
            Returns None if the input frame is invalid.
        """
        if not self._is_valid_frame(frame):
            logger.warning("Invalid frame passed to process_frame; skipping.")
            return None

        self.frame_counter += 1

        # -------- PRE-PROCESSING: flip and convert BGR to RGB --------
        try:
            # Mirror horizontally for a natural selfie view
            if flip:
                frame = cv2.flip(frame, 1)

            # MediaPipe expects RGB, but OpenCV gives BGR
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as exc:
            logger.error("Frame preprocessing failed: %s", exc)
            return None

        # -------- MEDIAPIPE INFERENCE --------
        try:
            results = self.hands.process(rgb_frame)
        except Exception as exc:
            logger.error("MediaPipe processing error: %s", exc)
            return None

        # -------- HANDLE RESULT --------
        if results is None or not getattr(results, "multi_hand_landmarks", None):
            self._update_buffer(0)  # placeholder raw prediction
            stable_prediction = self._get_stable_prediction()
            return {
                "landmarks": None,
                "raw_prediction": None,
                "stable_prediction": stable_prediction,
                "num_hands_detected": 0,
                "confidence": None,
            }

        num_hands_detected = len(results.multi_hand_landmarks)
        landmarks = self._extract_landmarks(results)

        # Placeholder for model inference; actual classification model
        # will replace this in a later stage of the pipeline.
        raw_prediction = 0

        self._update_buffer(raw_prediction)
        stable_prediction = self._get_stable_prediction()

        return {
            "landmarks": landmarks,
            "raw_prediction": raw_prediction,
            "stable_prediction": stable_prediction,
            "num_hands_detected": num_hands_detected,
            "confidence": None,
        }

    def _extract_landmarks(self, results: Any) -> Optional[np.ndarray]:
        """
        Extract flattened landmark coordinates from MediaPipe results.

        Extracts the 21 hand landmarks (x, y, z each) from the first
        detected hand and flattens them into a 1D numpy array.

        Parameters
        ----------
        results : Any
            The results object returned by ``self.hands.process``.

        Returns
        -------
        Optional[np.ndarray]
            A flattened numpy array of shape (63,) containing the
            first detected hand's landmark coordinates, or None if no
            hand landmarks are present.
        """
        if not getattr(results, "multi_hand_landmarks", None):
            return None

        try:
            first_hand = results.multi_hand_landmarks[0]
            coords = []
            for landmark in first_hand.landmark:
                coords.extend([landmark.x, landmark.y, landmark.z])

            landmarks_array = np.array(coords, dtype=np.float32)

            if landmarks_array.shape[0] != LANDMARKS_PER_HAND_FLAT:
                logger.warning(
                    "Unexpected landmark array shape: %s (expected %d).",
                    landmarks_array.shape,
                    LANDMARKS_PER_HAND_FLAT,
                )
                return None

            return landmarks_array

        except Exception as exc:
            logger.error("Failed to extract landmarks: %s", exc)
            return None

    def _update_buffer(self, prediction: Any) -> None:
        """
        Append a new raw prediction to the rolling buffer.

        Parameters
        ----------
        prediction : Any
            The raw prediction value to add to the buffer.
        """
        self.buffer.append(prediction)

    def _get_stable_prediction(self) -> Optional[Any]:
        """
        Compute a temporally-smoothed prediction from the buffer.

        Uses majority voting over the current contents of the
        prediction buffer. If the buffer is empty or insufficient, None.

        Returns
        -------
        Optional[Any]
            The stable prediction, or None if there are fewer than
            ``min_predictions`` entries in the buffer.
        """
        if len(self.buffer) < self.min_predictions:
            return None

        try:
            return majority_vote(list(self.buffer))
        except ValueError as exc:
            logger.error("Failed to compute stable prediction: %s", exc)
            return None

    def reset(self) -> None:
        """Clear the internal prediction buffer."""
        self.buffer.clear()
        logger.info("FrameProcessor buffer has been reset.")

    @staticmethod
    def _is_valid_frame(frame: np.ndarray) -> bool:
        """
        Validate that a frame is suitable for MediaPipe processing.

        Parameters
        ----------
        frame : np.ndarray
            The frame to validate.

        Returns
        -------
        bool
            True if the frame is a non-None numpy array with 3
            dimensions and dtype uint8, False otherwise.
        """
        if frame is None:
            return False
        if not isinstance(frame, np.ndarray):
            return False
        if frame.ndim != 3:
            return False
        if frame.dtype != np.uint8:
            return False
        return True