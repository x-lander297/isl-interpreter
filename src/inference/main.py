"""
Module: src.inference.main

Purpose: Application entry point for the real-time ISL Interpreter. Ties
    together webcam capture, landmark extraction/preprocessing, XGBoost
    inference, prediction smoothing, label mapping, and audio feedback into
    a single live loop with an on-screen overlay.

Owner: P1
Dependencies: opencv-python, numpy, config.constants,
    src.inference.camera, src.inference.processor, src.models.predict,
    src.utils.audio, src.utils.label_mapper, src.utils.smoothing
"""

import argparse
import logging
import sys
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from config import constants
from src.inference.camera import Camera
from src.inference.processor import FrameProcessor
from src.models.predict import SignPredictor
from src.utils.audio import TextToSpeech
from src.utils.label_mapper import LabelMapper
from src.utils.smoothing import majority_vote


logger = logging.getLogger(__name__)

_TARGET_FPS: int = 30
_FRAME_INTERVAL_SECONDS: float = 1.0 / _TARGET_FPS

_NO_HAND_TEXT: str = "No hand detected"
_TEXT_POSITION: tuple = (20, 40)
_TEXT_COLOR_PREDICTION: tuple = (0, 255, 0)
_TEXT_COLOR_NO_HAND: tuple = (0, 165, 255)
_TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
_TEXT_SCALE: float = 1.0
_TEXT_THICKNESS: int = 2


class ISLInterpreterApp:
    """Real-time Indian Sign Language interpreter application.

    Orchestrates webcam capture, hand landmark extraction, feature padding,
    XGBoost-based classification, prediction smoothing, on-screen overlay,
    and text-to-speech feedback in a single live inference loop.

    Attributes:
        camera (Camera): Webcam capture handle.
        processor (FrameProcessor): Frame-to-landmark extraction utility.
        predictor (SignPredictor): Loaded classifier used for inference.
        audio (TextToSpeech): Text-to-speech engine with cooldown/mute logic.
        label_mapper (LabelMapper): Integer label to class-name mapper.
        buffer (deque): Rolling buffer of raw predictions used for smoothing.
        running (bool): Whether the main loop is currently active.
        prediction_count (int): Number of stable predictions produced so far.
    """

    def __init__(
        self,
        camera_id: int = constants.CAMERA_INDEX,
        mute: bool = constants.MUTE_DEFAULT,
    ) -> None:
        """Initialize the ISL interpreter application and its subsystems.

        Args:
            camera_id (int): Webcam device index. Defaults to
                `constants.CAMERA_INDEX`.
            mute (bool): Whether audio output starts muted. Defaults to
                `constants.MUTE_DEFAULT`.

        Raises:
            RuntimeError: If the camera fails to open or the model fails
                to load.
        """
        logger.info("Initializing ISL Interpreter application.")

        try:
            self.camera = Camera(camera_id)
            logger.info("Camera initialized (device index=%d).", camera_id)
        except Exception as exc:
            logger.error("Failed to initialize camera: %s", exc)
            raise RuntimeError(f"Camera initialization failed: {exc}") from exc

        try:
            self.processor = FrameProcessor()
            self.predictor = SignPredictor()
            logger.info("Model and frame processor loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load model/processor: %s", exc)
            raise RuntimeError(f"Model initialization failed: {exc}") from exc

        self.audio = TextToSpeech()
        if mute:
            self.audio.toggle_mute()

        self.label_mapper = LabelMapper()

        self.buffer: deque = deque(maxlen=constants.PREDICTION_BUFFER_SIZE)
        self.running: bool = False
        self.prediction_count: int = 0
        self._last_spoken_label: Optional[str] = None

    def _process_frame(self, frame: np.ndarray) -> Optional[str]:
        """Extract landmarks, classify, and smooth a single frame's prediction.

        Args:
            frame (np.ndarray): BGR frame captured from the webcam.

        Returns:
            Optional[str]: The smoothed, human-readable class label if the
                prediction buffer is full and a majority consensus exists;
                None if no hand was detected or no consensus has formed yet.
        """
        result = self.processor.process_frame(frame, flip=False)

        # Extract landmarks from the returned dict
        if isinstance(result, dict):
            landmarks = result.get('landmarks')
            if landmarks is None:
                # No hand detected
                return None
        else:
            # Fallback in case processor returns something else
            landmarks = result
            if landmarks is None:
                return None

        # Ensure landmarks is a numpy array with correct shape
        if not isinstance(landmarks, np.ndarray) or landmarks.ndim != 1:
            logger.error(f"Unexpected landmarks type: {type(landmarks)}")
            return None

        # Defensive padding: pad to constants.INPUT_DIM if needed
        if landmarks.shape[0] < constants.INPUT_DIM:
            padded = np.zeros(constants.INPUT_DIM, dtype=np.float32)
            padded[: landmarks.shape[0]] = landmarks
        else:
            padded = landmarks

        # Run inference
        raw_prediction = self.predictor.predict(padded)
        self.buffer.append(int(raw_prediction))

        if len(self.buffer) < self.buffer.maxlen:
            return None

        smoothed_index = majority_vote(list(self.buffer))
        if smoothed_index is None:
            return None #no consensus yet
        
        # Use the correct method of LabelMapper:
        # If your method is named differently (e.g. get_label),
        # change this line accordingly.
        label = self.label_mapper.decode(smoothed_index)

        self.prediction_count += 1
        return label

    def _overlay_text(
        self,
        frame: np.ndarray,
        label: Optional[str],
        confidence: Optional[float] = None,
    ) -> np.ndarray:
        """Draw the current prediction (or "no hand" status) onto the frame.

        Args:
            frame (np.ndarray): BGR frame to annotate.
            label (Optional[str]): Predicted class label, or None if no
                hand is currently detected.
            confidence (Optional[float]): Optional confidence score to
                display alongside the label.

        Returns:
            np.ndarray: The annotated frame.
        """
        if label is None:
            text = _NO_HAND_TEXT
            color = _TEXT_COLOR_NO_HAND
        else:
            text = f"{label} ({confidence:.2f})" if confidence is not None else label
            color = _TEXT_COLOR_PREDICTION

        cv2.putText(
            frame,
            text,
            _TEXT_POSITION,
            _TEXT_FONT,
            _TEXT_SCALE,
            color,
            _TEXT_THICKNESS,
            cv2.LINE_AA,
        )
        return frame

    def run(self) -> None:
        """Run the main capture-predict-display-speak loop until quit.

        Handles keyboard input:
            'q' - quit the application.
            'm' - toggle audio mute.
            'r' - reset the prediction smoothing buffer.

        Returns:
            None.
        """
        self.running = True
        logger.info("Starting main inference loop. Press 'q' to quit.")

        try:
            while self.running:
                loop_start = time.time()

                frame = self.camera.read()
                if frame is None:
                    logger.error("Failed to read frame from camera. Exiting.")
                    break

                try:
                    label = self._process_frame(frame)
                except Exception as exc:
                    logger.exception("Error processing frame: %s", exc)
                    label = None

                if label is not None and label != self._last_spoken_label:
                    self.audio.speak(label)
                    self._last_spoken_label = label
                elif label is None:
                    self._last_spoken_label = None

                frame = self._overlay_text(frame, label)
                cv2.imshow("ISL Interpreter", frame)

                key = cv2.waitKey(10) & 0xFF
                if key == ord("q"):
                    logger.info("Quit key pressed. Shutting down.")
                    self.running = False
                elif key == ord("m"):
                    self.audio.toggle_mute()
                    logger.info("Mute toggled.")
                elif key == ord("r"):
                    self.buffer.clear()
                    self._last_spoken_label = None
                    logger.info("Prediction buffer reset.")

                elapsed = time.time() - loop_start
                remaining = _FRAME_INTERVAL_SECONDS - elapsed
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Release the camera and close all OpenCV windows.

        Returns:
            None.
        """
        self.camera.release()
        self.audio.stop()
        cv2.destroyAllWindows()
        logger.info(
            "Application shut down. Total stable predictions made: %d.",
            self.prediction_count,
        )

    def __enter__(self) -> "ISLInterpreterApp":
        """Enter the context manager, returning the initialized app instance."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager, ensuring resources are released."""
        self.cleanup()


def main() -> int:
    """CLI entry point for the ISL interpreter.

    Returns:
        int: Process exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="ISL Interpreter - Real-time gesture recognition"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=constants.CAMERA_INDEX,
        help=f"Camera device index (default: {constants.CAMERA_INDEX})",
    )
    parser.add_argument(
        "--mute", action="store_true", help="Start with audio muted"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without displaying video (for headless testing)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    try:
        with ISLInterpreterApp(camera_id=args.camera, mute=args.mute) as app:
            if not args.no_display:
                app.run()
            else:
                logger.info(
                    "--no-display set, running in headless mode (no video output)."
                )
                for i in range(10):
                    frame = app.camera.read()
                    if frame is not None:
                        label = app._process_frame(frame)
                        logger.info("Frame %d: prediction=%s", i + 1, label)
                    time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        return 0
    except Exception as exc:
        logger.exception("Application error: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
