"""
camera.py

Camera capture utility for the Indian Sign Language interpreter
pipeline. Provides a thin, robust wrapper around ``cv2.VideoCapture``
with lazy initialization, frame skipping, resizing, and BGR-to-RGB
conversion suitable for feeding downstream inference models.
"""

import logging
import time
from typing import Optional

import cv2 # type: ignore
import numpy as np # type: ignore

logger = logging.getLogger(__name__)


class Camera:
    """
    A lazy-initializing wrapper around ``cv2.VideoCapture``.

    This class encapsulates camera lifecycle management (start/read/
    release), frame skipping, resizing to a target resolution, and
    color-space conversion from BGR (OpenCV's native format) to RGB
    (commonly expected by ML models).

    Attributes
    ----------
    camera_id : int
        Identifier of the camera device (e.g., 0 for default webcam).
    width : int
        Desired frame width in pixels.
    height : int
        Desired frame height in pixels.
    fps : int
        Desired capture frame rate.
    skip_frames : int
        Number of frames to skip between returned frames. A value of
        0 means every frame is returned; a value of 2 means every
        third frame is returned, etc.
    """

    def __init__(
        self,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        skip_frames: int = 0,
    ) -> None:
        """
        Initialize camera configuration without opening the device.

        Parameters
        ----------
        camera_id : int, optional
            Identifier of the camera device to open. Defaults to 0.
        width : int, optional
            Target frame width in pixels. Defaults to 640.
        height : int, optional
            Target frame height in pixels. Defaults to 480.
        fps : int, optional
            Target capture frame rate. Defaults to 30.
        skip_frames : int, optional
            Number of frames to skip between returned frames. Defaults
            to 0 (no skipping).

        Notes
        -----
        The camera device is not opened here. Initialization is lazy
        and occurs on the first call to :meth:`start` or :meth:`read`.
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.skip_frames = skip_frames

        self.cap: Optional[cv2.VideoCapture] = None
        self._is_opened: bool = False
        self._frame_counter: int = 0

    def start(self) -> bool:
        """
        Open the camera device and configure its resolution and FPS.

        Returns
        -------
        bool
            True if the camera was successfully opened, False
            otherwise.
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_id)

            if self.cap is None or not self.cap.isOpened():
                logger.error(
                    "Failed to open camera with id=%s", self.camera_id
                )
                self._is_opened = False
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            self._is_opened = True
            self._frame_counter = 0
            logger.info(
                "Camera id=%s started (requested %dx%d @ %d fps)",
                self.camera_id,
                self.width,
                self.height,
                self.fps,
            )
            return True

        except cv2.error as exc:
            logger.error("OpenCV error while starting camera: %s", exc)
            self._is_opened = False
            return False
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error while starting camera: %s", exc)
            self._is_opened = False
            return False

    def read(self) -> Optional[np.ndarray]:
        """
        Read and return the next processed frame from the camera.

        If the camera has not been started yet, this method will
        attempt to start it automatically. The returned frame is
        resized to the configured (width, height) and converted from
        BGR to RGB color space. Frame skipping is applied based on
        the ``skip_frames`` configuration.

        Returns
        -------
        Optional[np.ndarray]
            The processed RGB frame as a uint8 numpy array, or None
            if a frame could not be read, is invalid, is skipped, or
            an error occurred during processing.
        """
        if not self._is_opened:
            if not self.start():
                logger.error("Camera could not be started; read() aborted.")
                return None

        try:
            ret, frame = self.cap.read()
        except cv2.error as exc:
            logger.error("OpenCV error while reading frame: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error while reading frame: %s", exc)
            return None

        if not ret or frame is None:
            logger.warning("Failed to read a valid frame from camera.")
            return None

        current_index = self._frame_counter
        self._frame_counter += 1

        if self.skip_frames > 0:
            if current_index % (self.skip_frames + 1) != 0:
                return None

        try:
            resized = cv2.resize(frame, (self.width, self.height))
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        except cv2.error as exc:
            logger.error("OpenCV error while processing frame: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error while processing frame: %s", exc)
            return None

        return rgb_frame.astype(np.uint8)

    def release(self) -> None:
        """
        Release the camera device and reset internal state.

        Safe to call multiple times; subsequent calls after the
        camera has already been released are no-ops aside from
        logging.
        """
        if self.cap is not None:
            try:
                self.cap.release()
            except cv2.error as exc:
                logger.error("OpenCV error while releasing camera: %s", exc)
            except Exception as exc:  # noqa: BLE001
                logger.error("Unexpected error while releasing camera: %s", exc)

        self._is_opened = False
        logger.info("Camera id=%s released.", self.camera_id)

    def is_opened(self) -> bool:
        """
        Check whether the camera is currently opened.

        Returns
        -------
        bool
            True if the camera has been successfully started and has
            not been released, False otherwise.
        """
        return self._is_opened