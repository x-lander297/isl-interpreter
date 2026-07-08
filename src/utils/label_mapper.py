<<<<<<< HEAD
from config.constants import LABEL_INDEX_TO_NAME

def get_class_name(label_idx):
    return LABEL_INDEX_TO_NAME.get(label_idx, "Unknown")

def get_all_labels():
    return list(LABEL_INDEX_TO_NAME.values())
=======
"""
label_mapper.py
================

Utility for converting Indian Sign Language (ISL) XGBoost classifier
prediction outputs (numeric class indices) into human-readable gesture
labels.

Two label-resolution strategies are supported:

1. **Encoder-backed mapping** (preferred): a fitted
   ``sklearn.preprocessing.LabelEncoder`` (or any object exposing an
   equivalent ``classes_`` / ``inverse_transform`` interface) is loaded
   from disk via ``joblib`` and used to decode indices back to their
   original string labels.
2. **Direct mapping fallback**: when no encoder file is available (or
   none was fitted because training used already-numeric labels), a
   configurable, deterministic index-to-label mapping is used instead.
   By default this is the static ISL alphabet/number class set
   (``0``-``9`` followed by ``A``-``Z``), but callers may supply their
   own mapping to support additional gesture classes, dynamic gestures,
   or alternate label sets.

This module performs **no model training, webcam processing,
MediaPipe, audio output, or prediction generation** -- it strictly maps
already-produced numeric indices to human-readable labels.

Example
-------
>>> from src.utils.label_mapper import LabelMapper
>>> mapper = LabelMapper()  # doctest: +SKIP
>>> mapper.decode(3)  # doctest: +SKIP
'3'
>>> mapper.get_classes()  # doctest: +SKIP
['0', '1', ..., '9', 'A', 'B', ..., 'Z']
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import joblib

from config.constants import LABEL_ENCODER_PATH

__all__ = [
    "LabelMapper",
    "LabelMappingError",
    "EncoderLoadError",
    "InvalidPredictionError",
    "DEFAULT_STATIC_ISL_CLASSES",
]

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Default fallback mapping: static ISL alphabet (A-Z) and numbers (0-9),
# index-ordered, used when no fitted LabelEncoder is available on disk.
# ---------------------------------------------------------------------------
DEFAULT_STATIC_ISL_CLASSES: tuple[str, ...] = tuple(
    [str(digit) for digit in range(10)]
    + [chr(code) for code in range(ord("A"), ord("Z") + 1)]
)


class LabelMappingError(RuntimeError):
    """Base exception for all label-mapping failures in this module."""


class EncoderLoadError(LabelMappingError):
    """Raised when a saved label encoder cannot be located, loaded, or
    validated (e.g. the file is missing, unreadable, or does not expose
    the expected encoder interface).
    """


class InvalidPredictionError(LabelMappingError):
    """Raised when a numeric prediction index is invalid: negative,
    non-integer, or outside the range of known classes.
    """


class LabelMapper:
    """Map numeric XGBoost prediction indices to human-readable ISL labels.

    Parameters
    ----------
    encoder_path:
        Path to a joblib-serialized label encoder (typically a fitted
        ``sklearn.preprocessing.LabelEncoder``). Defaults to
        ``config.constants.LABEL_ENCODER_PATH``.
    fallback_classes:
        A sequence of class labels, index-ordered, used when no encoder
        file is found at ``encoder_path`` (Case 2: direct character
        mapping). Defaults to :data:`DEFAULT_STATIC_ISL_CLASSES` (digits
        ``0``-``9`` followed by letters ``A``-``Z``). Supplying a custom
        sequence here allows this class to support additional gesture
        classes, dynamic-gesture label sets, or entirely different
        vocabularies without modifying this module.
    require_encoder:
        If ``True``, the absence of a valid encoder file at
        ``encoder_path`` raises :class:`EncoderLoadError` instead of
        silently falling back to ``fallback_classes``. Defaults to
        ``False`` (fallback is used), since not all trained models use
        an encoder (e.g. when labels were already numeric at training
        time).

    Notes
    -----
    Each ``LabelMapper`` instance holds exactly one "active" label
    source (either an encoder or a fallback list). Multiple instances
    can be constructed side-by-side -- e.g. one per trained model or per
    label set -- to support future multi-model or multi-label-set
    scenarios.
    """

    def __init__(
        self,
        encoder_path: Path | str = LABEL_ENCODER_PATH,
        fallback_classes: Sequence[str] = DEFAULT_STATIC_ISL_CLASSES,
        require_encoder: bool = False,
    ) -> None:
        if not fallback_classes:
            raise ValueError("'fallback_classes' must not be empty.")

        self._encoder_path: Path = Path(encoder_path)
        self._fallback_classes: tuple[str, ...] = tuple(fallback_classes)
        self._require_encoder: bool = require_encoder

        self._encoder: object | None = None
        self._classes: tuple[str, ...] = ()
        self._using_encoder: bool = False

        self._load(initial=True)

        logger.info(
            "LabelMapper initialized: source=%s, num_classes=%d.",
            "encoder" if self._using_encoder else "fallback_mapping",
            len(self._classes),
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load(self, initial: bool = False) -> None:
        """Load the active label source (encoder or fallback mapping).

        Parameters
        ----------
        initial:
            Whether this is the first load performed during
            construction (affects only log phrasing).

        Raises
        ------
        EncoderLoadError
            If ``require_encoder`` is ``True`` and a valid encoder
            cannot be loaded, or if an encoder file exists but is
            corrupted / does not expose the expected interface.
        """
        verb = "Loading" if initial else "Reloading"

        if self._encoder_path.exists():
            logger.info("%s label encoder from: %s", verb, self._encoder_path)
            try:
                encoder = joblib.load(self._encoder_path)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failed to load label encoder from %s: %s",
                    self._encoder_path,
                    exc,
                )
                raise EncoderLoadError(
                    f"Failed to load label encoder from: {self._encoder_path}. "
                    "The file may be corrupted or incompatible."
                ) from exc

            classes = self._extract_classes(encoder)
            if classes is None:
                message = (
                    f"Object loaded from {self._encoder_path} does not expose "
                    "a valid label-encoder interface ('classes_' and "
                    "'inverse_transform'). The file may be corrupted."
                )
                logger.error(message)
                raise EncoderLoadError(message)

            self._encoder = encoder
            self._classes = classes
            self._using_encoder = True
            logger.info(
                "Label encoder loaded successfully: %d classes -> %s.",
                len(self._classes),
                self._classes,
            )
            return

        if self._require_encoder:
            message = (
                f"No label encoder file found at {self._encoder_path}, and "
                "'require_encoder' is True."
            )
            logger.error(message)
            raise EncoderLoadError(message)

        logger.warning(
            "No label encoder file found at %s; falling back to direct "
            "class mapping with %d classes.",
            self._encoder_path,
            len(self._fallback_classes),
        )
        self._encoder = None
        self._classes = self._fallback_classes
        self._using_encoder = False

    @staticmethod
    def _extract_classes(encoder: object) -> tuple[str, ...] | None:
        """Extract an ordered class-label tuple from a loaded encoder object.

        Parameters
        ----------
        encoder:
            The object loaded via ``joblib.load``.

        Returns
        -------
        Optional[tuple[str, ...]]
            The encoder's classes, in index order, as strings; or
            ``None`` if ``encoder`` does not expose the expected
            interface (``classes_`` attribute and ``inverse_transform``
            method), indicating a corrupted or incompatible file.
        """
        if encoder is None:
            return None
        if not hasattr(encoder, "classes_") or not hasattr(encoder, "inverse_transform"):
            return None
        try:
            return tuple(str(label) for label in encoder.classes_)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to enumerate encoder classes_: %s", exc)
            return None

    def reload(self) -> None:
        """Reload the label source from ``encoder_path`` (or fallback).

        Useful if the underlying encoder file has been replaced on disk
        (e.g. after retraining a model) and this mapper should pick up
        the new class set without constructing a new instance.

        Raises
        ------
        EncoderLoadError
            Under the same conditions as during construction (see
            :meth:`__init__` / :meth:`_load`).
        """
        logger.info("Reloading label mapping from: %s", self._encoder_path)
        self._load(initial=False)

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------
    def decode(self, prediction_index: int) -> str:
        """Convert a numeric model prediction index into a human-readable label.

        Parameters
        ----------
        prediction_index:
            The numeric class index predicted by the classifier (e.g.
            ``argmax`` of an XGBoost ``predict_proba`` output).

        Returns
        -------
        str
            The corresponding human-readable gesture label.

        Raises
        ------
        InvalidPredictionError
            If ``prediction_index`` is not an integer, is negative, or
            is out of range for the currently loaded class set.
        """
        self.validate_prediction(prediction_index)

        if self._using_encoder:
            try:
                label = self._encoder.inverse_transform([prediction_index])[0]  # type: ignore[union-attr]
                label_str = str(label)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Encoder failed to decode index %d: %s", prediction_index, exc
                )
                raise InvalidPredictionError(
                    f"Failed to decode prediction index {prediction_index} "
                    "using the loaded label encoder."
                ) from exc
        else:
            label_str = self._classes[prediction_index]

        logger.debug(
            "Decoded prediction index %d -> label %r (source=%s).",
            prediction_index,
            label_str,
            "encoder" if self._using_encoder else "fallback_mapping",
        )
        return label_str

    def validate_prediction(self, prediction_index: int) -> None:
        """Validate that a numeric prediction index is decodable.

        Parameters
        ----------
        prediction_index:
            The candidate prediction index to validate.

        Raises
        ------
        InvalidPredictionError
            If ``prediction_index`` is not an integer (``bool`` is
            rejected too, since it is not a meaningful class index),
            is negative, or is greater than or equal to the number of
            known classes.
        """
        if isinstance(prediction_index, bool) or not isinstance(prediction_index, int):
            logger.error(
                "Invalid prediction index type: %s (%r).",
                type(prediction_index).__name__,
                prediction_index,
            )
            raise InvalidPredictionError(
                f"'prediction_index' must be an int; got "
                f"{type(prediction_index).__name__} ({prediction_index!r})."
            )

        if prediction_index < 0:
            logger.error("Negative prediction index received: %d.", prediction_index)
            raise InvalidPredictionError(
                f"'prediction_index' must be non-negative; got {prediction_index}."
            )

        num_classes = len(self._classes)
        if prediction_index >= num_classes:
            logger.error(
                "Out-of-range prediction index received: %d (num_classes=%d).",
                prediction_index,
                num_classes,
            )
            raise InvalidPredictionError(
                f"'prediction_index' ({prediction_index}) is out of range "
                f"for the current class set of size {num_classes}."
            )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def get_classes(self) -> list[str]:
        """Return the full, index-ordered list of known class labels.

        Returns
        -------
        list[str]
            All currently known gesture labels, ordered by class index.
        """
        return list(self._classes)

    def num_classes(self) -> int:
        """Return the number of known classes in the active label set.

        Returns
        -------
        int
            The number of distinct gesture labels currently loaded.
        """
        return len(self._classes)

    def is_using_encoder(self) -> bool:
        """Return whether label decoding is backed by a fitted encoder.

        Returns
        -------
        bool
            ``True`` if a ``LabelEncoder`` (or equivalent) was
            successfully loaded from ``encoder_path``; ``False`` if the
            direct fallback class mapping is in use.
        """
        return self._using_encoder
>>>>>>> origin/p3-dev
