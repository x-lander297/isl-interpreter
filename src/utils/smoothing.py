"""
smoothing.py
============

Reusable prediction-smoothing utilities for real-time sign-language
inference pipelines (e.g., an XGBoost-based Indian Sign Language
Interpreter).

This module currently provides temporal majority-vote smoothing over a
sliding window of per-frame predictions, which helps stabilize noisy
frame-by-frame classifier output before it is surfaced to downstream
consumers (UI, TTS, gesture-sequence logic, etc.).

The module is intentionally dependency-free (Python standard library
only) so it can be reused across different classifiers and pipelines,
including future dynamic-gesture modules that may need similar
temporal-smoothing primitives.

Example
-------
>>> from collections import deque
>>> from src.utils.smoothing import majority_vote
>>> window = deque(["A", "A", "B", "A"], maxlen=5)
>>> majority_vote(window)
'A'
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from typing import Hashable, Optional, TypeVar

__all__ = ["majority_vote"]

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
# Library modules should not configure logging handlers themselves; they
# should attach a NullHandler so that, absent any application-level logging
# configuration, no "No handlers could be found" warnings are emitted. The
# hosting application (e.g., the real-time inference loop) is expected to
# configure logging (handlers, levels, formatters) as appropriate.
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

# A prediction label must be hashable (so it can be counted) and, in
# practice, is typically a str class name or an int class index.
PredictionT = TypeVar("PredictionT", bound=Hashable)


def majority_vote(predictions: Sequence[PredictionT]) -> Optional[PredictionT]:
    """Return the most frequent prediction in a sequence of predictions.

    This function is designed for temporal smoothing of per-frame
    classifier outputs (e.g., XGBoost class predictions) collected over a
    sliding window during real-time inference. It counts occurrences of
    each distinct prediction in ``predictions`` and returns the one with
    the highest frequency.

    Parameters
    ----------
    predictions:
        A :class:`~collections.abc.Sequence` of hashable prediction
        labels. Supported concrete types include ``list``, ``tuple``, and
        ``collections.deque`` (as well as any other object satisfying the
        :class:`~collections.abc.Sequence` protocol, i.e. supporting
        ``__len__`` and ``__iter__``/``__getitem__``).

    Returns
    -------
    Optional[PredictionT]
        The most frequent prediction in ``predictions``. If several
        predictions are tied for the highest frequency, the one that
        occurs **earliest** in ``predictions`` is returned, guaranteeing
        a deterministic result independent of dictionary/hash ordering.
        Returns ``None`` if ``predictions`` is empty.

    Raises
    ------
    TypeError
        If ``predictions`` is not a :class:`~collections.abc.Sequence`
        (e.g., a bare generator, ``set``, or scalar value), or if it
        contains one or more unhashable elements (e.g., ``list`` or
        ``dict`` items).

    Examples
    --------
    >>> majority_vote(["A", "B", "A", "C", "A"])
    'A'
    >>> majority_vote(("hello", "hello", "world"))
    'hello'
    >>> from collections import deque
    >>> majority_vote(deque(["X", "Y"], maxlen=2))  # tie -> earliest wins
    'X'
    >>> majority_vote([]) is None
    True
    """
    _validate_sequence(predictions)

    length = len(predictions)
    if length == 0:
        logger.debug("majority_vote received an empty sequence; returning None.")
        return None

    try:
        counts = Counter(predictions)
    except TypeError as exc:
        # Raised by Counter/hash() if an element is unhashable.
        logger.error(
            "majority_vote received unhashable prediction elements: %s", exc
        )
        raise TypeError(
            "All elements of 'predictions' must be hashable to be counted."
        ) from exc

    max_count = max(counts.values())

    # Deterministic tie-break: earliest occurrence in the original sequence
    # among labels that share the maximum count.
    winner: Optional[PredictionT] = None
    for item in predictions:
        if counts[item] == max_count:
            winner = item
            break

    logger.debug(
        "majority_vote resolved winner=%r out of %d samples (max_count=%d, "
        "distinct_labels=%d).",
        winner,
        length,
        max_count,
        len(counts),
    )
    return winner


def _validate_sequence(predictions: object) -> None:
    """Validate that ``predictions`` is a proper :class:`Sequence`.

    Parameters
    ----------
    predictions:
        The object to validate.

    Raises
    ------
    TypeError
        If ``predictions`` is ``None`` or does not satisfy the
        :class:`~collections.abc.Sequence` protocol (e.g., a raw
        generator, ``set``, or a scalar value). Strings are technically
        ``Sequence`` instances but are rejected here because a single
        string is almost certainly a misuse of this API (a single
        prediction label, not a collection of them).
    """
    if predictions is None:
        logger.error("majority_vote received None instead of a sequence.")
        raise TypeError("'predictions' must not be None.")

    if isinstance(predictions, str):
        logger.error(
            "majority_vote received a raw string (%r); expected a sequence "
            "of prediction labels, not a single string.",
            predictions,
        )
        raise TypeError(
            "'predictions' must be a Sequence of labels (list, tuple, "
            "deque, etc.), not a single str."
        )

    if not isinstance(predictions, Sequence):
        logger.error(
            "majority_vote received unsupported type %s; expected a "
            "Sequence (list, tuple, deque, etc.).",
            type(predictions).__name__,
        )
        raise TypeError(
            "'predictions' must be a Sequence (list, tuple, deque, etc.); "
            f"got {type(predictions).__name__!r}."
        )