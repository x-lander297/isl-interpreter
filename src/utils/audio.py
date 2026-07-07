"""
audio.py
========

Thread-safe text-to-speech utility for the Indian Sign Language (ISL)
Interpreter.

This module wraps ``pyttsx3`` to provide spoken feedback for recognized
static gestures, with per-prediction cooldown suppression (to avoid
repeatedly re-announcing a gesture that is being held in front of the
camera), mute/unmute controls, and safe handling of missing audio
devices or engine failures.

This module is intentionally independent of the rest of the inference
pipeline: it contains **no webcam code, no MediaPipe code, no model
loading, and no prediction logic** -- it only speaks whatever text
string it is given.

Example
-------
>>> from src.utils.audio import TextToSpeech
>>> tts = TextToSpeech()
>>> tts.speak("Hello")  # doctest: +SKIP
True
>>> tts.mute()
>>> tts.is_muted()
True
>>> tts.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

try:
    import pyttsx3
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "The 'pyttsx3' package is required by src.utils.audio but is not "
        "installed."
    ) from exc

from config.constants import MUTE_DEFAULT, SPEECH_COOLDOWN_SECONDS

__all__ = ["TextToSpeech", "AudioError"]

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


class AudioError(RuntimeError):
    """Raised for unrecoverable text-to-speech engine failures.

    Note that :class:`TextToSpeech` deliberately avoids raising this for
    routine, expected failure modes (e.g. missing audio device at
    startup, or a transient speak error) so that audio feedback remains
    a best-effort subsystem that never crashes the calling real-time
    inference loop. It is reserved for programming errors such as
    invalid configuration values passed at construction time.
    """


class TextToSpeech:
    """Thread-safe text-to-speech utility for recognized ISL predictions.

    ``TextToSpeech`` wraps a single ``pyttsx3`` engine instance and
    provides:

    - Non-blocking, thread-safe spoken output (``speak``).
    - Per-text cooldown suppression, so the same recognized sign is not
      re-announced on every frame while it is held in view.
    - Mute / unmute / toggle-mute controls.
    - Safe handling of missing audio devices, engine initialization
      failures, and runtime speech errors -- none of which raise out of
      the public API during normal operation.

    Parameters
    ----------
    cooldown_seconds:
        Minimum time, in seconds, that must elapse between two
        consecutive spoken announcements of the *same* text. Defaults
        to ``config.constants.SPEECH_COOLDOWN_SECONDS``.
    muted:
        Initial mute state. Defaults to
        ``config.constants.MUTE_DEFAULT``.
    rate:
        Optional speech rate (words per minute) to configure on the
        underlying engine. If ``None``, the engine's default rate is
        used.
    volume:
        Optional speech volume, in ``[0.0, 1.0]``, to configure on the
        underlying engine. If ``None``, the engine's default volume is
        used.

    Notes
    -----
    This class is designed for straightforward unit testing: engine
    construction is isolated in :meth:`_initialize_engine`, and all
    engine access is funneled through a single lock, so a test double
    can be substituted for ``pyttsx3.init`` if desired (e.g. by
    monkeypatching the ``pyttsx3`` module before construction).
    """

    def __init__(
        self,
        cooldown_seconds: float = SPEECH_COOLDOWN_SECONDS,
        muted: bool = MUTE_DEFAULT,
        rate: Optional[int] = None,
        volume: Optional[float] = None,
    ) -> None:
        if cooldown_seconds < 0:
            raise AudioError(
                f"'cooldown_seconds' must be non-negative; got {cooldown_seconds}."
            )
        if volume is not None and not 0.0 <= volume <= 1.0:
            raise AudioError(f"'volume' must be in [0.0, 1.0]; got {volume}.")

        self._cooldown_seconds: float = cooldown_seconds
        self._rate = rate
        self._volume = volume

        # Guards engine access, mute state, and cooldown bookkeeping so
        # concurrent prediction events cannot corrupt shared state or
        # collide inside the (not inherently thread-safe) pyttsx3 engine.
        self._lock = threading.RLock()

        self._muted: bool = muted
        self._last_spoken: dict[str, float] = {}

        self._engine: Optional["pyttsx3.Engine"] = self._initialize_engine()

        logger.info(
            "TextToSpeech initialized: cooldown_seconds=%.2f, muted=%s, "
            "engine_available=%s.",
            self._cooldown_seconds,
            self._muted,
            self._engine is not None,
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _initialize_engine(self) -> Optional["pyttsx3.Engine"]:
        """Safely construct and configure the underlying ``pyttsx3`` engine.

        Returns
        -------
        Optional[pyttsx3.Engine]
            The initialized engine, or ``None`` if initialization failed
            (e.g. no audio device is available on this host).
        """
        try:
            engine = pyttsx3.init()
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to initialize pyttsx3 speech engine; audio feedback "
                "will be disabled. Reason: %s",
                exc,
            )
            return None

        if engine is None:
            logger.error(
                "pyttsx3.init() returned None; audio feedback will be "
                "disabled (no audio device or driver found)."
            )
            return None

        try:
            if self._rate is not None:
                engine.setProperty("rate", self._rate)
            if self._volume is not None:
                engine.setProperty("volume", self._volume)
        except Exception as exc:  # noqa: BLE001 - non-fatal; engine still usable
            logger.warning(
                "Failed to apply custom rate/volume settings to speech "
                "engine; continuing with engine defaults. Reason: %s",
                exc,
            )

        return engine

    # ------------------------------------------------------------------
    # Speech
    # ------------------------------------------------------------------
    def speak(self, text: str) -> bool:
        """Speak ``text`` aloud, subject to mute state and cooldown.

        The actual utterance is dispatched to a background daemon
        thread, so this method returns quickly and does not block the
        calling (e.g. real-time inference) loop. Engine access across
        concurrent calls is serialized via an internal lock.

        Parameters
        ----------
        text:
            The text to speak (typically a recognized ISL gesture
            label).

        Returns
        -------
        bool
            ``True`` if a speech request was accepted and dispatched;
            ``False`` if the request was rejected or suppressed
            (invalid input, engine unavailable, muted, or still within
            the per-text cooldown window).
        """
        if not self._validate_text(text):
            return False

        with self._lock:
            if self._muted:
                logger.debug("speak() suppressed: currently muted. text=%r", text)
                return False

            if self._engine is None:
                logger.warning(
                    "speak() suppressed: speech engine is unavailable. text=%r",
                    text,
                )
                return False

            now = time.monotonic()
            last_time = self._last_spoken.get(text)
            if last_time is not None and (now - last_time) < self._cooldown_seconds:
                remaining = self._cooldown_seconds - (now - last_time)
                logger.debug(
                    "speak() suppressed by cooldown: text=%r, %.2fs remaining.",
                    text,
                    remaining,
                )
                return False

            # Reserve the slot immediately so rapid-fire calls for the same
            # text (from concurrent prediction events) don't all pass the
            # cooldown check before any of them finishes speaking.
            self._last_spoken[text] = now

        thread = threading.Thread(
            target=self._speak_worker, args=(text,), daemon=True, name="tts-speak"
        )
        thread.start()
        logger.info("Speech request dispatched: text=%r.", text)
        return True

    def _speak_worker(self, text: str) -> None:
        """Background worker that performs the actual blocking speech call.

        Parameters
        ----------
        text:
            The text to speak.
        """
        with self._lock:
            if self._engine is None:
                logger.warning(
                    "Speech worker aborted: engine became unavailable. text=%r",
                    text,
                )
                return
            try:
                self._engine.say(text)
                self._engine.runAndWait()
                logger.debug("Speech completed: text=%r.", text)
            except Exception as exc:  # noqa: BLE001 - never crash the caller
                logger.error("Runtime speech error for text=%r: %s", text, exc)

    @staticmethod
    def _validate_text(text: object) -> bool:
        """Validate that ``text`` is a non-empty, speakable string.

        Parameters
        ----------
        text:
            The candidate value to validate.

        Returns
        -------
        bool
            ``True`` if ``text`` is a non-empty (post-strip) ``str``;
            ``False`` otherwise (with a warning logged).
        """
        if not isinstance(text, str):
            logger.warning(
                "speak() rejected non-string input of type %s.",
                type(text).__name__,
            )
            return False
        if not text.strip():
            logger.warning("speak() rejected empty or whitespace-only text.")
            return False
        return True

    # ------------------------------------------------------------------
    # Mute controls
    # ------------------------------------------------------------------
    def mute(self) -> None:
        """Mute speech output.

        Subsequent calls to :meth:`speak` will be suppressed (returning
        ``False``) until :meth:`unmute` is called.
        """
        with self._lock:
            was_muted = self._muted
            self._muted = True
        if not was_muted:
            logger.info("Speech output muted.")

    def unmute(self) -> None:
        """Unmute speech output, re-enabling subsequent :meth:`speak` calls."""
        with self._lock:
            was_muted = self._muted
            self._muted = False
        if was_muted:
            logger.info("Speech output unmuted.")

    def toggle_mute(self) -> bool:
        """Toggle the current mute state.

        Returns
        -------
        bool
            The new mute state (``True`` if now muted, ``False``
            otherwise).
        """
        with self._lock:
            self._muted = not self._muted
            new_state = self._muted
        logger.info("Speech output mute toggled -> %s.", new_state)
        return new_state

    def is_muted(self) -> bool:
        """Return whether speech output is currently muted.

        Returns
        -------
        bool
            ``True`` if muted; ``False`` otherwise.
        """
        with self._lock:
            return self._muted

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Safely stop any in-progress speech and halt the engine loop.

        Safe to call even if the engine failed to initialize, or if no
        speech is currently in progress.
        """
        with self._lock:
            if self._engine is None:
                logger.debug("stop() called but no speech engine is available; no-op.")
                return
            try:
                self._engine.stop()
                logger.info("Speech engine stopped.")
            except Exception as exc:  # noqa: BLE001 - log, do not re-raise
                logger.error("Exception while stopping speech engine: %s", exc)

    def reset_cooldowns(self) -> None:
        """Clear all per-text cooldown timestamps.

        Useful when starting a new interpretation session and wanting
        immediate re-announcement of any gesture, regardless of prior
        speech history.
        """
        with self._lock:
            self._last_spoken.clear()
        logger.info("Speech cooldown history cleared.")