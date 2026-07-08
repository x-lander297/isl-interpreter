"""
src/models/predict.py

Prediction module for the Indian Sign Language Interpreter.

Loads a trained XGBoost classifier and its corresponding LabelEncoder to
translate 126-dimensional hand-landmark feature vectors (42 landmarks x 3
coordinates) into predicted sign classes (A-Z, 0-9).
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import joblib
import numpy as np

try:
    from config.constants import MODEL_PATH, INPUT_DIM
except ImportError:
    MODEL_PATH = "models/xgb_model.pkl"
    INPUT_DIM = 126

DEFAULT_MODEL_PATH = "models/xgb_model.pkl"
DEFAULT_ENCODER_PATH = "models/label_encoder.pkl"
LOW_CONFIDENCE_THRESHOLD = 0.5

logger = logging.getLogger(__name__)


class SignPredictor:
    """
    Predicts Indian Sign Language characters (A-Z, 0-9) from hand-landmark
    feature vectors using a pretrained XGBoost model.

    This class is stateless with respect to prediction calls (no shared
    mutable state is mutated across calls), making instances safe to use
    concurrently from multiple threads as long as the underlying model
    object's `predict`/`predict_proba` methods are themselves thread-safe
    (true for XGBoost's sklearn API in inference mode).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        encoder_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the SignPredictor by loading the trained model and
        label encoder from disk.

        Args:
            model_path: Path to the serialized XGBoost model (.pkl).
                Defaults to config.constants.MODEL_PATH, falling back to
                'models/xgb_model.pkl'.
            encoder_path: Path to the serialized LabelEncoder (.pkl).
                Defaults to 'models/label_encoder.pkl'.

        Raises:
            FileNotFoundError: If the model or encoder file does not exist.
            RuntimeError: If loading either artifact fails for any other
                reason.
        """
        self.model_path = model_path or MODEL_PATH or DEFAULT_MODEL_PATH
        self.encoder_path = encoder_path or DEFAULT_ENCODER_PATH
        self.input_dim = INPUT_DIM

        self.model = self._load_artifact(self.model_path, "model")
        self.label_encoder = self._load_artifact(self.encoder_path, "label encoder")

        logger.info(
            "SignPredictor initialized. model=%s encoder=%s classes=%d",
            self.model_path,
            self.encoder_path,
            len(getattr(self.label_encoder, "classes_", [])),
        )

    @staticmethod
    def _load_artifact(path: str, description: str):
        """
        Load a joblib-serialized artifact from disk with validation.

        Args:
            path: Filesystem path to the artifact.
            description: Human-readable name used in log/error messages.

        Returns:
            The deserialized object.

        Raises:
            FileNotFoundError: If the file does not exist.
            RuntimeError: If deserialization fails.
        """
        artifact_path = Path(path)
        if not artifact_path.is_file():
            logger.error("%s file not found at path: %s", description, path)
            raise FileNotFoundError(f"{description} file not found at path: {path}")

        try:
            artifact = joblib.load(artifact_path)
        except Exception as exc:
            logger.error("Failed to load %s from %s: %s", description, path, exc)
            raise RuntimeError(f"Failed to load {description} from {path}") from exc

        logger.info("Loaded %s from %s", description, path)
        return artifact

    def _validate_and_reshape(self, landmarks: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Validate a single sample's shape and reshape it for XGBoost.

        Args:
            landmarks: A 1D sequence of 126 floats, or an array already
                shaped (1, 126).

        Returns:
            A numpy array of shape (1, 126) and dtype float32.

        Raises:
            ValueError: If the input is not convertible to a (1, 126) array.
        """
        arr = np.asarray(landmarks, dtype=np.float32)

        if arr.ndim == 1:
            if arr.shape[0] != self.input_dim:
                raise ValueError(
                    f"Expected {self.input_dim}-dimensional landmarks, "
                    f"got shape {arr.shape}"
                )
            arr = arr.reshape(1, self.input_dim)
        elif arr.ndim == 2:
            if arr.shape != (1, self.input_dim):
                raise ValueError(
                    f"Expected shape (1, {self.input_dim}) for a single "
                    f"sample, got shape {arr.shape}"
                )
        else:
            raise ValueError(
                f"Expected a 1D or 2D array for a single sample, got "
                f"{arr.ndim} dimensions"
            )

        return arr

    def _validate_and_reshape_batch(
        self, landmarks_batch: Union[List[List[float]], np.ndarray]
    ) -> np.ndarray:
        """
        Validate a batch's shape for XGBoost.

        Args:
            landmarks_batch: A 2D sequence of shape (N, 126).

        Returns:
            A numpy array of shape (N, 126) and dtype float32.

        Raises:
            ValueError: If the input is not convertible to an (N, 126) array.
        """
        arr = np.asarray(landmarks_batch, dtype=np.float32)

        if arr.ndim != 2:
            raise ValueError(
                f"Expected a 2D array for batch prediction, got "
                f"{arr.ndim} dimensions"
            )
        if arr.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim}-dimensional landmarks per "
                f"sample, got shape {arr.shape}"
            )
        if arr.shape[0] == 0:
            raise ValueError("Batch is empty; at least one sample is required")

        return arr

    def _decode_labels(self, indices: np.ndarray) -> List[str]:
        """
        Map predicted class indices to their character labels.

        Args:
            indices: Array of integer class indices.

        Returns:
            List of decoded character labels (e.g. 'A', '7').
        """
        try:
            return list(self.label_encoder.inverse_transform(indices))
        except Exception as exc:
            logger.error("Failed to decode predicted labels: %s", exc)
            raise RuntimeError("Failed to decode predicted labels") from exc

    def predict(
        self,
        landmarks: Union[List[float], np.ndarray],
        return_confidence: bool = False,
    ) -> Union[str, Tuple[str, float]]:
        """
        Predict the sign class for a single landmark sample.

        Args:
            landmarks: A 126-dimensional feature vector (list or ndarray).
            return_confidence: If True, also return the prediction
                confidence (max class probability).

        Returns:
            The predicted character, or a (character, confidence) tuple
            if `return_confidence` is True.

        Raises:
            ValueError: If the input shape is invalid.
            RuntimeError: If the underlying model fails to predict.
        """
        sample = self._validate_and_reshape(landmarks)

        try:
            if return_confidence:
                probabilities = self.model.predict_proba(sample)[0]
                pred_index = int(np.argmax(probabilities))
                confidence = float(probabilities[pred_index])
            else:
                pred_index = int(self.model.predict(sample)[0])
                confidence = None
        except Exception as exc:
            logger.error("Prediction failed: %s", exc)
            raise RuntimeError("XGBoost prediction failed") from exc

        label = self._decode_labels(np.array([pred_index]))[0]

        if confidence is not None:
            logger.debug("Predicted '%s' with confidence %.4f", label, confidence)
            if confidence < LOW_CONFIDENCE_THRESHOLD:
                logger.warning(
                    "Low confidence prediction: '%s' (%.4f) below threshold %.2f",
                    label,
                    confidence,
                    LOW_CONFIDENCE_THRESHOLD,
                )
            return label, confidence

        logger.debug("Predicted '%s'", label)
        return label

    def predict_batch(
        self,
        landmarks_batch: Union[List[List[float]], np.ndarray],
        return_confidence: bool = False,
    ) -> Union[List[str], Tuple[List[str], List[float]]]:
        """
        Predict sign classes for a batch of landmark samples.

        Args:
            landmarks_batch: A sequence of 126-dimensional feature vectors,
                shape (N, 126).
            return_confidence: If True, also return per-sample confidences.

        Returns:
            A list of predicted characters, or a (labels, confidences)
            tuple if `return_confidence` is True.

        Raises:
            ValueError: If the input shape is invalid.
            RuntimeError: If the underlying model fails to predict.
        """
        batch = self._validate_and_reshape_batch(landmarks_batch)

        try:
            if return_confidence:
                probabilities = self.model.predict_proba(batch)
                pred_indices = np.argmax(probabilities, axis=1)
                confidences = probabilities[np.arange(len(pred_indices)), pred_indices]
            else:
                pred_indices = np.asarray(self.model.predict(batch), dtype=int)
                confidences = None
        except Exception as exc:
            logger.error("Batch prediction failed: %s", exc)
            raise RuntimeError("XGBoost batch prediction failed") from exc

        labels = self._decode_labels(pred_indices)

        if confidences is not None:
            for label, confidence in zip(labels, confidences):
                logger.debug("Predicted '%s' with confidence %.4f", label, confidence)
                if confidence < LOW_CONFIDENCE_THRESHOLD:
                    logger.warning(
                        "Low confidence prediction: '%s' (%.4f) below threshold %.2f",
                        label,
                        confidence,
                        LOW_CONFIDENCE_THRESHOLD,
                    )
            logger.debug("Batch prediction complete for %d samples", len(labels))
            return labels, [float(c) for c in confidences]

        logger.debug("Batch prediction complete for %d samples", len(labels))
        return labels

    def predict_proba(self, landmarks: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Compute the full class probability distribution for a single sample.

        Args:
            landmarks: A 126-dimensional feature vector (list or ndarray).

        Returns:
            A 1D numpy array of probabilities, ordered to match
            `self.label_encoder.classes_`.

        Raises:
            ValueError: If the input shape is invalid.
            RuntimeError: If the underlying model fails to predict.
        """
        sample = self._validate_and_reshape(landmarks)

        try:
            probabilities = self.model.predict_proba(sample)[0]
        except Exception as exc:
            logger.error("predict_proba failed: %s", exc)
            raise RuntimeError("XGBoost predict_proba failed") from exc

        logger.debug("Computed probability distribution over %d classes", len(probabilities))
        return probabilities

    def get_top_k(
        self, landmarks: Union[List[float], np.ndarray], k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Get the top-K most probable sign classes for a single sample.

        Args:
            landmarks: A 126-dimensional feature vector (list or ndarray).
            k: Number of top predictions to return. Must be a positive
                integer no greater than the number of known classes.

        Returns:
            A list of (character, confidence) tuples sorted by descending
            confidence.

        Raises:
            ValueError: If the input shape is invalid or `k` is invalid.
            RuntimeError: If the underlying model fails to predict.
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")

        probabilities = self.predict_proba(landmarks)

        num_classes = len(probabilities)
        effective_k = min(k, num_classes)
        if effective_k < k:
            logger.warning(
                "Requested top-%d but only %d classes are available; "
                "returning %d results",
                k,
                num_classes,
                effective_k,
            )

        top_indices = np.argsort(probabilities)[::-1][:effective_k]
        labels = self._decode_labels(top_indices)
        results = [
            (label, float(probabilities[idx]))
            for label, idx in zip(labels, top_indices)
        ]

        logger.debug("Top-%d predictions: %s", effective_k, results)
        return results
