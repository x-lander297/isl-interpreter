#!/usr/bin/env python3
"""
Evaluate the trained XGBoost model on a held-out test split.
"""

import logging
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from config import constants

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load combined dataset
    X = np.load(constants.X_COMBINED_PATH)
    y = np.load(constants.Y_COMBINED_PATH)

    # Split (same as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=constants.TEST_SPLIT_RATIO,
        random_state=constants.RANDOM_SEED, stratify=y
    )

    # Load model
    with open(constants.XGBOOST_MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Test Accuracy: {acc:.4f}")
    logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred, target_names=constants.STATIC_LABELS))
    logger.info("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))

if __name__ == "__main__":
    main()