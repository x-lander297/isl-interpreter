import os
import numpy as np
from config.constants import DATA_PROCESSED, INPUT_DIM, STATIC_FEATURE_DIM

def merge_and_pad_static():
    static_landmarks_path = os.path.join(DATA_PROCESSED, 'static_landmarks.npy')
    static_labels_path = os.path.join(DATA_PROCESSED, 'static_labels.npy')

    if not os.path.exists(static_landmarks_path):
        print("❌ static_landmarks.npy not found. Run P1's static_pipeline.py first.")
        return

    X_static = np.load(static_landmarks_path)
    y_static = np.load(static_labels_path)

    # Pad static features (63 → 126) with zeros
    pad_width = INPUT_DIM - STATIC_FEATURE_DIM  # 63
    X_static_padded = np.pad(X_static, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)

    # For now, only static exists. In future, we'll load dynamic and stack.
    X_combined = X_static_padded
    y_combined = y_static

    # Save combined
    np.save(os.path.join(DATA_PROCESSED, 'X_combined.npy'), X_combined)
    np.save(os.path.join(DATA_PROCESSED, 'y_combined.npy'), y_combined)

    print(f"✅ Merged dataset saved.")
    print(f"   X_combined shape: {X_combined.shape} (expected: (n, 126))")
    print(f"   y_combined shape: {y_combined.shape}")
    print(f"   Number of classes: {len(np.unique(y_combined))}")

if __name__ == '__main__':
    merge_and_pad_static()