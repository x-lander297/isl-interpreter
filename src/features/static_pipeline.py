import os
import numpy as np
from config.constants import DATA_PROCESSED
from src.data.loader import load_static_images
from src.features.extractor import get_landmarks_from_image
from tqdm import tqdm

def extract_static_features():
    print("📂 Loading static images...")
    X_imgs, y_labels = load_static_images()
    print(f"✅ Loaded {len(X_imgs)} images.")

    X_features = []
    y_filtered = []

    print("🖐️ Extracting MediaPipe landmarks...")
    for idx, img in tqdm(enumerate(X_imgs), total=len(X_imgs)):
        feat = get_landmarks_from_image(img)
        if feat is not None:
            X_features.append(feat)
            y_filtered.append(y_labels[idx])

    X_features = np.array(X_features, dtype=np.float32)
    y_filtered = np.array(y_filtered, dtype=np.int32)

    # Save
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    np.save(os.path.join(DATA_PROCESSED, 'static_landmarks.npy'), X_features)
    np.save(os.path.join(DATA_PROCESSED, 'static_labels.npy'), y_filtered)

    print(f"✅ Extracted landmarks for {len(X_features)} samples.")
    print(f"   Feature shape: {X_features.shape} (expected: (n, 63))")
    print(f"   Saved to {DATA_PROCESSED}")

if __name__ == '__main__':
    extract_static_features()