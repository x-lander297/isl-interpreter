import os
import cv2
import numpy as np
from tqdm import tqdm
from config.constants import DATA_RAW_STATIC, LABEL_NAME_TO_INDEX

def load_static_images(data_dir=None):
    """
    Loads images from subfolders (A, B, ..., Z, 0, 1, ..., 9).
    Returns: X_imgs (list of RGB arrays), y_labels (list of ints)
    """
    if data_dir is None:
        data_dir = DATA_RAW_STATIC

    X_imgs = []
    y_labels = []
    label_map = LABEL_NAME_TO_INDEX.copy()

    for folder_name in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Skip if folder name not in our label map (safety)
        if folder_name not in label_map:
            print(f"⚠️ Skipping unknown folder: {folder_name}")
            continue

        label_idx = label_map[folder_name]
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for img_file in tqdm(image_files, desc=folder_name):
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224))
            X_imgs.append(img_resized)
            y_labels.append(label_idx)

    return np.array(X_imgs, dtype=np.uint8), np.array(y_labels, dtype=np.int32)