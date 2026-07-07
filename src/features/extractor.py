import cv2
import numpy as np
import mediapipe as mp
from config.constants import NUM_HAND_LANDMARKS, COORDS_PER_LANDMARK, CONFIDENCE_THRESHOLD

# Initialize MediaPipe once (singleton)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=CONFIDENCE_THRESHOLD
)

def get_landmarks_from_image(img_rgb):
    """
    Expects RGB image (H, W, 3). Returns 63-dim vector or None.
    """
    results = hands.process(img_rgb)
    if not results.multi_hand_landmarks:
        return None
    landmarks = results.multi_hand_landmarks[0].landmark
    feat = []
    for lm in landmarks:
        feat.extend([lm.x, lm.y, lm.z])
    return np.array(feat, dtype=np.float32)

def get_landmarks_from_frame(frame_bgr):
    """
    Convenience wrapper: converts BGR → RGB internally.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return get_landmarks_from_image(rgb)