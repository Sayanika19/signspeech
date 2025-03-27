import cv2
import numpy as np
import os
import mediapipe as mp

# Initialize Mediapipe utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Mediapipe Detection Function
def mediapipe_detection(image, model):
    """Process an image and detect hands with Mediapipe."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Draw landmarks on hand detection
def draw_styled_landmarks(image, results):
    """Draw styled Mediapipe hand landmarks on the image."""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

# Extract hand keypoints from an image
def extract_keypoints(image, hands_model):
    """Extract keypoints from hand landmarks."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = hands_model.process(image_rgb)
    image_rgb.flags.writeable = True

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            rh = np.array(
                [[res.x, res.y, res.z] for res in hand_landmarks.landmark]
            ).flatten()
            return np.concatenate([rh])

    return np.zeros(63)

# Setup data path and sequence details
DATA_PATH = os.path.join("MP_Data")
actions = np.array([chr(i) for i in range(65, 91)])  # A-Z

no_sequences = 30
sequence_length = 30
