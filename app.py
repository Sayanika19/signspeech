import cv2
import numpy as np
import time
import pyttsx3
from keras.models import model_from_json
from function import extract_keypoints, actions
import mediapipe as mp

# Load the trained model
with open("model.json", "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights("model.keras")

# Initialize the text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)

# Setup recognition variables
sequence = []
sentence = []
predictions = []
threshold = 0.9  # Higher confidence to avoid wrong guesses
word = ""
waiting_for_next_word = False

# Define gestures for space and backspace
SPACE_GESTURE = "OPEN_PALM"
BACKSPACE_GESTURE = "CLOSED_FIST"

# Open the webcam
cap = cv2.VideoCapture(0)

# Initialize Mediapipe hands model
mp_hands = mp.solutions.hands
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Define the active detection region
        cropframe = frame[40:400, 0:300]
        frame = cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)

        # Extract keypoints
        keypoints = extract_keypoints(cropframe, hands)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        try:
            # Predict only when the sequence is complete
            if len(sequence) == 30 and not waiting_for_next_word:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predicted_index = np.argmax(res)
                predicted_label = actions[predicted_index]

                # ✅ Handle space gesture (open palm)
                if predicted_label == SPACE_GESTURE:
                    word += " "
                    sentence.append(" ")
                    print("✅ Space added!")
                    waiting_for_next_word = True

                # ✅ Handle backspace gesture (closed fist)
                elif predicted_label == BACKSPACE_GESTURE:
                    word = word[:-1]  # Remove last character
                    sentence.append("BACKSPACE")
                    print("❌ Backspace pressed!")
                    waiting_for_next_word = True

                # ✅ Stop wrong guesses (higher confidence filter)
                elif res[predicted_index] > threshold and predicted_label not in sentence[-1:]:
                    word += f"{predicted_label}"
                    sentence.append(predicted_label)
                    print(f"✅ Letter added: {predicted_label}")

                    waiting_for_next_word = True

                # Reset pause after each word/gesture
                if waiting_for_next_word:
                    time.sleep(1)  # Small pause between gestures
                    waiting_for_next_word = False

                # Keep only the last few predictions
                if len(sentence) > 5:
                    sentence = sentence[-5:]

        except Exception as e:
            print(f"Error: {e}")

        # 🎧 Speak the completed word when done
        if len(word) > 0 and waiting_for_next_word:
            print(f"🔊 Speaking: {word.strip()}")
            engine.say(word.strip())
            engine.runAndWait()
            word = ""  # Reset the word after speaking

        # Display the current word output
        cv2.rectangle(frame, (0, 0), (350, 40), (245, 117, 16), -1)
        cv2.putText(frame, f"Word: {word}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show the live camera feed
        cv2.imshow('Sign Language Recognition', frame)

        # 🛠️ Controls: Quit or Reset
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):  # Quit the app
            break
        if key == ord('r'):  # Reset the sentence
            sentence = []
            word = ""

# Cleanup when done
cap.release()
cv2.destroyAllWindows()
