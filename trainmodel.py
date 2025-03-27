import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Input
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from function import actions, DATA_PATH, no_sequences, sequence_length
import cv2

# 🛠️ Training Parameters
no_sequences = 50
sequence_length = 30
EPOCHS = 300
LEARNING_RATE = 0.001

# Ensure paths are correct
os.makedirs(DATA_PATH, exist_ok=True)

# 🎯 Data augmentation (flips, rotation, noise, brightness)
def augment_data(keypoints):
    keypoints = keypoints.reshape(sequence_length, 21, 3)
    flipped = np.flip(keypoints, axis=1)
    rotated = np.roll(keypoints, shift=1, axis=1)
    brightened = keypoints * 1.1
    noise = keypoints + np.random.normal(0, 0.05, keypoints.shape)
    return [flipped.flatten(), rotated.flatten(), brightened.flatten(), noise.flatten()]

# 🔥 Label mapping
label_map = {label: num for num, label in enumerate(actions)}

# ✅ Load and prepare data
sequences, labels = [], []
expected_shape = 63

for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            file_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            try:
                res = np.load(file_path, allow_pickle=True)
                if res.shape != (expected_shape,):
                    res = np.zeros(expected_shape)
            except:
                res = np.zeros(expected_shape)

            window.append(res)

        # Add original data
        if len(window) == sequence_length:
            sequences.append(window)
            labels.append(label_map[action])
            
            # Augmented data
            for aug_data in augment_data(np.array(window).flatten()):
                augmented_window = np.array(aug_data).reshape(sequence_length, expected_shape)
                sequences.append(augmented_window)
                labels.append(label_map[action])

# 🎯 Convert to arrays
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# 🛠️ Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# 🚀 Define improved model architecture
model = Sequential([
    Input(shape=(sequence_length, X.shape[2])),
    Bidirectional(LSTM(64, return_sequences=True, activation='relu')),
    Dropout(0.2),
    BatchNormalization(),
    Bidirectional(LSTM(128, return_sequences=True, activation='relu')),
    Dropout(0.3),
    BatchNormalization(),
    LSTM(64, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(32, activation='relu'),
    Dense(len(actions), activation='softmax')
])

# 🛠️ Compile the model (Fixed loss function)
optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# 📉 Learning rate scheduler
lr_scheduler = ReduceLROnPlateau(
    monitor="categorical_accuracy",
    factor=0.5,
    patience=10,
    min_lr=1e-5,
    verbose=1,
)

# 📌 TensorBoard logging
log_dir = os.path.join('Logs')
os.makedirs(log_dir, exist_ok=True)
tb_callback = TensorBoard(log_dir=log_dir)

# 🚀 Train model
model.fit(X_train, y_train, epochs=EPOCHS, callbacks=[tb_callback, lr_scheduler], verbose=1)

# 💾 Save model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save('model.keras')  # Save in Keras format
print("✅ Model retrained and saved as 'model.keras'")
