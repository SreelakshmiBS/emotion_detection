import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt

model = load_model("model.h5")

emotion_labels = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Neutral',
    'Sad',
    'Surprise'
]
# =========================
# IMAGE PATH
# =========================

file_path = "dataset\\test\\fearful\\im70.png"

# =========================
# READ IMAGE
# =========================

image = cv2.imread(file_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resize image
resized = cv2.resize(gray, (48,48))

# Normalize image
normalized = resized / 255.0

# Reshape image
reshaped = np.reshape(normalized, (1,48,48,1))

# =========================
# PREDICT EMOTION
# =========================

prediction = model.predict(reshaped)

emotion = emotion_labels[np.argmax(prediction)]

print("\nPredicted Emotion:", emotion)

# =========================
# DISPLAY IMAGE
# =========================

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.title(f"Predicted Emotion: {emotion}")

plt.axis("off")

plt.show()