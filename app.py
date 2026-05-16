import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Emotion Detection App",
    layout="centered"
)

# =========================
# TITLE
# =========================

st.title("Emotion Detection using CNN")
st.write("Upload an image and the AI model will predict the emotion.")

# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_emotion_model():
    return load_model("model.h5")

model = load_emotion_model()

# =========================
# EMOTION LABELS
# =========================

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
# LOAD FACE CASCADE
# =========================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================
# FILE UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PROCESS IMAGE
# =========================

if uploaded_file is not None:

    # Convert image to RGB
    image = Image.open(uploaded_file).convert("RGB")

    # Convert PIL image to NumPy array
    image_np = np.array(image)

    # Convert RGB to BGR for OpenCV
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # Improve brightness/contrast
    gray = cv2.equalizeHist(gray)

    # =========================
    # FACE DETECTION
    # =========================

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(30, 30)
    )

    st.write(f"Faces Detected: {len(faces)}")

    # =========================
    # IF NO FACE DETECTED
    # =========================

    if len(faces) == 0:
        st.error("No face detected. Upload a clear frontal face image.")

    # =========================
    # PREDICT EMOTION
    # =========================

    for (x, y, w, h) in faces:

        # Crop face
        face = gray[y:y+h, x:x+w]

        # Resize to model input size
        face = cv2.resize(face, (48, 48))

        # Normalize
        face = face.astype("float32") / 255.0

        # Reshape
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)

        # Predict emotion
        prediction = model.predict(face, verbose=0)

        # Get emotion label
        emotion = emotion_labels[np.argmax(prediction)]

        # Confidence score
        confidence = np.max(prediction) * 100

        # =========================
        # DRAW RESULT
        # =========================

        # Rectangle around face
        cv2.rectangle(
            image_cv,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        # Text label
        text = f"{emotion} ({confidence:.2f}%)"

        cv2.putText(
            image_cv,
            text,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # =========================
    # DISPLAY RESULT
    # =========================

    result_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    st.image(
        result_image,
        caption="Emotion Prediction Result",
        use_container_width=True
    )