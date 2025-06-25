import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import os
from tempfile import NamedTemporaryFile
from ultralytics import YOLO

# -----------------------------
# Model Initialization
# -----------------------------
model = tf.keras.models.load_model("violence_Model.h5", compile=False)
yolo_model = YOLO("yolov8n (1).pt")  # Change to your specific YOLOv8 model if needed

# -----------------------------
# Constants
# -----------------------------
NUM_FRAMES = 30
FRAME_SIZE = (224, 224)
LABEL_MAP = {0: "NonViolence", 1: "Violence"}

# -----------------------------
# Helper Function: Preprocess Video
# -----------------------------
def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num=NUM_FRAMES, dtype=np.int32)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            resized = cv2.resize(frame, FRAME_SIZE)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            norm = tf.keras.applications.mobilenet_v2.preprocess_input(rgb.astype(np.float32))
            frames.append(norm)
        else:
            frames.append(np.zeros((FRAME_SIZE[0], FRAME_SIZE[1], 3), dtype=np.float32))

    cap.release()
    return np.expand_dims(np.array(frames), axis=0)

# -----------------------------
# Prediction from Uploaded Video
# -----------------------------
def predict_from_video(video_path):
    video_array = preprocess_video(video_path)
    prediction = model.predict(video_array)
    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model.predict(source=frame, conf=0.4, verbose=False)[0]
        frame = results.plot()

        # Add label
        cv2.putText(
            frame,
            f"{LABEL_MAP[predicted_class]} ({confidence:.2f})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0) if predicted_class == 0 else (0, 0, 255),
            3,
            cv2.LINE_AA
        )
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", width=480)

    cap.release()
    return LABEL_MAP[predicted_class], confidence

# -----------------------------
# Prediction from Webcam Feed
# -----------------------------
def predict_from_webcam():
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    buffer = []
    label = "Analyzing..."
    color = (255, 255, 0)

    stop_button = st.button("🛑 Stop Detection")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame, FRAME_SIZE)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        norm = tf.keras.applications.mobilenet_v2.preprocess_input(rgb.astype(np.float32))
        buffer.append(norm)

        if len(buffer) == NUM_FRAMES:
            input_array = np.expand_dims(np.array(buffer), axis=0)
            prediction = model.predict(input_array)
            predicted_class = int(np.argmax(prediction))
            confidence = float(np.max(prediction))
            label = f"{LABEL_MAP[predicted_class]} ({confidence:.2f})"
            color = (0, 255, 0) if predicted_class == 0 else (0, 0, 255)
            buffer = []

        results = yolo_model.predict(source=frame, conf=0.4, verbose=False)[0]
        frame = results.plot()

        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", width=480)

        if stop_button:
            break

    cap.release()

# -----------------------------
# Streamlit App UI
# -----------------------------
st.set_page_config(page_title="Violence Detection System", layout="wide")
st.markdown("""
    <style>
        .main-title {
            font-size: 2.5em;
            color: #ff4b4b;
            text-align: center;
            margin-bottom: 10px;
        }
        .description {
            text-align: center;
            color: #555;
            font-size: 1.1em;
            margin-bottom: 30px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🔍 Real-Time Violence Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='description'>Detect violent activity in videos or live webcam feed using deep learning models</div>", unsafe_allow_html=True)

# --- Mode Selection ---
option = st.selectbox("Select Input Source", ["Upload a Video", "Use Webcam"])

if option == "Upload a Video":
    uploaded_file = st.file_uploader("Choose an MP4 file", type=["mp4"])
    if uploaded_file:
        with NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            video_path = tmp.name

        st.video(uploaded_file)
        if st.button("🎯 Start Prediction"):
            label, confidence = predict_from_video(video_path)
            st.success(f"Prediction: {label}")
            st.info(f"Confidence Score: {confidence:.2f}")

elif option == "Use Webcam":
    st.warning("Ensure your webcam is enabled. Click below to start detection.")
    if st.button("🎥 Start Webcam Feed"):
        predict_from_webcam()
