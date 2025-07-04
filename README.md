# 🥊 Real-Time Fight Detection System

A real-time fight detection system using YOLOv8 and a fine-tuned MobileNet model. This project detects physical fights in live webcam feeds or pre-recorded video footage and presents a user-friendly interface powered by Streamlit.

---

## 🔍 Overview

This project combines object detection (YOLOv8) with a custom-trained MobileNet-based fight classifier to accurately identify fights in real time. It leverages:

- **YOLOv8**: For real-time person detection.
- **MobileNet (fine-tuned)**: For classifying whether the detected activity is a fight or not.
- **OpenCV**: For video processing and frame-by-frame analysis.
- **TensorFlow**: For loading and running the trained deep learning model.
- **Streamlit**: To create an intuitive UI for users to run and interact with the model easily.

---

## 📦 Features

- 🔴 Live webcam fight detection
- 🎞️ Video file input support
- ✅ Real-time detection and alerts
- 💻 Streamlit-based user interface
- 📦 Lightweight and fast

---

## 🧠 Model Training

- MobileNet was fine-tuned using a labeled Kaggle fight/non-fight dataset.
- YOLOv8 is used as a lightweight object detector for identifying people in each frame.
- Final prediction is made based on the extracted region-of-interest (ROI) from YOLO.

---

## 🖥️ Installation

```bash
git clone https://github.com/your-username/fight-detection.git
cd fight-detection

# Create and activate virtual environment (optional)
python -m venv venv
source venv/bin/activate    # For Linux/macOS
venv\Scripts\activate       # For Windows

# Install requirements
pip install -r requirements.txt
