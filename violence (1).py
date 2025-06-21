import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
IMG_SIZE = (64, 64)  # Resize frames to 64x64
FRAME_COUNT = 16  # Number of frames per video clip
BATCH_SIZE = 8
EPOCHS = 20

# Function to preprocess a single video
def preprocess_video(video_path, img_size=IMG_SIZE, frame_count=FRAME_COUNT):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame step to uniformly sample frame_count frames
    frame_step = max(1, total_frames // frame_count)
    
    for i in range(0, total_frames, frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        # Resize and normalize frame
        frame = cv2.resize(frame, img_size)
        frame = frame / 255.0  # Normalize to [0,1]
        frames.append(frame)
        if len(frames) == frame_count:
            break
    
    cap.release()
    
    # If fewer frames, pad with zeros
    while len(frames) < frame_count:
        frames.append(np.zeros((img_size[0], img_size[1], 3)))
    
    return np.array(frames[:frame_count])

# Load dataset
def load_dataset(data_dir, classes=['Violence', 'NonViolence']):
    X, y = [], []
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for video_file in tqdm(os.listdir(class_dir), desc=f"Loading {class_name}"):
            if video_file.endswith('.mp4'):
                video_path = os.path.join(class_dir, video_file)
                frames = preprocess_video(video_path)
                X.append(frames)
                y.append(label)
    return np.array(X), np.array(y)

# Define 3D CNN model
def create_model(input_shape=(FRAME_COUNT, IMG_SIZE[0], IMG_SIZE[1], 3)):
    model = Sequential([
        Conv3D(32, (3, 3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling3D((2, 2, 2)),
        Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
        MaxPooling3D((2, 2, 2)),
        Conv3D(128, (3, 3, 3), activation='relu', padding='same'),
        MaxPooling3D((2, 2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Main function
def main():
    # Set dataset path
    data_dir = r'C:\Users\Namo\fight recognization\extract\Real Life Violence Dataset'
    
    # Load and preprocess data
    print("Loading dataset...")
    X, y = load_dataset(data_dir)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    model = create_model()
    print("Training model...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1
    )
    
    # Evaluate model
    print("Evaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Save model
    model.save('violence_detection_model.h5')
    print("Model saved as 'violence_detection_model.h5'")

if __name__ == '__main__':
    main()