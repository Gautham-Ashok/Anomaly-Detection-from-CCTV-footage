import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.decomposition import PCA

# Load trained LSTM model
model = tf.keras.models.load_model(r"E:\Duo_Project\anomaly_detection_project\model\lstm_model.keras")

# Load ResNet50 for feature extraction
resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")


# Function to extract frames from video
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))  # Resize for ResNet50
        frames.append(frame)
    cap.release()
    return np.array(frames)


# Function to extract and reduce features
def extract_features(frames):
    frames = preprocess_input(frames)
    features = resnet_model.predict(frames)

    # Reduce features using PCA
    pca = PCA(n_components=10)
    features_reduced = pca.fit_transform(features)

    return features_reduced


# Test video
video_path = r"E:\Duo_Project\data set\Explosion\Explosion018_x264.mp4"
frames = extract_frames(video_path)
features = extract_features(frames)

# Reshape for LSTM
features = features.reshape(features.shape[0], 1, features.shape[1])

# Predict anomalies
predictions = model.predict(features)
predictions = (predictions > 0.5).astype(int)  # Convert to binary (0 = normal, 1 = anomaly)

# Overlay predictions on video
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(r"E:\Duo_Project\test_videos\output_video.avi", fourcc, 20.0, (224, 224))

for i in range(len(frames)):
    frame = frames[i]
    label = "Anomaly" if predictions[i] == 1 else "Normal"
    color = (0, 0, 255) if predictions[i] == 1 else (0, 255, 0)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    out.write(frame)

cap.release()
out.release()

print("✅ Process complete! Check 'output_video.avi' for results.")
