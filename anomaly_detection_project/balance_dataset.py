from imblearn.over_sampling import RandomOverSampler
import numpy as np

# Load features and labels
features = np.load("E:\\Duo_Project\\anomaly_detection_project\\data\\processed\\reduced_features.npy")
labels = np.load("E:\\Duo_Project\\anomaly_detection_project\\data\\processed\\reduced_labels.npy")

# Apply oversampling
ros = RandomOverSampler(random_state=42)
features_resampled, labels_resampled = ros.fit_resample(features, labels)

# Save the balanced dataset
np.save("E:\\Duo_Project\\anomaly_detection_project\\data\\processed\\balanced_features.npy", features_resampled)
np.save("E:\\Duo_Project\\anomaly_detection_project\\data\\processed\\balanced_labels.npy", labels_resampled)

print("✅ Dataset Balanced!")
from collections import Counter

# Check class distribution
label_counts = Counter(labels_resampled)
print("Balanced class distribution:", label_counts)

