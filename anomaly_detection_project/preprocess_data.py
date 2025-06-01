import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import Counter

# Load features and labels
features = np.load("E:\\Duo_Project\\anomaly_detection_project\\data\\processed\\balanced_features.npy")
labels = np.load("E:\\Duo_Project\\anomaly_detection_project\\data\\processed\\balanced_labels.npy")

# Normalize features
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# Split into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

# Check class distribution
print("Training class distribution:", Counter(y_train))
print("Test class distribution:", Counter(y_test))

# Save processed data
np.save("E:\\Duo_Project\\anomaly_detection_project\\data\\processed\\X_train.npy", X_train)
np.save("E:\\Duo_Project\\anomaly_detection_project\\data\\processed\\X_test.npy", X_test)
np.save("E:\\Duo_Project\\anomaly_detection_project\\data\\processed\\y_train.npy", y_train)
np.save("E:\\Duo_Project\\anomaly_detection_project\\data\\processed\\y_test.npy", y_test)

print("✅ Data Preprocessing Complete!")
