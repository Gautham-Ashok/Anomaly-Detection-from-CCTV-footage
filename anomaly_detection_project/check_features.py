import numpy as np

FEATURES_PATH = "E:\\Duo_Project\\anomaly_detection_project\\data\\processed\\features.npy"
LABELS_PATHS = [
    "E:\\Duo_Project\\anomaly_detection_project\\data\\processed\\balanced_labels.npy",
    "E:\\Duo_Project\\anomaly_detection_project\\data\\processed\\reduced_labels.npy",
    "E:\\Duo_Project\\anomaly_detection_project\\data\\processed\\y_train.npy"
]

features = np.load(FEATURES_PATH)
print(f"Features Shape: {features.shape}")

for path in LABELS_PATHS:
    labels = np.load(path)
    print(f"{path} -> Labels Shape: {labels.shape}")
