import numpy as np
import os

label_files = [
    "E:\\Duo_Project\\anomaly_detection_project\\data\\processed\\balanced_labels.npy",
    "E:\\Duo_Project\\anomaly_detection_project\\data\\processed\\reduced_labels.npy",
    "E:\\Duo_Project\\anomaly_detection_project\\data\\processed\\y_train.npy"
]

for label_file in label_files:
    if os.path.exists(label_file):
        labels = np.load(label_file)
        print(f"{label_file} -> Labels Shape: {labels.shape}")
    else:
        print(f"{label_file} -> ❌ File not found!")
