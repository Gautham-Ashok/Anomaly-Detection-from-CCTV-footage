import numpy as np

features = np.load("E:\\Duo_Project\\anomaly_detection_project\\data\\processed\\reduced_features.npy")

print("Feature Mean:", np.mean(features))
print("Feature Std Dev:", np.std(features))
print("Feature Min:", np.min(features))
print("Feature Max:", np.max(features))
