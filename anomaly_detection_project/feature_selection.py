import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Define base directory
base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
features_dir = os.path.join(base_dir,"utils", "data", "processed", "features")  # Path to features

# Define class labels based on folder names
class_labels = {
    "Explosion": 0,
    "Fighting": 1,
    "Normal_Videos_for_Event_Recognition": 2,
    "RoadAccidents": 3
}

# Initialize empty lists for features and labels
X_list = []
y_list = []

# Loop through each category folder and load .npy files
for category, label in class_labels.items():
    category_path = os.path.join(features_dir, category)

    if not os.path.exists(category_path):
        print(f"Warning: {category_path} does not exist. Skipping...")
        continue

    for file in os.listdir(category_path):
        if file.endswith(".npy"):  # Only process .npy files
            file_path = os.path.join(category_path, file)
            features = np.load(file_path)  # Load the numpy file

            # Ensure features are 2D (samples, features)
            if features.ndim == 1:
                features = features.reshape(1, -1)

            X_list.append(features)
            y_list.extend([label] * features.shape[0])  # Assign labels

# Convert lists to NumPy arrays
X = np.vstack(X_list)  # Stack all features
y = np.array(y_list)  # Convert labels to NumPy array

# Check if we have valid data
if X.shape[0] == 0:
    raise ValueError("No valid feature files found in the directory!")

print(f"Loaded dataset with {X.shape[0]} samples and {X.shape[1]} features.")

# Step 1: Remove Highly Correlated Features (Above 0.9)
correlation_matrix = np.corrcoef(X, rowvar=False)  # Compute correlation between features
upper_triangle = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
high_correlation_features = [i for i in range(correlation_matrix.shape[0]) if
                             any(correlation_matrix[i, upper_triangle[i]] > 0.9)]

X = np.delete(X, high_correlation_features, axis=1)

# Step 2: Use SelectKBest to Select Top Features
k_best = 10  # Select top 10 features
selector = SelectKBest(score_func=mutual_info_classif, k=min(k_best, X.shape[1]))
X_selected = selector.fit_transform(X, y)

# Get selected feature indices
selected_feature_indices = selector.get_support(indices=True)

# Save the reduced dataset
output_dir = os.path.join(base_dir, "data", "processed")
os.makedirs(output_dir, exist_ok=True)

X_output_path = os.path.join(output_dir, "reduced_features.npy")
y_output_path = os.path.join(output_dir, "reduced_labels.npy")

np.save(X_output_path, X_selected)
np.save(y_output_path, y)

print(f"Feature selection complete! Reduced features saved at: {X_output_path}")
print(f"Labels saved at: {y_output_path}")
