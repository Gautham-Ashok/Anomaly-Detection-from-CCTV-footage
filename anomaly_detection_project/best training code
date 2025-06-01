import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load data
X_train = np.load("E:\\Duo_Project\\anomaly_detection_project\\data\\processed\\balanced_features.npy")
y_train = np.load("E:\\Duo_Project\\anomaly_detection_project\\data\\processed\\balanced_labels.npy")

X_test = np.load("E:\\Duo_Project\\anomaly_detection_project\\data\\processed\\reduced_features.npy")
y_test = np.load("E:\\Duo_Project\\anomaly_detection_project\\data\\processed\\reduced_labels.npy")

# Ensure labels are binary (0 and 1)
y_train = np.where(y_train > 0.5, 1, 0).astype(np.float32)
y_test = np.where(y_test > 0.5, 1, 0).astype(np.float32)

# Normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape input to 3D for LSTM (samples, timesteps, features)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Handle class imbalance using SMOTE
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train.reshape(X_train.shape[0], -1), y_train)
X_train_resampled = X_train_resampled.reshape(X_train_resampled.shape[0], 1, X_train_resampled.shape[1])

# Compute class weights to handle imbalance
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_resampled), y=y_train_resampled)
class_weights_dict = dict(enumerate(class_weights))

# Ensure directory exists for saving model
model_path = "E:\\Duo_Project\\anomaly_detection_project\\model\\lstm_model.keras"
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# Define LSTM model
model = Sequential([
    LSTM(256, input_shape=(X_train_resampled.shape[1], X_train_resampled.shape[2]), return_sequences=True),
    BatchNormalization(),
    Dropout(0.4),

    LSTM(128),
    BatchNormalization(),
    Dropout(0.4),

    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")  # Sigmoid for binary classification
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model with class weights
history = model.fit(
    X_train_resampled, y_train_resampled,
    epochs=50,
    batch_size=64,
    validation_data=(X_test, y_test),
    class_weight=class_weights_dict
)

# Save model
model.save(model_path)

# Evaluate model
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Print classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot training vs validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()
