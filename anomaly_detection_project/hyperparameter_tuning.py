import optuna
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler

# Load data
X_train = np.load("E:\\Duo_Project\\anomaly_detection_project\\data\\processed\\balanced_features.npy")
y_train = np.load("E:\\Duo_Project\\anomaly_detection_project\\data\\processed\\balanced_labels.npy")

X_test = np.load("E:\\Duo_Project\\anomaly_detection_project\\data\\processed\\reduced_features.npy")
y_test = np.load("E:\\Duo_Project\\anomaly_detection_project\\data\\processed\\reduced_labels.npy")

# Ensure labels are binary (0 or 1)
y_train = np.where(y_train > 0.5, 1, 0).astype(np.float32)
y_test = np.where(y_test > 0.5, 1, 0).astype(np.float32)

# Normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape input to 3D for LSTM
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Handle class imbalance using SMOTE
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train.reshape(X_train.shape[0], -1), y_train)
X_train_resampled = X_train_resampled.reshape(X_train_resampled.shape[0], 1, X_train_resampled.shape[1])

# Compute class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_resampled), y=y_train_resampled)
class_weights_dict = dict(enumerate(class_weights))


# Objective function for Optuna
def objective(trial):
    # Hyperparameters to tune
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    lstm_units_1 = trial.suggest_int("lstm_units_1", 64, 256, step=64)
    lstm_units_2 = trial.suggest_int("lstm_units_2", 32, 128, step=32)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    # Define LSTM model
    model = Sequential([
        LSTM(lstm_units_1, input_shape=(X_train_resampled.shape[1], X_train_resampled.shape[2]), return_sequences=True),
        BatchNormalization(),
        Dropout(dropout_rate),

        LSTM(lstm_units_2),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")  # Binary classification
    ])

    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    # Train model
    model.fit(X_train_resampled, y_train_resampled,
              epochs=20,  # You can increase for better tuning
              batch_size=batch_size,
              validation_data=(X_test, y_test),
              class_weight=class_weights_dict,
              verbose=0)  # Silent training

    # Evaluate model
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)

    return 1 - accuracy  # Minimize error (1 - accuracy)


# Run hyperparameter tuning
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# Print best parameters
print("Best hyperparameters:", study.best_params)
