import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROCESSED_DATA_DIR / "models"
FEATURES_DIR = PROCESSED_DATA_DIR / "features"
TEMP_DIR = DATA_DIR / "temp"

# Create directories
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, FEATURES_DIR, TEMP_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Video categories
CATEGORIES = {
    "normal": 0,
    "road_accidents": 1,
    "robbery": 2,
    "abuse": 3
}

# Feature extraction parameters
FRAME_HEIGHT = 64
FRAME_WIDTH = 64
MAX_FRAMES = 30

# Training parameters
RANDOM_SEED = 42
TRAIN_TEST_SPLIT = 0.8

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 5000
MAX_UPLOAD_SIZE = 100 * 1024 * 1024