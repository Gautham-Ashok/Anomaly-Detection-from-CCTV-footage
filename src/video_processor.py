import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from src.feature_extractor import VideoFeatureExtractor
import config.config as cfg
import pickle
from tqdm import tqdm
import joblib

logger = logging.getLogger(__name__)


class VideoDataProcessor:
    def __init__(self, augment_data=False):
        self.feature_extractor = VideoFeatureExtractor(augment_data=augment_data)
        self.feature_names = None

    def process_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Process all videos in the dataset"""
        all_features = []
        all_labels = []
        all_video_paths = []

        for category, label in cfg.CATEGORIES.items():
            category_path = cfg.RAW_DATA_DIR / category
            if not category_path.exists():
                logger.warning(f"Category folder not found: {category_path}")
                continue

            video_files = list(category_path.glob("*.mp4")) + list(category_path.glob("*.avi")) + list(
                category_path.glob("*.mov"))
            logger.info(f"Found {len(video_files)} videos in {category}")

            for video_file in tqdm(video_files, desc=f"Processing {category}"):
                try:
                    features = self.feature_extractor.process_video(
                        str(video_file),
                        cfg.MAX_FRAMES
                    )

                    if len(features) > 0:
                        # Use mean of features across frames
                        mean_features = np.mean(features, axis=0)
                        all_features.append(mean_features)
                        all_labels.append(label)
                        all_video_paths.append(str(video_file))

                except Exception as e:
                    logger.error(f"Error processing {video_file}: {e}")

        # Save video paths for reference
        if all_video_paths:
            with open(cfg.FEATURES_DIR / 'video_paths.pkl', 'wb') as f:
                pickle.dump(all_video_paths, f)

        return np.array(all_features), np.array(all_labels)

    def balance_dataset(self, X, y, strategy='smote'):
        """Balance the dataset to handle class imbalance"""
        from collections import Counter
        logger.info(f"Original class distribution: {Counter(y)}")

        if strategy == 'smote':
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=cfg.RANDOM_SEED)
                X_balanced, y_balanced = smote.fit_resample(X, y)
                logger.info(f"After SMOTE balancing: {Counter(y_balanced)}")
                return X_balanced, y_balanced
            except ImportError:
                logger.warning("imblearn not installed. Using random oversampling instead.")
                strategy = 'oversample'

        if strategy == 'oversample':
            # Manual oversampling of minority classes
            unique_classes, counts = np.unique(y, return_counts=True)
            max_count = np.max(counts)

            X_balanced_list = []
            y_balanced_list = []

            for class_label in unique_classes:
                X_class = X[y == class_label]
                y_class = y[y == class_label]

                if len(X_class) < max_count:
                    # Oversample minority class
                    X_oversampled = resample(X_class,
                                             replace=True,
                                             n_samples=max_count,
                                             random_state=cfg.RANDOM_SEED)
                    y_oversampled = np.full(max_count, class_label)
                else:
                    X_oversampled = X_class
                    y_oversampled = y_class

                X_balanced_list.append(X_oversampled)
                y_balanced_list.append(y_oversampled)

            X_balanced = np.vstack(X_balanced_list)
            y_balanced = np.concatenate(y_balanced_list)
            logger.info(f"After oversampling: {Counter(y_balanced)}")
            return X_balanced, y_balanced

        return X, y

    def extract_feature_names(self):
        """Generate descriptive feature names"""
        if self.feature_names is None:
            feature_names = []
            # HOG features
            for i in range(144):
                feature_names.append(f'hog_{i}')
            # Optical flow features
            for i in range(8):
                feature_names.append(f'optical_flow_{i}')
            # Texture features
            for i in range(10):
                feature_names.append(f'texture_{i}')
            # Color features
            for i in range(24):
                feature_names.append(f'color_{i}')
            self.feature_names = feature_names
        return self.feature_names

    def prepare_data(self, balance_data=True) -> Dict:
        """Prepare and save processed data"""
        logger.info("Starting data preparation...")

        # Check if processed data already exists
        processed_data_path = cfg.FEATURES_DIR / 'processed_data.pkl'
        if processed_data_path.exists():
            logger.info("Loading existing processed data...")
            with open(processed_data_path, 'rb') as f:
                data = pickle.load(f)
            return data

        X, y = self.process_dataset()

        if len(X) == 0:
            raise ValueError("No data processed. Check your dataset.")

        # Balance the dataset if requested
        if balance_data:
            X, y = self.balance_dataset(X, y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1 - cfg.TRAIN_TEST_SPLIT,
            random_state=cfg.RANDOM_SEED, stratify=y
        )

        # Save feature names
        feature_names = self.extract_feature_names()
        with open(cfg.FEATURES_DIR / 'feature_names.pkl', 'wb') as f:
            pickle.dump(feature_names, f)

        # Save processed data
        data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names
        }

        with open(processed_data_path, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"Data saved to {processed_data_path}")
        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        logger.info(f"Feature dimension: {X_train.shape[1]}")

        return data

    def load_processed_data(self):
        """Load previously processed data"""
        processed_data_path = cfg.FEATURES_DIR / 'processed_data.pkl'
        if not processed_data_path.exists():
            raise FileNotFoundError("Processed data not found. Run prepare_data() first.")

        with open(processed_data_path, 'rb') as f:
            data = pickle.load(f)

        return data

    def get_class_distribution(self, y):
        """Get class distribution statistics"""
        from collections import Counter
        counts = Counter(y)
        total = len(y)

        distribution = {}
        for class_idx, class_name in cfg.CATEGORIES.items():
            count = counts.get(class_idx, 0)
            percentage = (count / total) * 100
            distribution[class_name] = {
                'count': count,
                'percentage': f"{percentage:.1f}%"
            }

        return distribution