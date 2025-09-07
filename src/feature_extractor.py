import cv2
import numpy as np
from pathlib import Path
import logging
from skimage.feature import hog, local_binary_pattern
import joblib
from sklearn.decomposition import PCA
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoFeatureExtractor:
    def __init__(self, augment_data=False):
        self.feature_dim = 256
        self.pca = PCA(n_components=self.feature_dim)
        self.pca_fitted = False
        self.augment_data = augment_data

    def _augment_frame(self, frame):
        """Apply data augmentation to frame"""
        if not self.augment_data:
            return frame

        # Random horizontal flip
        if random.random() > 0.5:
            frame = cv2.flip(frame, 1)

        # Random brightness adjustment
        brightness = random.uniform(0.8, 1.2)
        frame = np.clip(frame * brightness, 0, 255).astype(np.uint8)

        # Random contrast adjustment
        contrast = random.uniform(0.8, 1.2)
        frame = np.clip(128 + contrast * (frame - 128), 0, 255).astype(np.uint8)

        return frame

    def extract_frames(self, video_path: str, max_frames: int = 30) -> np.ndarray:
        """Extract frames from video with optional augmentation"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            frames = []

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                logger.error(f"No frames in video: {video_path}")
                return np.array([])

            interval = max(1, total_frames // max_frames)

            frame_count = 0
            while len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % interval == 0:
                    frame = cv2.resize(frame, (64, 64))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Apply augmentation
                    frame = self._augment_frame(frame)

                    frames.append(frame)

                frame_count += 1

            cap.release()
            return np.array(frames)

        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            return np.array([])

    def extract_features(self, frames: np.ndarray) -> np.ndarray:
        """Extract handcrafted features from frames"""
        if len(frames) == 0:
            return np.array([])

        features = []
        prev_gray = None

        for frame in frames:
            try:
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

                # Extract HOG features
                hog_features = self._extract_hog_features(gray)

                # Extract optical flow features (if previous frame exists)
                if prev_gray is not None:
                    flow_features = self._extract_optical_flow_features(gray, prev_gray)
                else:
                    flow_features = np.zeros(8)

                # Extract texture features
                texture_features = self._extract_texture_features(gray)

                # Extract color features
                color_features = self._extract_color_features(frame)

                # Combine all features
                combined_features = np.concatenate([
                    hog_features,
                    flow_features,
                    texture_features,
                    color_features
                ])

                features.append(combined_features)
                prev_gray = gray

            except Exception as e:
                logger.warning(f"Error extracting features from frame: {e}")
                continue

        if len(features) == 0:
            return np.array([])

        return np.array(features)

    def _extract_hog_features(self, gray_image: np.ndarray) -> np.ndarray:
        """Extract HOG features"""
        try:
            features = hog(gray_image, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), feature_vector=True)
            return features
        except Exception as e:
            logger.warning(f"HOG extraction failed: {e}")
            return np.zeros(144)  # Default size for 64x64 image

    def _extract_optical_flow_features(self, current_gray: np.ndarray,
                                       previous_gray: np.ndarray) -> np.ndarray:
        """Extract optical flow features"""
        try:
            flow = cv2.calcOpticalFlowFarneback(previous_gray, current_gray,
                                                None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Create histogram of optical flow magnitudes
            hist, _ = np.histogram(magnitude, bins=8, range=(0, np.max(magnitude) + 1e-6))
            return hist.astype(np.float32)
        except Exception as e:
            logger.warning(f"Optical flow extraction failed: {e}")
            return np.zeros(8)

    def _extract_texture_features(self, gray_image: np.ndarray) -> np.ndarray:
        """Extract texture features using LBP"""
        try:
            lbp = local_binary_pattern(gray_image, 8, 1, method='uniform')
            hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
            return hist.astype(np.float32)
        except Exception as e:
            logger.warning(f"LBP extraction failed: {e}")
            return np.zeros(10)

    def _extract_color_features(self, frame: np.ndarray) -> np.ndarray:
        """Extract color histogram features"""
        try:
            # Extract histograms for each channel
            hist_r = cv2.calcHist([frame], [0], None, [8], [0, 256]).flatten()
            hist_g = cv2.calcHist([frame], [1], None, [8], [0, 256]).flatten()
            hist_b = cv2.calcHist([frame], [2], None, [8], [0, 256]).flatten()

            # Normalize histograms
            hist_r = hist_r / (np.sum(hist_r) + 1e-6)
            hist_g = hist_g / (np.sum(hist_g) + 1e-6)
            hist_b = hist_b / (np.sum(hist_b) + 1e-6)

            return np.concatenate([hist_r, hist_g, hist_b])
        except Exception as e:
            logger.warning(f"Color feature extraction failed: {e}")
            return np.zeros(24)

    def process_video(self, video_path: str, max_frames: int = 30) -> np.ndarray:
        """Process a single video and return features"""
        frames = self.extract_frames(video_path, max_frames)
        if len(frames) == 0:
            return np.array([])

        features = self.extract_features(frames)
        return features

    def save_pca(self, path: str):
        """Save PCA model"""
        joblib.dump(self.pca, path)

    def load_pca(self, path: str):
        """Load PCA model"""
        self.pca = joblib.load(path)
        self.pca_fitted = True