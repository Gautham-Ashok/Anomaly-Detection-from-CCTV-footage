import numpy as np
import logging
import joblib
import time
from pathlib import Path
from typing import Dict
from src.feature_extractor import VideoFeatureExtractor
import config.config as cfg

logger = logging.getLogger(__name__)

class AnomalyDetector:
    def __init__(self, model_path: str = None):
        self.feature_extractor = VideoFeatureExtractor()
        if model_path is None:
            model_path = cfg.MODEL_DIR / 'anomaly_detection_model.joblib'
        try:
            self.model = joblib.load(str(model_path))
            self.model_loaded = True
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Could not load model: {e}")
            self.model = None
            self.model_loaded = False

        self.label_map = {v: k for k, v in cfg.CATEGORIES.items()}

    def detect_anomaly(self, video_path: str) -> Dict:
        if not self.model_loaded:
            return {'error': 'Model not loaded', 'status': 'error'}

        try:
            start_time = time.time()
            features = self.feature_extractor.process_video(video_path, cfg.MAX_FRAMES)
            if len(features) == 0:
                return {'error': 'Could not extract features from video', 'status': 'error'}
            mean_features = np.mean(features, axis=0)
            prediction, probabilities = self._predict(mean_features)
            processing_time = time.time() - start_time

            result = {
                'status': 'success',
                'anomaly_type': self.label_map.get(prediction, 'unknown'),
                'anomaly_id': int(prediction),
                'confidence': float(probabilities[prediction]),
                'processing_time': round(processing_time, 2),
                'all_probabilities': {
                    self.label_map.get(i, 'unknown'): {
                        'probability': float(probabilities[i]),
                        'id': i
                    } for i in range(len(probabilities))
                },
                'frame_count': len(features)
            }
            return result
        except Exception as e:
            logger.error(f"Error detecting anomaly: {e}")
            return {'error': str(e), 'status': 'error'}

    def _predict(self, features: np.ndarray):
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if hasattr(self, 'models'):  # Ensemble
            return self._predict_ensemble(features)
        else:
            if hasattr(self.model, 'predict_proba'):
                prediction = self.model.predict(features)
                probability = self.model.predict_proba(features)
            else:
                prediction = self.model.predict(features)
                probability = np.zeros((1, len(self.label_map)))
                probability[0, prediction[0]] = 1.0
            return prediction[0], probability[0]

    def _predict_ensemble(self, features: np.ndarray):
        all_probabilities = []
        total_weight = 0
        for model_name, (model, weight) in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features)
                else:
                    pred = model.predict(features)
                    proba = np.zeros((1, len(self.label_map)))
                    proba[0, pred[0]] = 1.0
                weighted_proba = proba * weight
                all_probabilities.append(weighted_proba)
                total_weight += weight
            except Exception as e:
                logger.warning(f"Error from model {model_name}: {e}")
                continue
        if not all_probabilities:
            raise ValueError("No models could make predictions")
        ensemble_proba = sum(all_probabilities) / total_weight
        prediction = np.argmax(ensemble_proba, axis=1)
        return prediction[0], ensemble_proba[0]

    def get_model_info(self):
        if not self.model_loaded:
            return {'loaded': False}
        info = {'loaded': True, 'type': type(self.model).__name__, 'input_shape': 'N/A'}
        try:
            if hasattr(self.model, 'n_features_in_'):
                info['input_features'] = self.model.n_features_in_
        except Exception:
            pass
        return info
