import numpy as np
import joblib
from pathlib import Path
import config.config as cfg
import logging
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleAnomalyDetector:
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.scaler = StandardScaler()

    def add_model(self, model_name, model_path, weight=1.0):
        """Add a pre-trained model to the ensemble"""
        try:
            model = joblib.load(model_path)
            self.models[model_name] = model
            self.weights[model_name] = weight
            logger.info(f"Added model: {model_name} with weight {weight}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")

    def predict_proba_ensemble(self, X):
        """Get weighted average probabilities from all models"""
        all_probabilities = []
        total_weight = 0

        for model_name, model in self.models.items():
            weight = self.weights[model_name]

            try:
                # Handle different model types
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                elif hasattr(model, 'decision_function'):
                    decision_scores = model.decision_function(X)
                    if decision_scores.ndim == 1:
                        proba = 1 / (1 + np.exp(-decision_scores))
                        proba = np.vstack([1 - proba, proba]).T
                    else:
                        proba = np.exp(decision_scores) / np.sum(np.exp(decision_scores), axis=1, keepdims=True)
                else:
                    # For models without probability support
                    predictions = model.predict(X)
                    proba = np.zeros((len(X), len(cfg.CATEGORIES)))
                    for i, pred in enumerate(predictions):
                        proba[i, pred] = 1.0

                # Apply weight
                weighted_proba = proba * weight
                all_probabilities.append(weighted_proba)
                total_weight += weight

            except Exception as e:
                logger.error(f"Error getting predictions from {model_name}: {e}")
                continue

        if not all_probabilities:
            raise ValueError("No models could make predictions")

        # Calculate weighted average
        ensemble_proba = sum(all_probabilities) / total_weight
        return ensemble_proba

    def predict(self, X):
        """Make ensemble predictions"""
        probabilities = self.predict_proba_ensemble(X)
        predictions = np.argmax(probabilities, axis=1)
        return predictions, probabilities

    def evaluate_ensemble(self, X_test, y_test):
        """Evaluate ensemble performance"""
        predictions, probabilities = self.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')

        logger.info(f"Ensemble Accuracy: {accuracy:.4f}")
        logger.info(f"Ensemble F1 Score: {f1:.4f}")

        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': predictions,
            'probabilities': probabilities
        }

    def save_ensemble(self, path=None):
        """Save the ensemble configuration"""
        if path is None:
            path = cfg.MODEL_DIR / 'ensemble_model.joblib'

        ensemble_data = {
            'models': self.models,
            'weights': self.weights
        }

        joblib.dump(ensemble_data, str(path))
        logger.info(f"Ensemble saved to {path}")

    def load_ensemble(self, path=None):
        """Load ensemble configuration"""
        if path is None:
            path = cfg.MODEL_DIR / 'ensemble_model.joblib'

        ensemble_data = joblib.load(str(path))
        self.models = ensemble_data['models']
        self.weights = ensemble_data['weights']
        logger.info(f"Ensemble loaded from {path}")


# Example usage:
def create_best_ensemble():
    """Create an ensemble with the best performing models"""
    ensemble = EnsembleAnomalyDetector()

    # Add your best models with optimized weights
    ensemble.add_model('svm', cfg.MODEL_DIR / 'svm_model.joblib', weight=0.7)
    ensemble.add_model('linear_svc', cfg.MODEL_DIR / 'linear_svc_model.joblib', weight=0.3)

    return ensemble