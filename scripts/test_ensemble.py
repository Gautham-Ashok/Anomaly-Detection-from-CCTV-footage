import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pickle
import numpy as np
from src.ensemble_trainer import EnsembleAnomalyDetector, create_best_ensemble
import config.config as cfg
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_ensemble():
    """Test different ensemble combinations"""
    # Load processed data
    processed_data_path = cfg.FEATURES_DIR / 'processed_data.pkl'
    with open(processed_data_path, 'rb') as f:
        data = pickle.load(f)

    X_test, y_test = data['X_test'], data['y_test']

    # Test different ensemble combinations
    ensemble_configs = [
        {'name': 'SVM Only', 'models': [('svm', 1.0)]},
        {'name': 'SVM + LinearSVC (70/30)', 'models': [('svm', 0.7), ('linear_svc', 0.3)]},
        {'name': 'SVM + LinearSVC (60/40)', 'models': [('svm', 0.6), ('linear_svc', 0.4)]},
        {'name': 'All Models Equal', 'models': [('svm', 0.33), ('linear_svc', 0.33), ('random_forest', 0.34)]},
    ]

    best_score = 0
    best_ensemble = None
    best_config_name = ""

    for config in ensemble_configs:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Testing ensemble: {config['name']}")
        logger.info(f"{'=' * 50}")

        try:
            ensemble = EnsembleAnomalyDetector()

            # Add models with specified weights
            for model_name, weight in config['models']:
                model_path = cfg.MODEL_DIR / f'{model_name}_model.joblib'
                if model_path.exists():
                    ensemble.add_model(model_name, model_path, weight)
                else:
                    logger.warning(f"Model not found: {model_path}")

            # Evaluate ensemble
            results = ensemble.evaluate_ensemble(X_test, y_test)

            # Check if this is the best ensemble
            if results['f1_score'] > best_score:
                best_score = results['f1_score']
                best_ensemble = ensemble
                best_config_name = config['name']

        except Exception as e:
            logger.error(f"Error testing ensemble {config['name']}: {e}")
            continue

    # Save the best ensemble
    if best_ensemble is not None:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"BEST ENSEMBLE: {best_config_name}")
        logger.info(f"Best F1 Score: {best_score:.4f}")
        logger.info(f"{'=' * 50}")

        best_ensemble.save_ensemble()

        # Also save as the main model for API use
        main_model_path = cfg.MODEL_DIR / 'anomaly_detection_model.joblib'
        best_ensemble.save_ensemble(main_model_path)
        logger.info(f"Best ensemble saved as main model: {main_model_path}")


if __name__ == "__main__":
    test_ensemble()