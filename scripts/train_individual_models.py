import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pickle
import numpy as np
import joblib
from src.model_trainer import AnomalyDetectionModel
import config.config as cfg
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_and_save_models():
    """Train and save individual models for ensemble"""
    # Load processed data
    processed_data_path = cfg.FEATURES_DIR / 'processed_data.pkl'
    with open(processed_data_path, 'rb') as f:
        data = pickle.load(f)

    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

    models_to_train = {
        'svm': {'type': 'svm', 'params': {'C': 0.1}},
        'linear_svc': {'type': 'linear_svc', 'params': {'C': 0.1}},
        'random_forest': {'type': 'random_forest', 'params': {}}
    }

    results = {}

    for model_name, config in models_to_train.items():
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Training {model_name.upper()} model")
        logger.info(f"{'=' * 50}")

        try:
            model = AnomalyDetectionModel()
            model.build_model(model_type=config['type'])

            # Train the model
            train_acc, train_f1 = model.train(X_train, y_train)

            # Evaluate
            test_results = model.evaluate(X_test, y_test, save_plots=False)

            # Save the model
            model_path = cfg.MODEL_DIR / f'{model_name}_model.joblib'
            joblib.dump(model.model, str(model_path))

            results[model_name] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_results['accuracy'],
                'test_f1': test_results['f1_weighted'],
                'model_path': model_path
            }

            logger.info(f"{model_name.upper()} saved to {model_path}")

        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")
            continue

    return results


if __name__ == "__main__":
    results = train_and_save_models()

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("MODEL TRAINING SUMMARY")
    logger.info("=" * 60)
    for model_name, metrics in results.items():
        logger.info(f"{model_name:15s}: Train Acc: {metrics['train_accuracy']:.4f}, "
                    f"Test Acc: {metrics['test_accuracy']:.4f}, Test F1: {metrics['test_f1']:.4f}")