import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pickle
import numpy as np
from src.model_trainer import AnomalyDetectionModel
import config.config as cfg
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting quick training process...")

    # Load processed data
    processed_data_path = cfg.FEATURES_DIR / 'processed_data.pkl'
    with open(processed_data_path, 'rb') as f:
        data = pickle.load(f)

    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

    logger.info(f"Data shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
    logger.info(f"Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")

    # Try different models
    models_to_try = ['svm', 'random_forest', 'linear_svc']

    best_score = 0
    best_model = None
    best_model_name = ""
    best_results = None

    for model_type in models_to_try:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Testing {model_type.upper()} model")
        logger.info(f"{'=' * 50}")

        try:
            model = AnomalyDetectionModel()

            # Train the model
            train_acc, train_f1 = model.train(X_train, y_train, model_type=model_type)

            # Evaluate
            results = model.evaluate(X_test, y_test, save_plots=False)

            logger.info(f"{model_type.upper()} - Train Acc: {train_acc:.4f}, Test Acc: {results['accuracy']:.4f}")
            logger.info(f"{model_type.upper()} - Train F1: {train_f1:.4f}, Test F1: {results['f1_weighted']:.4f}")

            # Check generalization gap
            generalization_gap = train_acc - results['accuracy']
            logger.info(f"Generalization gap: {generalization_gap:.4f}")

            # Prefer models with smaller generalization gap and good test performance
            if (results['f1_weighted'] > best_score and generalization_gap < 0.2) or (
                    results['f1_weighted'] > best_score + 0.1):
                best_score = results['f1_weighted']
                best_model = model
                best_model_name = model_type
                best_results = results

        except Exception as e:
            logger.error(f"Error training {model_type}: {e}")
            continue

    logger.info(f"\n{'=' * 50}")
    if best_model is not None:
        logger.info(f"BEST MODEL: {best_model_name.upper()} with F1 score: {best_score:.4f}")
        logger.info(f"{'=' * 50}")

        # Save the best model
        best_model.save_model()
        logger.info(f"Best model saved as {best_model_name}")

        # Final evaluation with plots
        results = best_model.evaluate(X_test, y_test, save_plots=True)
        best_model.analyze_performance(X_test, y_test)

        # Show detailed results
        logger.info("\nDetailed Results:")
        logger.info(f"Overall Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Weighted F1 Score: {results['f1_weighted']:.4f}")

        logger.info("\nPer-class Performance:")
        for class_name, metrics in results['class_metrics'].items():
            logger.info(f"{class_name:15s}: Precision={metrics['precision']:.3f}, "
                        f"Recall={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")
    else:
        logger.warning("No suitable model found. All models are overfitting.")


if __name__ == "__main__":
    main()