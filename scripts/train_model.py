import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pickle
import numpy as np
from src.video_processor import VideoDataProcessor
from src.model_trainer import AnomalyDetectionModel
import config.config as cfg
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting training process...")

    # Process data
    processor = VideoDataProcessor()
    processed_data_path = cfg.FEATURES_DIR / 'processed_data.pkl'

    if processed_data_path.exists():
        logger.info("Loading existing processed data...")
        with open(processed_data_path, 'rb') as f:
            data = pickle.load(f)
    else:
        logger.info("Processing videos...")
        data = processor.prepare_data()

    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

    logger.info(f"Data shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
    logger.info(f"Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")

    # Build and train model
    model = AnomalyDetectionModel()
    model.build_model(model_type='random_forest')

    logger.info("Training model...")
    train_acc, train_f1 = model.train(X_train, y_train)


    # Evaluate
    logger.info("Evaluating model...")
    results = model.evaluate(X_test, y_test)

    logger.info(f"Test Accuracy: {results['accuracy']:.4f}")
    logger.info(f"Test F1 Score: {results['f1_weighted']:.4f}")

    logger.info("\nDetailed Classification Report:")
    for category, metrics in results['class_metrics'].items():
        logger.info(f"{category:15s}: Precision={metrics['precision']:.3f}, "
                    f"Recall={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")

    # Comprehensive analysis
    model.analyze_performance(X_test, y_test)

    # Check for overfitting
    if train_acc - results['accuracy'] > 0.15:
        logger.warning("⚠️  Potential overfitting detected! Consider:")
        logger.warning("   - Adding more training data")
        logger.warning("   - Using data augmentation")
        logger.warning("   - Trying a simpler model")

    # Save model
    model.save_model()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()