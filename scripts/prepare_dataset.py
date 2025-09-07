import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.video_processor import VideoDataProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting dataset preparation...")

    processor = VideoDataProcessor()

    try:
        data = processor.prepare_data()
        logger.info("Dataset preparation completed successfully!")
        logger.info(f"Training samples: {len(data['X_train'])}")
        logger.info(f"Test samples: {len(data['X_test'])}")

    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        raise


if __name__ == "__main__":
    main()