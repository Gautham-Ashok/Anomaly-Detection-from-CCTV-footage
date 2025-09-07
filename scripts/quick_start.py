#!/usr/bin/env python
"""
Quick start script to set up and test the anomaly detection system
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import logging
import config.config as cfg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories"""
    directories = [
        cfg.RAW_DATA_DIR / "normal",
        cfg.RAW_DATA_DIR / "road_accidents",
        cfg.RAW_DATA_DIR / "robbery",
        cfg.RAW_DATA_DIR / "abuse",
        cfg.PROCESSED_DATA_DIR,
        cfg.MODEL_DIR,
        cfg.FEATURES_DIR,
        cfg.TEMP_DIR
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def check_data():
    """Check if data is present"""
    data_present = False
    for category in cfg.CATEGORIES.keys():
        category_path = cfg.RAW_DATA_DIR / category
        video_count = len(list(category_path.glob("*.mp4")))
        if video_count > 0:
            logger.info(f"Found {video_count} videos in {category}")
            data_present = True
        else:
            logger.warning(f"No videos found in {category}")

    return data_present


def main():
    logger.info("=== Anomaly Detection System Setup ===")

    # Step 1: Setup directories
    logger.info("\n1. Setting up directories...")
    setup_directories()

    # Step 2: Check data
    logger.info("\n2. Checking for video data...")
    if not check_data():
        logger.error("\n❌ No video data found!")
        logger.info("\nPlease add your video files to:")
        logger.info(f"  - Normal videos: {cfg.RAW_DATA_DIR}/normal/")
        logger.info(f"  - Road accident videos: {cfg.RAW_DATA_DIR}/road_accidents/")
        logger.info(f"  - Robbery videos: {cfg.RAW_DATA_DIR}/robbery/")
        logger.info(f"  - Abuse videos: {cfg.RAW_DATA_DIR}/abuse/")
        return

    logger.info("\n✅ Setup complete!")
    logger.info("\nNext steps:")
    logger.info("1. Run: python scripts/train_model.py")
    logger.info("2. Run: python api/app.py")
    logger.info("3. Access API at: http://localhost:5000")


if __name__ == "__main__":
    main()