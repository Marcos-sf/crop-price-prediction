#!/usr/bin/env python3
"""
Data processing script for crop price prediction project.
This script processes XLS files and prepares them for ML model training.
"""

import sys
import os
from pathlib import Path

# Add the app directory to Python path
current_dir = Path(__file__).parent
app_dir = current_dir / "app"
sys.path.insert(0, str(app_dir))

try:
    from utils.data_processor import DataProcessor
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    sys.exit(1)

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to process crop data."""
    logger.info("Starting data processing...")
    
    # Initialize data processor
    processor = DataProcessor()
    
    # Check if raw data directory exists and has files
    raw_dir = processor.raw_dir
    if not raw_dir.exists():
        logger.error(f"Raw data directory not found: {raw_dir}")
        logger.info("Please create the directory and place your XLS files there:")
        logger.info(f"  {raw_dir}")
        return
    
    # List available data files (XLS, XLSX, CSV)
    data_files = list(raw_dir.glob("*.xls*")) + list(raw_dir.glob("*.csv"))
    if not data_files:
        logger.error("No data files found in raw data directory")
        logger.info(f"Please place your data files (XLS, XLSX, CSV) in: {raw_dir}")
        return
    
    logger.info(f"Found {len(data_files)} data files:")
    for file in data_files:
        logger.info(f"  - {file.name}")
    
    # Process each crop
    crops_to_process = ['arecanut', 'coconut']
    
    for crop in crops_to_process:
        logger.info(f"\nProcessing {crop} data...")
        try:
            df = processor.process_crop_data(crop)
            if not df.empty:
                logger.info(f"✅ Successfully processed {crop} data")
                logger.info(f"   Shape: {df.shape}")
                logger.info(f"   Date range: {df['date'].min()} to {df['date'].max()}")
                
                # Show available mandis
                mandis = processor.get_mandis_for_crop(crop)
                if mandis:
                    logger.info(f"   Available mandis: {', '.join(mandis)}")
                else:
                    logger.warning(f"   No mandi information found for {crop}")
                
                # Test training data preparation
                features, target = processor.prepare_training_data(crop)
                if len(features) > 0:
                    logger.info(f"   Training data: {features.shape[0]} samples, {features.shape[1]} features")
                else:
                    logger.warning(f"   Could not prepare training data for {crop}")
            else:
                logger.warning(f"❌ No data found for {crop}")
        except Exception as e:
            logger.error(f"❌ Error processing {crop}: {e}")
    
    # Show summary
    logger.info("\n" + "="*50)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*50)
    
    available_crops = processor.get_available_crops()
    if available_crops:
        logger.info(f"✅ Successfully processed crops: {', '.join(available_crops)}")
        for crop in available_crops:
            mandis = processor.get_mandis_for_crop(crop)
            logger.info(f"   {crop}: {len(mandis)} mandis available")
    else:
        logger.warning("❌ No crops were successfully processed")
    
    logger.info("\nNext steps:")
    logger.info("1. Check the processed data in: backend/app/data/processed/")
    logger.info("2. Run the backend server: python -m uvicorn app.main:app --reload")
    logger.info("3. Test the API endpoints with your real data")

if __name__ == "__main__":
    main() 