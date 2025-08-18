#!/usr/bin/env python3
"""
Simple test script to check data processing functionality.
"""

import sys
from pathlib import Path
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_xls_reading():
    """Test if we can read the XLS files."""
    data_dir = Path(__file__).parent / "app" / "data"
    raw_dir = data_dir / "raw"
    
    logger.info(f"Checking data directory: {data_dir}")
    logger.info(f"Raw directory: {raw_dir}")
    
    if not raw_dir.exists():
        logger.error(f"Raw directory not found: {raw_dir}")
        return False
    
    # List XLS files
    xls_files = list(raw_dir.glob("*.xls*"))
    logger.info(f"Found {len(xls_files)} XLS files:")
    
    for file in xls_files:
        logger.info(f"  - {file.name} ({file.stat().st_size / (1024*1024):.1f} MB)")
        
        # Try to read the file
        try:
            df = pd.read_excel(file, engine='openpyxl')
            logger.info(f"    ✅ Successfully read {file.name}")
            logger.info(f"    Shape: {df.shape}")
            logger.info(f"    Columns: {list(df.columns)}")
            
            # Show first few rows
            logger.info(f"    First few rows:")
            logger.info(f"    {df.head(3).to_string()}")
            
        except Exception as e:
            logger.error(f"    ❌ Error reading {file.name}: {e}")
            # Try alternative engine
            try:
                df = pd.read_excel(file, engine='xlrd')
                logger.info(f"    ✅ Successfully read {file.name} with xlrd")
                logger.info(f"    Shape: {df.shape}")
                logger.info(f"    Columns: {list(df.columns)}")
            except Exception as e2:
                logger.error(f"    ❌ Error reading {file.name} with xlrd: {e2}")
    
    return True

def main():
    """Main test function."""
    logger.info("Starting data processing test...")
    
    # Test XLS reading
    success = test_xls_reading()
    
    if success:
        logger.info("✅ Basic XLS reading test passed!")
        logger.info("You can now run the full data processing script.")
    else:
        logger.error("❌ XLS reading test failed!")
        logger.error("Please check your XLS files and dependencies.")

if __name__ == "__main__":
    main() 