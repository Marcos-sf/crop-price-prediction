#!/usr/bin/env python3
"""
Debug script to check column names in the data files.
"""

import pandas as pd
from pathlib import Path

def check_columns():
    """Check the column names in the data files."""
    data_dir = Path(__file__).parent / "app" / "data"
    raw_dir = data_dir / "raw"
    
    print("Checking column names in data files...")
    
    for file_path in raw_dir.glob("*.xls*"):
        print(f"\nFile: {file_path.name}")
        try:
            # Try reading as CSV first
            df = pd.read_csv(file_path)
            print(f"  Successfully read as CSV")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  First few rows:")
            print(df.head(3).to_string())
            
        except Exception as e:
            print(f"  Error reading as CSV: {e}")
            try:
                df = pd.read_excel(file_path, engine='openpyxl')
                print(f"  Successfully read as Excel")
                print(f"  Shape: {df.shape}")
                print(f"  Columns: {list(df.columns)}")
                print(f"  First few rows:")
                print(df.head(3).to_string())
            except Exception as e2:
                print(f"  Error reading as Excel: {e2}")

if __name__ == "__main__":
    check_columns() 