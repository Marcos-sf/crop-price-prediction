import pandas as pd
import os

# Quick fix for the API column detection issue
def test_and_fix_column_detection():
    """Test column detection and create a simple fix"""
    
    file_path = "app/data/processed/arecanut_sirsi_for_training.csv"
    if not os.path.exists(file_path):
        print("File not found!")
        return
    
    df = pd.read_csv(file_path)
    print("Available columns:")
    for i, col in enumerate(df.columns):
        print(f"  {i}: '{col}' (stripped: '{col.strip()}')")
    
    # Test direct column access
    if 'Arrival_Date' in df.columns:
        print("✅ 'Arrival_Date' found directly")
    
    if 'Modal_Price' in df.columns:
        print("✅ 'Modal_Price' found directly")
    
    # Create the fix
    fix_code = '''
def find_column_simple(df, target_columns):
    """Simple column finder that works with exact matches"""
    for target in target_columns:
        if target in df.columns:
            return target
    return None
'''
    
    print("\nRecommended fix:")
    print(fix_code)
    
    # Test the fix
    def find_column_simple(df, target_columns):
        for target in target_columns:
            if target in df.columns:
                return target
        return None
    
    date_col = find_column_simple(df, ['Arrival_Date', 'arrival_date', 'Date', 'date'])
    price_col = find_column_simple(df, ['Modal_Price', 'modal_price', 'Modal Price', 'modal price'])
    
    print(f"Date column found: {date_col}")
    print(f"Price column found: {price_col}")

if __name__ == "__main__":
    test_and_fix_column_detection()
