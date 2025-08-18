import pandas as pd
import numpy as np

# === CONFIGURATION ===
# Set your filtered file path here (update as needed)
input_path = "app/data/processed/arecanut_arakalgud_for_training.csv"  # Change as needed

# === LOAD DATA ===
df = pd.read_csv(input_path)

# Standardize column names for easier processing
original_cols = list(df.columns)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Rename columns for consistency
col_map = {
    'arrival_date': 'date',
    'price_date': 'date',
    'modal_price': 'modal_price',
    'modal_price_(rs./quintal)': 'modal_price'
}
df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

# Check for date and modal_price columns
if 'date' not in df.columns:
    print("Columns after renaming:", list(df.columns))
    raise Exception("No 'date' column found after renaming! Please check your CSV and column mapping.")

if 'modal_price' not in df.columns:
    print("Columns after renaming:", list(df.columns))
    raise Exception("No 'modal_price' column found after renaming! Please check your CSV and column mapping.")

df = df[['date', 'modal_price']].dropna()

# Convert date to datetime and sort
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])
df = df.sort_values('date')

# Create lag features (for time series models)
df['modal_price_lag1'] = df['modal_price'].shift(1)
df['modal_price_lag7'] = df['modal_price'].shift(7)
df['day_of_year'] = df['date'].dt.dayofyear
df['month'] = df['date'].dt.month

df = df.dropna()

# Split into features and target
features = df[['modal_price_lag1', 'modal_price_lag7', 'day_of_year', 'month']].values
target = df['modal_price'].values

print(f"Prepared data: {features.shape[0]} samples, {features.shape[1]} features")
print("Sample features:\n", features[:5])
print("Sample target:\n", target[:5])

# Only save if there is data
if features.size > 0 and target.size > 0:
    np.save("app/data/processed/features.npy", features)
    np.save("app/data/processed/target.npy", target)
    print("Saved features.npy and target.npy in app/data/processed/")
else:
    print("No data to save! Check your filtered CSV and try again.") 