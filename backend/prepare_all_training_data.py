import pandas as pd
import numpy as np
import os

# Top 6 mandis for each crop
mandis = {
    'arecanut': ['Sirsi', 'Yellapur', 'Siddapur', 'Shimoga', 'Sagar', 'Kumta'],
    'coconut': ['Bangalore', 'Arasikere', 'Channarayapatna', 'Ramanagara', 'Sira', 'Tumkur']
}

for crop, mandi_list in mandis.items():
    for mandi in mandi_list:
        input_path = f"app/data/processed/{crop}_{mandi.lower()}_for_training.csv"
        if not os.path.exists(input_path):
            print(f"File not found: {input_path} (skipping)")
            continue
        df = pd.read_csv(input_path)
        print(f"[{crop} - {mandi}] Initial load: {df.shape[0]} rows")
        original_cols = list(df.columns)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        col_map = {
            'arrival_date': 'date',
            'price_date': 'date',
            'modal_price': 'modal_price',
            'modal_price_(rs./quintal)': 'modal_price'
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        if 'date' not in df.columns or 'modal_price' not in df.columns:
            print(f"Missing required columns in {input_path}! Columns found: {original_cols}")
            continue
        df = df[['date', 'modal_price'] + [col for col in df.columns if col not in ['date', 'modal_price']]].dropna(subset=['date', 'modal_price'])
        print(f"[{crop} - {mandi}] After selecting columns and dropna: {df.shape[0]} rows")
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df = df.sort_values('date')
        # Only use data from 2022 onwards
        df = df[df['date'] >= pd.Timestamp('2022-01-01')]
        print(f"[{crop} - {mandi}] After date filtering: {df.shape[0]} rows")
        # --- Outlier Removal (IQR method) ---
        Q1 = df['modal_price'].quantile(0.25)
        Q3 = df['modal_price'].quantile(0.75)
        IQR = Q3 - Q1
        mask = (df['modal_price'] >= Q1 - 1.5*IQR) & (df['modal_price'] <= Q3 + 1.5*IQR)
        df = df[mask]
        print(f"[{crop} - {mandi}] After outlier removal: {df.shape[0]} rows")
        # Group by date and aggregate by mean
        df = df.groupby('date').mean(numeric_only=True).reset_index()
        print(f"[{crop} - {mandi}] After grouping by date: {df.shape[0]} rows")
        df = df.sort_values('date').reset_index(drop=True)
        # --- Feature Engineering ---
        df['modal_price_lag1'] = df['modal_price'].shift(1)
        df['modal_price_lag2'] = df['modal_price'].shift(2)
        df['modal_price_lag3'] = df['modal_price'].shift(3)
        df['modal_price_lag5'] = df['modal_price'].shift(5)
        df['modal_price_lag7'] = df['modal_price'].shift(7)
        df['rolling_mean_7'] = df['modal_price'].rolling(window=7).mean()
        df['rolling_std_7'] = df['modal_price'].rolling(window=7).std()
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        # Drop NA after all feature creation
        df = df.dropna()
        print(f"[{crop} - {mandi}] After feature engineering and dropna: {df.shape[0]} rows")
        feature_cols = [
            'modal_price_lag1', 'modal_price_lag2', 'modal_price_lag3', 'modal_price_lag5', 'modal_price_lag7',
            'rolling_mean_7', 'rolling_std_7',
            'day_of_year', 'month', 'month_sin', 'month_cos'
        ]
        features = df[feature_cols].values
        target = df['modal_price'].values
        dates = df['date'].values.astype('datetime64[D]')
        print(f"{crop.title()} in {mandi}: Using features {feature_cols}")
        if features.size > 0 and target.size > 0:
            features_path = f"app/data/processed/features_{crop}_{mandi.lower()}.npy"
            target_path = f"app/data/processed/target_{crop}_{mandi.lower()}.npy"
            dates_path = f"app/data/processed/dates_{crop}_{mandi.lower()}.npy"
            np.save(features_path, features)
            np.save(target_path, target)
            np.save(dates_path, dates)
            print(f"Saved {features.shape[0]} samples for {crop.title()} in {mandi} to {features_path}, {target_path}, and {dates_path}")
        else:
            print(f"No data to save for {crop.title()} in {mandi}!") 