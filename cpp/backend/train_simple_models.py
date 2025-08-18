import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def create_simple_features(df, date_col, price_col):
    """Create simple features from basic data"""
    # Sort by date
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Create lag features
    df['modal_price_lag1'] = df[price_col].shift(1)
    df['modal_price_lag2'] = df[price_col].shift(2)
    df['modal_price_lag3'] = df[price_col].shift(3)
    df['modal_price_lag5'] = df[price_col].shift(5)
    df['modal_price_lag7'] = df[price_col].shift(7)
    
    # Create rolling statistics
    df['rolling_mean_7'] = df[price_col].rolling(window=7).mean()
    df['rolling_std_7'] = df[price_col].rolling(window=7).std()
    
    # Create temporal features
    df[date_col] = pd.to_datetime(df[date_col])
    df['day_of_year'] = df[date_col].dt.dayofyear
    df['month'] = df[date_col].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Drop rows with NaN values (due to lag features)
    df = df.dropna()
    
    return df

def train_simple_model(crop, mandi):
    """Train a simple model that matches our API structure"""
    print(f"Training simple model for {crop}-{mandi}...")
    
    # Load data
    data_path = f"app/data/processed/{crop}_{mandi}_for_training.csv"
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return False
    
    df = pd.read_csv(data_path)
    print(f"Loaded data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Find date and price columns
    date_col = None
    price_col = None
    
    for col in df.columns:
        col_lower = col.strip().lower()
        if 'date' in col_lower or 'arrival' in col_lower:
            date_col = col
        elif 'modal' in col_lower and 'price' in col_lower:
            price_col = col
    
    if not date_col or not price_col:
        print(f"Could not find date or price columns")
        print(f"Date column: {date_col}, Price column: {price_col}")
        return False
    
    print(f"Using date column: {date_col}")
    print(f"Using price column: {price_col}")
    
    # Create features
    df_features = create_simple_features(df, date_col, price_col)
    print(f"Features created, shape: {df_features.shape}")
    
    # Define feature columns (matching our API)
    feature_cols = [
        'modal_price_lag1', 'modal_price_lag2', 'modal_price_lag3',
        'modal_price_lag5', 'modal_price_lag7', 'rolling_mean_7',
        'rolling_std_7', 'day_of_year', 'month', 'month_sin', 'month_cos'
    ]
    
    # Check if all features exist
    missing_features = [col for col in feature_cols if col not in df_features.columns]
    if missing_features:
        print(f"Missing features: {missing_features}")
        return False
    
    # Prepare data
    X = df_features[feature_cols].values
    y = df_features[price_col].values
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    accuracy = max(0, r2 * 100)  # Convert R² to percentage
    
    print(f"Model Performance:")
    print(f"  MAE: {mae:.2f}")
    print(f"  R²: {r2:.4f}")
    print(f"  Accuracy: {accuracy:.2f}%")
    
    # Save model and scaler
    model_path = f"app/data/processed/simple_{crop}_{mandi}.joblib"
    scaler_path = f"app/data/processed/simple_scaler_{crop}_{mandi}.joblib"
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    
    return True

def main():
    """Train simple models for all crop-mandi combinations"""
    combinations = [
        ("arecanut", "sirsi"), ("arecanut", "yellapur"), ("arecanut", "siddapur"),
        ("arecanut", "shimoga"), ("arecanut", "sagar"), ("arecanut", "kumta"),
        ("coconut", "bangalore"), ("coconut", "arasikere"), ("coconut", "channarayapatna"),
        ("coconut", "ramanagara"), ("coconut", "sira"), ("coconut", "tumkur")
    ]
    
    results = []
    
    for crop, mandi in combinations:
        print(f"\n{'='*50}")
        success = train_simple_model(crop, mandi)
        results.append((crop, mandi, success))
    
    print(f"\n{'='*50}")
    print("TRAINING SUMMARY:")
    print(f"{'='*50}")
    
    for crop, mandi, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{crop}-{mandi}: {status}")

if __name__ == "__main__":
    main()
