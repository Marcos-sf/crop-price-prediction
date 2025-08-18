import numpy as np
import os
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

mandis = {
    'arecanut': ['Sirsi', 'Yellapur', 'Siddapur', 'Shimoga', 'Sagar', 'Kumta'],
    'coconut': ['Bangalore', 'Arasikere', 'Channarayapatna', 'Ramanagara', 'Sira', 'Tumkur']
}

def create_weather_enhanced_features(df, price_col, date_col):
    """Create advanced features including weather data for better prediction"""
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Weather columns to use
    weather_cols = ['tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'pres']
    available_weather_cols = [col for col in weather_cols if col in df.columns]
    
    print(f"Available weather columns: {available_weather_cols}")
    
    features = []
    targets = []
    dates = []
    
    # Start from index 30 to have enough history
    for i in range(30, len(df)):
        current_price = df.iloc[i][price_col]
        current_date = df.iloc[i][date_col]
        
        # Basic lags
        lag1 = df.iloc[i-1][price_col]
        lag2 = df.iloc[i-2][price_col]
        lag3 = df.iloc[i-3][price_col]
        lag5 = df.iloc[i-5][price_col]
        lag7 = df.iloc[i-7][price_col]
        lag14 = df.iloc[i-14][price_col] if i >= 14 else lag7
        lag30 = df.iloc[i-30][price_col] if i >= 30 else lag14
        
        # Advanced rolling statistics
        past_7_prices = df.iloc[i-7:i][price_col].values
        past_14_prices = df.iloc[i-14:i][price_col].values if i >= 14 else past_7_prices
        past_30_prices = df.iloc[i-30:i][price_col].values if i >= 30 else past_14_prices
        
        rolling_mean_7 = np.mean(past_7_prices)
        rolling_mean_14 = np.mean(past_14_prices)
        rolling_mean_30 = np.mean(past_30_prices)
        
        rolling_std_7 = np.std(past_7_prices)
        rolling_std_14 = np.std(past_14_prices)
        rolling_std_30 = np.std(past_30_prices)
        
        # Price momentum and volatility features
        price_momentum_7 = current_price - lag7
        price_momentum_14 = current_price - lag14
        price_momentum_30 = current_price - lag30
        
        volatility_7 = rolling_std_7 / rolling_mean_7 if rolling_mean_7 > 0 else 0
        volatility_14 = rolling_std_14 / rolling_mean_14 if rolling_mean_14 > 0 else 0
        volatility_30 = rolling_std_30 / rolling_mean_30 if rolling_mean_30 > 0 else 0
        
        # Seasonal features
        day_of_year = current_date.timetuple().tm_yday
        month = current_date.month
        quarter = (month - 1) // 3 + 1
        
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        day_sin = np.sin(2 * np.pi * day_of_year / 365)
        day_cos = np.cos(2 * np.pi * day_of_year / 365)
        
        # Trend features
        trend_7 = (current_price - lag7) / lag7 if lag7 > 0 else 0
        trend_14 = (current_price - lag14) / lag14 if lag14 > 0 else 0
        trend_30 = (current_price - lag30) / lag30 if lag30 > 0 else 0
        
        # Weather features (current day and rolling averages)
        weather_features = []
        for col in available_weather_cols:
            # Current weather
            current_weather = df.iloc[i][col] if pd.notna(df.iloc[i][col]) else 0
            weather_features.append(current_weather)
            
            # Weather lags (1, 3, 7 days)
            weather_lag1 = df.iloc[i-1][col] if i > 0 and pd.notna(df.iloc[i-1][col]) else current_weather
            weather_lag3 = df.iloc[i-3][col] if i >= 3 and pd.notna(df.iloc[i-3][col]) else current_weather
            weather_lag7 = df.iloc[i-7][col] if i >= 7 and pd.notna(df.iloc[i-7][col]) else current_weather
            
            weather_features.extend([weather_lag1, weather_lag3, weather_lag7])
            
            # Rolling weather averages
            past_7_weather = df.iloc[i-7:i][col].dropna().values
            past_14_weather = df.iloc[i-14:i][col].dropna().values if i >= 14 else past_7_weather
            
            weather_mean_7 = np.mean(past_7_weather) if len(past_7_weather) > 0 else current_weather
            weather_mean_14 = np.mean(past_14_weather) if len(past_14_weather) > 0 else weather_mean_7
            
            weather_features.extend([weather_mean_7, weather_mean_14])
        
        # Create comprehensive feature vector
        base_features = [
            lag1, lag2, lag3, lag5, lag7, lag14, lag30,
            rolling_mean_7, rolling_mean_14, rolling_mean_30,
            rolling_std_7, rolling_std_14, rolling_std_30,
            price_momentum_7, price_momentum_14, price_momentum_30,
            volatility_7, volatility_14, volatility_30,
            day_of_year, month, quarter,
            month_sin, month_cos, day_sin, day_cos,
            trend_7, trend_14, trend_30
        ]
        
        feature_vector = base_features + weather_features
        features.append(feature_vector)
        targets.append(current_price)
        dates.append(current_date)
    
    return np.array(features), np.array(targets), np.array(dates)

def train_weather_enhanced_xgboost(X_train, y_train, X_test, y_test, crop, mandi):
    """Train XGBoost model with weather features"""
    print(f"Training weather-enhanced XGBoost for {crop} in {mandi}")
    
    # Hyperparameter tuning for weather-enhanced model
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5]
    }
    
    best_score = float('inf')
    best_params = None
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    for n_estimators in param_grid['n_estimators']:
        for max_depth in param_grid['max_depth']:
            for learning_rate in param_grid['learning_rate']:
                for subsample in param_grid['subsample']:
                    for colsample_bytree in param_grid['colsample_bytree']:
                        for reg_alpha in param_grid['reg_alpha']:
                            for reg_lambda in param_grid['reg_lambda']:
                                params = {
                                    'n_estimators': n_estimators,
                                    'max_depth': max_depth,
                                    'learning_rate': learning_rate,
                                    'subsample': subsample,
                                    'colsample_bytree': colsample_bytree,
                                    'reg_alpha': reg_alpha,
                                    'reg_lambda': reg_lambda,
                                    'random_state': 42,
                                    'n_jobs': -1
                                }
                                
                                scores = []
                                for train_idx, val_idx in tscv.split(X_train):
                                    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                                    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                                    
                                    model = xgb.XGBRegressor(**params)
                                    model.fit(X_fold_train, y_fold_train)
                                    y_pred = model.predict(X_fold_val)
                                    score = mean_squared_error(y_fold_val, y_pred)
                                    scores.append(score)
                                
                                avg_score = np.mean(scores)
                                if avg_score < best_score:
                                    best_score = avg_score
                                    best_params = params
    
    print(f"Best parameters: {best_params}")
    
    # Train final model with best parameters
    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = final_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Weather-enhanced XGBoost Results:")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
    
    return final_model, best_params

def train_weather_enhanced_lstm(X_train, y_train, X_test, y_test, crop, mandi):
    """Train LSTM model with weather features"""
    print(f"Training weather-enhanced LSTM for {crop} in {mandi}")
    
    # Reshape data for LSTM (samples, timesteps, features)
    timesteps = 7
    X_train_lstm = []
    y_train_lstm = []
    
    for i in range(timesteps, len(X_train)):
        X_train_lstm.append(X_train[i-timesteps:i])
        y_train_lstm.append(y_train[i])
    
    X_train_lstm = np.array(X_train_lstm)
    y_train_lstm = np.array(y_train_lstm)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    # Reshape scaled data for LSTM
    X_train_lstm_scaled = []
    for i in range(timesteps, len(X_train_scaled)):
        X_train_lstm_scaled.append(X_train_scaled[i-timesteps:i])
    X_train_lstm_scaled = np.array(X_train_lstm_scaled)
    
    # Build LSTM model
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(timesteps, X_train.shape[1])),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
    
    # Train model
    history = model.fit(
        X_train_lstm_scaled, y_train_lstm,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate
    X_test_lstm = []
    for i in range(timesteps, len(X_test_scaled)):
        X_test_lstm.append(X_test_scaled[i-timesteps:i])
    X_test_lstm = np.array(X_test_lstm)
    
    y_pred = model.predict(X_test_lstm)
    mse = mean_squared_error(y_test[timesteps:], y_pred)
    mae = mean_absolute_error(y_test[timesteps:], y_pred)
    r2 = r2_score(y_test[timesteps:], y_pred)
    
    print(f"Weather-enhanced LSTM Results:")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
    
    return model, scaler

def main():
    """Main training function with weather integration"""
    print("Starting weather-enhanced model training...")
    
    for crop in mandis.keys():
        for mandi in mandis[crop]:
            print(f"\n{'='*50}")
            print(f"Processing {crop} in {mandi}")
            print(f"{'='*50}")
            
            # Try to load weather-enhanced data first
            weather_file = f"app/data/processed/{crop.lower()}_{mandi.lower()}_with_weather.csv"
            regular_file = f"app/data/processed/{crop.lower()}_{mandi.lower()}_for_training.csv"
            
            if os.path.exists(weather_file):
                print(f"Using weather-enhanced data: {weather_file}")
                df = pd.read_csv(weather_file)
            elif os.path.exists(regular_file):
                print(f"Using regular data (no weather): {regular_file}")
                df = pd.read_csv(regular_file)
            else:
                print(f"No data file found for {crop} in {mandi}, skipping...")
                continue
            
            # Find columns
            date_col = None
            price_col = None
            
            for col in df.columns:
                if col.strip().lower() in ['date', 'arrival_date', 'price_date']:
                    date_col = col
                elif col.strip().lower() in ['modal_price', 'modal price (rs./quintal)']:
                    price_col = col
            
            if not date_col or not price_col:
                print(f"Required columns not found in {crop} {mandi}, skipping...")
                continue
            
            # Convert date and clean data
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col, price_col])
            df = df.sort_values(date_col)
            
            if len(df) < 50:
                print(f"Not enough data for {crop} in {mandi}, skipping...")
                continue
            
            # Create features
            X, y, dates = create_weather_enhanced_features(df, price_col, date_col)
            
            if len(X) < 20:
                print(f"Not enough features for {crop} in {mandi}, skipping...")
                continue
            
            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            print(f"Training set: {len(X_train)} samples")
            print(f"Test set: {len(X_test)} samples")
            print(f"Feature dimensions: {X_train.shape[1]}")
            
            # Train XGBoost
            try:
                xgb_model, xgb_params = train_weather_enhanced_xgboost(X_train, y_train, X_test, y_test, crop, mandi)
                
                # Save XGBoost model
                model_path = f"app/data/processed/xgb_weather_{crop.lower()}_{mandi.lower()}.joblib"
                joblib.dump(xgb_model, model_path)
                print(f"Saved XGBoost model: {model_path}")
                
                # Save parameters
                params_path = f"app/data/processed/xgb_weather_params_{crop.lower()}_{mandi.lower()}.joblib"
                joblib.dump(xgb_params, params_path)
                
            except Exception as e:
                print(f"Error training XGBoost for {crop} {mandi}: {e}")
            
            # Train LSTM
            try:
                lstm_model, lstm_scaler = train_weather_enhanced_lstm(X_train, y_train, X_test, y_test, crop, mandi)
                
                # Save LSTM model
                model_path = f"app/data/processed/lstm_weather_{crop.lower()}_{mandi.lower()}.h5"
                lstm_model.save(model_path)
                print(f"Saved LSTM model: {model_path}")
                
                # Save scaler
                scaler_path = f"app/data/processed/lstm_weather_scaler_{crop.lower()}_{mandi.lower()}.joblib"
                joblib.dump(lstm_scaler, scaler_path)
                
            except Exception as e:
                print(f"Error training LSTM for {crop} {mandi}: {e}")
            
            # Save feature data
            np.save(f"app/data/processed/features_weather_{crop.lower()}_{mandi.lower()}.npy", X)
            np.save(f"app/data/processed/target_weather_{crop.lower()}_{mandi.lower()}.npy", y)
            np.save(f"app/data/processed/dates_weather_{crop.lower()}_{mandi.lower()}.npy", dates)
            
            print(f"Completed training for {crop} in {mandi}")

if __name__ == "__main__":
    main() 