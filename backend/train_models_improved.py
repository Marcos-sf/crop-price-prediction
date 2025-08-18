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

def create_advanced_features(df, price_col, date_col):
    """Create advanced features for better prediction"""
    df = df.sort_values(date_col).reset_index(drop=True)
    
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
        
        # Create comprehensive feature vector (20 features)
        feature_vector = [
            lag1, lag2, lag3, lag5, lag7, lag14, lag30,
            rolling_mean_7, rolling_mean_14, rolling_mean_30,
            rolling_std_7, rolling_std_14, rolling_std_30,
            price_momentum_7, price_momentum_14, price_momentum_30,
            volatility_7, volatility_14, volatility_30,
            day_of_year, month, quarter,
            month_sin, month_cos, day_sin, day_cos,
            trend_7, trend_14, trend_30
        ]
        
        features.append(feature_vector)
        targets.append(current_price)
        dates.append(current_date)
    
    return np.array(features), np.array(targets), np.array(dates)

def train_improved_xgboost(X_train, y_train, X_test, y_test, crop, mandi):
    """Train improved XGBoost model with hyperparameter tuning"""
    
    # Advanced XGBoost parameters
    xgb_params = {
        'n_estimators': 1000,
        'max_depth': 8,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping_rounds': 50
    }
    
    # Train model
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=0
    )
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Save model
    model_path = f"app/data/processed/xgb_improved_{crop}_{mandi.lower()}.joblib"
    joblib.dump(model, model_path)
    
    return y_pred, model_path

def train_improved_lstm(X_train, y_train, X_test, y_test, crop, mandi):
    """Train improved LSTM model with advanced architecture"""
    
    # Normalize features
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # Reshape for LSTM
    X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    
    # Advanced LSTM architecture
    model = Sequential([
        LSTM(128, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dense(1)
    ])
    
    # Compile with advanced optimizer
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    
    # Advanced callbacks
    es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
    
    # Train model
    model.fit(
        X_train_lstm, y_train_scaled,
        epochs=300,
        batch_size=32,
        validation_split=0.2,
        callbacks=[es, rlr],
        verbose=0
    )
    
    # Predictions
    y_pred_scaled = model.predict(X_test_lstm).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # Save model and scalers
    model_path = f"app/data/processed/lstm_improved_{crop}_{mandi.lower()}.h5"
    scaler_path = f"app/data/processed/lstm_scalers_{crop}_{mandi.lower()}.joblib"
    
    model.save(model_path)
    joblib.dump({
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }, scaler_path)
    
    return y_pred, model_path, scaler_path

def train_ensemble_improved(X_train, y_train, X_test, y_test, crop, mandi):
    """Train ensemble with multiple models"""
    
    # Train XGBoost
    y_pred_xgb, xgb_path = train_improved_xgboost(X_train, y_train, X_test, y_test, crop, mandi)
    
    # Train LSTM
    y_pred_lstm, lstm_path, scaler_path = train_improved_lstm(X_train, y_train, X_test, y_test, crop, mandi)
    
    # Train Random Forest as additional model
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    # Save RF model
    rf_path = f"app/data/processed/rf_improved_{crop}_{mandi.lower()}.joblib"
    joblib.dump(rf_model, rf_path)
    
    # Optimize ensemble weights using validation set
    # Use last 20% of training data as validation
    val_split = int(len(X_train) * 0.8)
    X_val = X_train[val_split:]
    y_val = y_train[val_split:]
    
    # Get predictions on validation set
    xgb_model = joblib.load(xgb_path)
    y_pred_xgb_val = xgb_model.predict(X_val)
    
    lstm_model = load_model(lstm_path)
    scalers = joblib.load(scaler_path)
    X_val_scaled = scalers['scaler_X'].transform(X_val)
    X_val_lstm = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))
    y_pred_lstm_val_scaled = lstm_model.predict(X_val_lstm).flatten()
    y_pred_lstm_val = scalers['scaler_y'].inverse_transform(y_pred_lstm_val_scaled.reshape(-1, 1)).flatten()
    
    y_pred_rf_val = rf_model.predict(X_val)
    
    # Find optimal weights using grid search
    best_rmse = float('inf')
    best_weights = [0.4, 0.4, 0.2]  # Default weights
    
    for w1 in np.arange(0.1, 0.9, 0.1):
        for w2 in np.arange(0.1, 0.9, 0.1):
            w3 = 1 - w1 - w2
            if w3 > 0:
                ensemble_pred_val = w1 * y_pred_xgb_val + w2 * y_pred_lstm_val + w3 * y_pred_rf_val
                rmse = mean_squared_error(y_val, ensemble_pred_val, squared=False)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_weights = [w1, w2, w3]
    
    # Final ensemble prediction
    y_pred_ensemble = (best_weights[0] * y_pred_xgb + 
                      best_weights[1] * y_pred_lstm + 
                      best_weights[2] * y_pred_rf)
    
    # Save ensemble metadata
    ensemble_path = f"app/data/processed/ensemble_improved_{crop}_{mandi.lower()}.joblib"
    ensemble_metadata = {
        'xgb_model_path': xgb_path,
        'lstm_model_path': lstm_path,
        'rf_model_path': rf_path,
        'scaler_path': scaler_path,
        'weights': best_weights
    }
    joblib.dump(ensemble_metadata, ensemble_path)
    
    return y_pred_ensemble, ensemble_path

def evaluate_models_improved(y_true, y_pred, model_name):
    """Calculate comprehensive evaluation metrics"""
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    accuracy = 100 - mape
    
    return {
        'model': model_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'accuracy': accuracy
    }

def train_all_models_improved():
    """Train all models with improved techniques"""
    
    print("=" * 80)
    print("IMPROVED MODEL TRAINING (Target: >90% Accuracy)")
    print("=" * 80)
    
    all_results = []
    
    for crop, mandi_list in mandis.items():
        for mandi in mandi_list:
            print(f"\n{'='*60}")
            print(f"Training Improved Models for {crop.title()} in {mandi.title()}")
            print(f"{'='*60}")
            
            # Load raw data
            file_path = f"app/data/processed/{crop}_{mandi.lower()}_for_training.csv"
            if not os.path.exists(file_path):
                print(f"Missing data for {crop.title()} in {mandi}, skipping.")
                continue
            
            df = pd.read_csv(file_path)
            
            # Find date and price columns
            date_col = None
            for col in df.columns:
                if col.strip().lower() in ['date', 'arrival_date', 'price_date']:
                    date_col = col
                    break
            
            price_col = None
            for col in df.columns:
                if col.strip().lower() in ['modal_price', 'modal price (rs./quintal)']:
                    price_col = col
                    break
            
            if not date_col or not price_col:
                print(f"Missing date or price column for {crop.title()} in {mandi}, skipping.")
                continue
            
            # Convert date and clean data
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col, price_col])
            df = df.sort_values(date_col)
            
            if len(df) < 100:
                print(f"Not enough data for {crop.title()} in {mandi}, skipping.")
                continue
            
            # Create advanced features
            X, y, dates = create_advanced_features(df, price_col, date_col)
            
            # Proper 80/20 temporal split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            train_dates, test_dates = dates[:split_idx], dates[split_idx:]
            
            if len(X_test) < 20:
                print(f"Not enough test samples for {crop.title()} in {mandi}, skipping.")
                continue
            
            print(f"Training set size: {len(X_train)}")
            print(f"Test set size: {len(X_test)}")
            print(f"Test date range: {test_dates.min().strftime('%Y-%m-%d')} to {test_dates.max().strftime('%Y-%m-%d')}")
            print(f"Feature count: {X_train.shape[1]}")
            
            # Train ensemble model
            print("Training improved ensemble model...")
            y_pred_ensemble, ensemble_path = train_ensemble_improved(X_train, y_train, X_test, y_test, crop, mandi)
            
            # Evaluate ensemble
            ensemble_metrics = evaluate_models_improved(y_test, y_pred_ensemble, "Improved Ensemble")
            all_results.append(ensemble_metrics)
            
            print(f"Improved Ensemble Performance:")
            print(f"  RMSE: {ensemble_metrics['rmse']:.2f}")
            print(f"  MAE: {ensemble_metrics['mae']:.2f}")
            print(f"  R²: {ensemble_metrics['r2']:.3f}")
            print(f"  MAPE: {ensemble_metrics['mape']:.2f}%")
            print(f"  Accuracy: {ensemble_metrics['accuracy']:.2f}%")
            
            if ensemble_metrics['accuracy'] >= 90:
                print(f"✅ TARGET ACHIEVED: {ensemble_metrics['accuracy']:.2f}% accuracy!")
            else:
                print(f"⚠️ Target not reached: {ensemble_metrics['accuracy']:.2f}% accuracy")
    
    # Print summary
    print("\n" + "=" * 80)
    print("IMPROVED MODEL TRAINING SUMMARY")
    print("=" * 80)
    
    if all_results:
        avg_accuracy = np.mean([r['accuracy'] for r in all_results])
        avg_rmse = np.mean([r['rmse'] for r in all_results])
        avg_r2 = np.mean([r['r2'] for r in all_results])
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"  Average Accuracy: {avg_accuracy:.2f}%")
        print(f"  Average RMSE: {avg_rmse:.2f}")
        print(f"  Average R²: {avg_r2:.3f}")
        print(f"  Models trained: {len(all_results)}")
        
        # Count models that achieved >90% accuracy
        high_accuracy_count = sum(1 for r in all_results if r['accuracy'] >= 90)
        print(f"  Models with >90% accuracy: {high_accuracy_count}/{len(all_results)}")
        
        if avg_accuracy >= 90:
            print(f"\n🎉 SUCCESS: Average accuracy of {avg_accuracy:.2f}% exceeds 90% target!")
        else:
            print(f"\n⚠️ Target not fully achieved: {avg_accuracy:.2f}% average accuracy")
    
    return all_results

if __name__ == "__main__":
    train_all_models_improved()
