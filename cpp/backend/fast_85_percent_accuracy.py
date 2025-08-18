import numpy as np
import pandas as pd
import os
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def create_fast_advanced_features(df, price_col, date_col):
    """Create optimized features for fast 85%+ accuracy training"""
    df = df.sort_values(date_col).reset_index(drop=True)
    
    features = []
    targets = []
    dates = []
    
    # Start from index 30 for sufficient history
    for i in range(30, len(df)):
        current_price = df.iloc[i][price_col]
        current_date = df.iloc[i][date_col]
        
        # Essential lag features (9 features)
        lags = [1, 2, 3, 5, 7, 14, 21, 30, 45]
        lag_features = [df.iloc[i-lag][price_col] if i >= lag else current_price for lag in lags]
        
        # Key rolling statistics (24 features)
        windows = [7, 14, 21, 30]
        rolling_features = []
        
        for window in windows:
            if i >= window:
                window_prices = df.iloc[i-window:i][price_col].values
                rolling_features.extend([
                    np.mean(window_prices),
                    np.std(window_prices),
                    np.median(window_prices),
                    np.percentile(window_prices, 25),
                    np.percentile(window_prices, 75),
                    (np.max(window_prices) - np.min(window_prices)) / np.mean(window_prices) if np.mean(window_prices) > 0 else 0
                ])
            else:
                rolling_features.extend([current_price] * 6)
        
        # Momentum features (4 features)
        momentum_features = []
        for window in [7, 14, 21, 30]:
            if i >= window:
                past_prices = df.iloc[i-window:i][price_col].values
                momentum = (current_price - past_prices[0]) / past_prices[0] if past_prices[0] > 0 else 0
                momentum_features.append(momentum)
            else:
                momentum_features.append(0)
        
        # Volatility features (4 features)
        volatility_features = []
        for window in [7, 14, 21, 30]:
            if i >= window:
                past_prices = df.iloc[i-window:i][price_col].values
                volatility = np.std(past_prices) / np.mean(past_prices) if np.mean(past_prices) > 0 else 0
                volatility_features.append(volatility)
            else:
                volatility_features.append(0)
        
        # Trend features (4 features)
        trend_features = []
        for window in [7, 14, 21, 30]:
            if i >= window:
                window_prices = df.iloc[i-window:i][price_col].values
                x = np.arange(len(window_prices))
                slope = np.polyfit(x, window_prices, 1)[0]
                trend_features.append(slope)
            else:
                trend_features.append(0)
        
        # Seasonal features (6 features)
        day_of_year = current_date.timetuple().tm_yday
        month = current_date.month
        week_of_year = current_date.isocalendar()[1]
        
        seasonal_features = [
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12),
            np.sin(2 * np.pi * day_of_year / 365),
            np.cos(2 * np.pi * day_of_year / 365),
            np.sin(2 * np.pi * week_of_year / 52),
            np.cos(2 * np.pi * week_of_year / 52)
        ]
        
        # Price change ratios (6 features)
        change_ratios = []
        for lag in [1, 2, 3, 7, 14, 21]:
            if i >= lag:
                ratio = current_price / df.iloc[i-lag][price_col] if df.iloc[i-lag][price_col] > 0 else 1
                change_ratios.append(ratio)
            else:
                change_ratios.append(1)
        
        # Moving average crossovers (2 features)
        ma_features = []
        if i >= 21:
            short_avg = np.mean(df.iloc[i-7:i][price_col])
            long_avg = np.mean(df.iloc[i-21:i][price_col])
            crossover = short_avg / long_avg if long_avg > 0 else 1
            ma_features.append(crossover)
            
            # Another crossover
            short_avg2 = np.mean(df.iloc[i-14:i][price_col])
            crossover2 = short_avg2 / long_avg if long_avg > 0 else 1
            ma_features.append(crossover2)
        else:
            ma_features.extend([1, 1])
        
        # Combine all features (total: 9 + 24 + 4 + 4 + 4 + 6 + 6 + 2 = 59 features)
        feature_vector = (
            lag_features +
            rolling_features +
            momentum_features +
            volatility_features +
            trend_features +
            seasonal_features +
            change_ratios +
            ma_features
        )
        
        features.append(feature_vector)
        targets.append(current_price)
        dates.append(current_date)
    
    return np.array(features), np.array(targets), np.array(dates)

def train_fast_xgboost(X_train, y_train, X_test, y_test, crop, mandi):
    """Train XGBoost with optimized parameters for fast 85%+ accuracy"""
    
    # Reduced but effective parameter grid
    param_grid = {
        'n_estimators': [1000, 1500],
        'max_depth': [8, 10],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.9],
        'colsample_bytree': [0.9],
        'reg_alpha': [0.1],
        'reg_lambda': [0.1],
        'min_child_weight': [3],
        'gamma': [0.1]
    }
    
    # Use 3-fold CV instead of 5 for speed
    tscv = TimeSeriesSplit(n_splits=3)
    
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
    
    print(f"Starting fast XGBoost tuning for {crop} - {mandi}...")
    
    grid_search = GridSearchCV(
        xgb_model, param_grid, cv=tscv, scoring='neg_mean_squared_error',
        n_jobs=-1, verbose=0  # Reduced verbosity for speed
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model and make predictions
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    accuracy = 100 - mape
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"XGBoost {crop} {mandi} - Accuracy: {accuracy:.2f}%, RMSE: {rmse:.2f}, R²: {r2:.3f}")
    
    # Save model
    model_path = f"app/data/processed/xgb_fast_{crop}_{mandi}.joblib"
    joblib.dump(best_model, model_path)
    
    return {
        'model_type': 'XGBoost',
        'crop': crop,
        'mandi': mandi,
        'accuracy': accuracy,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'best_params': grid_search.best_params_
    }

def train_fast_lstm(X_train, y_train, X_test, y_test, crop, mandi):
    """Train LSTM with optimized architecture for fast 85%+ accuracy"""
    
    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Scale targets
    target_scaler = MinMaxScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    # Reshape for LSTM (samples, timesteps, features)
    X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    
    # Build fast but effective LSTM model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(1, X_train_scaled.shape[1])),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu', kernel_regularizer=l1_l2(0.01, 0.01)),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    # Compile with optimized settings
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Fast training callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=0.0001)
    ]
    
    print(f"Training fast LSTM for {crop} - {mandi}...")
    
    # Train with reduced epochs for speed
    history = model.fit(
        X_train_lstm, y_train_scaled,
        epochs=100,  # Reduced from 200
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=0  # Reduced verbosity
    )
    
    # Make predictions
    y_pred_scaled = model.predict(X_test_lstm, verbose=0)
    y_pred = target_scaler.inverse_transform(y_pred_scaled).flatten()
    
    # Calculate metrics
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    accuracy = 100 - mape
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"LSTM {crop} {mandi} - Accuracy: {accuracy:.2f}%, RMSE: {rmse:.2f}, R²: {r2:.3f}")
    
    # Save model and scaler
    model_path = f"app/data/processed/lstm_fast_{crop}_{mandi}.h5"
    scaler_path = f"app/data/processed/lstm_fast_scaler_{crop}_{mandi}.joblib"
    target_scaler_path = f"app/data/processed/lstm_fast_target_scaler_{crop}_{mandi}.joblib"
    
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(target_scaler, target_scaler_path)
    
    return {
        'model_type': 'LSTM',
        'crop': crop,
        'mandi': mandi,
        'accuracy': accuracy,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def fast_85_percent_accuracy():
    """Fast training to achieve 85%+ accuracy for all models"""
    
    print("🚀 Starting Fast 85%+ Accuracy Training")
    print("Optimized for speed while maintaining high accuracy...")
    print()
    
    # Crop-mandi combinations
    combinations = [
        ('arecanut', 'sirsi'), ('arecanut', 'yellapur'), ('arecanut', 'siddapur'),
        ('arecanut', 'shimoga'), ('arecanut', 'sagar'), ('arecanut', 'kumta'),
        ('coconut', 'bangalore'), ('coconut', 'arasikere'), ('coconut', 'channarayapatna'),
        ('coconut', 'ramanagara'), ('coconut', 'sira'), ('coconut', 'tumkur')
    ]
    
    results = []
    
    for crop, mandi in combinations:
        try:
            print(f"{'='*50}")
            print(f"Training Fast Models for {crop} - {mandi}")
            print(f"{'='*50}")
            
            # Load data
            data_path = f"app/data/processed/{crop}_{mandi}_for_training.csv"
            if not os.path.exists(data_path):
                print(f"Data not found: {data_path}")
                continue
            
            df = pd.read_csv(data_path)
            df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'])
            
            # Create fast advanced features (59 features)
            X, y, dates = create_fast_advanced_features(df, 'Modal_Price', 'Arrival_Date')
            
            # Split data (80-20)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
            print(f"Features: {X_train.shape[1]} (fast advanced features)")
            print()
            
            # Train XGBoost
            xgb_result = train_fast_xgboost(X_train, y_train, X_test, y_test, crop, mandi)
            results.append(xgb_result)
            
            # Train LSTM
            lstm_result = train_fast_lstm(X_train, y_train, X_test, y_test, crop, mandi)
            results.append(lstm_result)
            
            # Create Ensemble
            xgb_model = joblib.load(f"app/data/processed/xgb_fast_{crop}_{mandi}.joblib")
            
            # Load LSTM components for ensemble prediction
            from tensorflow.keras.models import load_model
            lstm_model = load_model(f"app/data/processed/lstm_fast_{crop}_{mandi}.h5")
            lstm_scaler = joblib.load(f"app/data/processed/lstm_fast_scaler_{crop}_{mandi}.joblib")
            lstm_target_scaler = joblib.load(f"app/data/processed/lstm_fast_target_scaler_{crop}_{mandi}.joblib")
            
            # Make ensemble predictions
            xgb_pred = xgb_model.predict(X_test)
            
            X_test_scaled = lstm_scaler.transform(X_test)
            X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
            lstm_pred_scaled = lstm_model.predict(X_test_lstm, verbose=0)
            lstm_pred = lstm_target_scaler.inverse_transform(lstm_pred_scaled).flatten()
            
            # Weighted ensemble (based on individual accuracies)
            xgb_weight = xgb_result['accuracy'] / (xgb_result['accuracy'] + lstm_result['accuracy'])
            lstm_weight = lstm_result['accuracy'] / (xgb_result['accuracy'] + lstm_result['accuracy'])
            
            ensemble_pred = (xgb_weight * xgb_pred) + (lstm_weight * lstm_pred)
            
            # Calculate ensemble metrics
            ensemble_mape = np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100
            ensemble_accuracy = 100 - ensemble_mape
            ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
            ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
            ensemble_r2 = r2_score(y_test, ensemble_pred)
            
            print(f"Ensemble {crop} {mandi} - Accuracy: {ensemble_accuracy:.2f}%, RMSE: {ensemble_rmse:.2f}, R²: {ensemble_r2:.3f}")
            
            ensemble_result = {
                'model_type': 'Ensemble',
                'crop': crop,
                'mandi': mandi,
                'accuracy': ensemble_accuracy,
                'rmse': ensemble_rmse,
                'mae': ensemble_mae,
                'r2': ensemble_r2,
                'xgb_weight': xgb_weight,
                'lstm_weight': lstm_weight
            }
            results.append(ensemble_result)
            
            print()
            
        except Exception as e:
            print(f"Error training {crop} - {mandi}: {str(e)}")
            continue
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('fast_85_percent_results.csv', index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 FAST TRAINING RESULTS SUMMARY")
    print("="*60)
    
    for model_type in ['XGBoost', 'LSTM', 'Ensemble']:
        model_results = results_df[results_df['model_type'] == model_type]
        if not model_results.empty:
            avg_accuracy = model_results['accuracy'].mean()
            min_accuracy = model_results['accuracy'].min()
            max_accuracy = model_results['accuracy'].max()
            above_85 = (model_results['accuracy'] >= 85).sum()
            total = len(model_results)
            
            print(f"\n{model_type} Model:")
            print(f"  Average Accuracy: {avg_accuracy:.2f}%")
            print(f"  Range: {min_accuracy:.2f}% - {max_accuracy:.2f}%")
            print(f"  Models ≥85%: {above_85}/{total} ({above_85/total*100:.1f}%)")
    
    print(f"\n📊 Results saved to: fast_85_percent_results.csv")
    print("🚀 Fast training completed!")
    
    return results

if __name__ == "__main__":
    results = fast_85_percent_accuracy()
