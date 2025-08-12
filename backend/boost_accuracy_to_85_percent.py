import numpy as np
import pandas as pd
import os
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def create_ultra_advanced_features(df, price_col, date_col):
    """Create ultra-advanced features for 85%+ accuracy"""
    df = df.sort_values(date_col).reset_index(drop=True)
    
    features = []
    targets = []
    dates = []
    
    # Start from index 60 to have enough history
    for i in range(60, len(df)):
        current_price = df.iloc[i][price_col]
        current_date = df.iloc[i][date_col]
        
        # Extended lag features (11 features)
        lags = [1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60]
        lag_features = [df.iloc[i-lag][price_col] if i >= lag else current_price for lag in lags]
        
        # Advanced rolling statistics (48 features)
        windows = [7, 14, 21, 30, 45, 60]
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
                    np.max(window_prices),
                    np.min(window_prices),
                    (np.max(window_prices) - np.min(window_prices)) / np.mean(window_prices) if np.mean(window_prices) > 0 else 0
                ])
            else:
                rolling_features.extend([current_price] * 8)
        
        # Momentum and volatility features (8 features)
        momentum_features = []
        volatility_features = []
        
        for window in [7, 14, 21, 30]:
            if i >= window:
                past_prices = df.iloc[i-window:i][price_col].values
                momentum = (current_price - past_prices[0]) / past_prices[0] if past_prices[0] > 0 else 0
                volatility = np.std(past_prices) / np.mean(past_prices) if np.mean(past_prices) > 0 else 0
                momentum_features.append(momentum)
                volatility_features.append(volatility)
            else:
                momentum_features.append(0)
                volatility_features.append(0)
        
        # Advanced trend features (4 features)
        trend_features = []
        for window in [7, 14, 21, 30]:
            if i >= window:
                window_prices = df.iloc[i-window:i][price_col].values
                x = np.arange(len(window_prices))
                slope = np.polyfit(x, window_prices, 1)[0]
                trend_features.append(slope)
            else:
                trend_features.append(0)
        
        # Enhanced seasonal features (8 features)
        day_of_year = current_date.timetuple().tm_yday
        month = current_date.month
        week_of_year = current_date.isocalendar()[1]
        day_of_week = current_date.weekday()
        
        seasonal_features = [
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12),
            np.sin(2 * np.pi * day_of_year / 365),
            np.cos(2 * np.pi * day_of_year / 365),
            np.sin(2 * np.pi * week_of_year / 52),
            np.cos(2 * np.pi * week_of_year / 52),
            np.sin(2 * np.pi * day_of_week / 7),
            np.cos(2 * np.pi * day_of_week / 7)
        ]
        
        # Price change ratios (8 features)
        change_ratios = []
        for lag in [1, 2, 3, 5, 7, 14, 21, 30]:
            if i >= lag:
                ratio = current_price / df.iloc[i-lag][price_col] if df.iloc[i-lag][price_col] > 0 else 1
                change_ratios.append(ratio)
            else:
                change_ratios.append(1)
        
        # Moving average crossovers (4 features)
        ma_features = []
        for short_ma in [7, 14]:
            for long_ma in [21, 30]:
                if i >= long_ma:
                    short_avg = np.mean(df.iloc[i-short_ma:i][price_col])
                    long_avg = np.mean(df.iloc[i-long_ma:i][price_col])
                    crossover = short_avg / long_avg if long_avg > 0 else 1
                    ma_features.append(crossover)
                else:
                    ma_features.append(1)
        
        # RSI-like features (3 features)
        rsi_features = []
        for period in [14, 21, 30]:
            if i >= period:
                period_prices = df.iloc[i-period:i][price_col].values
                gains = np.where(np.diff(period_prices) > 0, np.diff(period_prices), 0)
                losses = np.where(np.diff(period_prices) < 0, -np.diff(period_prices), 0)
                avg_gain = np.mean(gains) if len(gains) > 0 else 0
                avg_loss = np.mean(losses) if len(losses) > 0 else 0
                rs = avg_gain / avg_loss if avg_loss > 0 else 1
                rsi = 100 - (100 / (1 + rs))
                rsi_features.append(rsi)
            else:
                rsi_features.append(50)
        
        # Combine all features (total: 11 + 48 + 8 + 4 + 8 + 8 + 4 + 3 = 94 features)
        feature_vector = (
            lag_features +
            rolling_features +
            momentum_features +
            volatility_features +
            trend_features +
            seasonal_features +
            change_ratios +
            ma_features +
            rsi_features
        )
        
        features.append(feature_vector)
        targets.append(current_price)
        dates.append(current_date)
    
    return np.array(features), np.array(targets), np.array(dates)

def train_ultra_accurate_xgboost(X_train, y_train, X_test, y_test, crop, mandi):
    """Train ultra-accurate XGBoost with extensive hyperparameter tuning"""
    
    # Advanced XGBoost parameters for high accuracy
    param_grid = {
        'n_estimators': [1000, 1500, 2000],
        'max_depth': [8, 10, 12, 15],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2]
    }
    
    # Use TimeSeriesSplit for time series data
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Grid search with cross-validation
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
    
    print(f"Starting XGBoost hyperparameter tuning for {crop} - {mandi}...")
    
    grid_search = GridSearchCV(
        xgb_model, param_grid, cv=tscv, scoring='neg_mean_squared_error',
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Train final model with best parameters
    final_model = xgb.XGBRegressor(**grid_search.best_params_, random_state=42)
    final_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = final_model.predict(X_test)
    
    # Calculate accuracy (100 - MAPE)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    accuracy = 100 - mape
    
    print(f"XGBoost {crop} {mandi} - Best Parameters: {grid_search.best_params_}")
    print(f"XGBoost {crop} {mandi} - Accuracy: {accuracy:.2f}%")
    
    # Save model
    model_path = f"app/data/processed/xgb_ultra_{crop}_{mandi}.joblib"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(final_model, model_path)
    
    return final_model, accuracy

def train_ultra_accurate_lstm(X_train, y_train, X_test, y_test, crop, mandi):
    """Train ultra-accurate LSTM with advanced architecture"""
    
    # Normalize features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Reshape for LSTM (samples, timesteps, features)
    X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
    X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
    
    # Advanced LSTM architecture for high accuracy
    model = Sequential([
        # First LSTM layer with more units
        Bidirectional(LSTM(128, return_sequences=True, 
                          kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                          recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-4))),
        BatchNormalization(),
        Dropout(0.3),
        
        # Second LSTM layer
        Bidirectional(LSTM(64, return_sequences=True,
                          kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                          recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-4))),
        BatchNormalization(),
        Dropout(0.3),
        
        # Third LSTM layer
        Bidirectional(LSTM(32, return_sequences=False,
                          kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                          recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-4))),
        BatchNormalization(),
        Dropout(0.3),
        
        # Dense layers
        Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    # Compile with advanced optimizer
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    # Advanced callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7),
        ModelCheckpoint(f'lstm_ultra_{crop}_{mandi}_best.h5', 
                       monitor='val_loss', save_best_only=True)
    ]
    
    print(f"Training LSTM for {crop} - {mandi}...")
    
    # Train with more epochs and validation split
    history = model.fit(
        X_train_lstm, y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Make predictions
    y_pred = model.predict(X_test_lstm).flatten()
    
    # Calculate accuracy (100 - MAPE)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    accuracy = 100 - mape
    
    print(f"LSTM {crop} {mandi} - Accuracy: {accuracy:.2f}%")
    
    # Save model and scaler
    model_path = f"app/data/processed/lstm_ultra_{crop}_{mandi}.h5"
    scaler_path = f"app/data/processed/lstm_ultra_scaler_{crop}_{mandi}.joblib"
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    
    return model, scaler, accuracy

def boost_accuracy_to_85_percent():
    """Main function to boost accuracy to 85%+"""
    
    mandis = {
        'arecanut': ['Sirsi', 'Yellapur', 'Siddapur', 'Shimoga', 'Sagar', 'Kumta'],
        'coconut': ['Bangalore', 'Arasikere', 'Channarayapatna', 'Ramanagara', 'Sira', 'Tumkur']
    }
    
    results = []
    
    print("🚀 Starting Ultra-Accurate Model Training (Target: 85%+ Accuracy)")
    print("This will take some time due to extensive hyperparameter tuning...")
    
    for crop in mandis.keys():
        for mandi in mandis[crop]:
            print(f"\n{'='*50}")
            print(f"Training Ultra-Accurate Models for {crop} - {mandi}")
            print(f"{'='*50}")
            
            # Load data
            data_path = f"app/data/processed/{crop}_{mandi}_for_training.csv"
            if not os.path.exists(data_path):
                print(f"Data not found: {data_path}")
                continue
            
            df = pd.read_csv(data_path)
            df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'])
            
            # Create ultra-advanced features
            X, y, dates = create_ultra_advanced_features(df, 'Modal_Price', 'Arrival_Date')
            
            # Split data (80-20)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
            print(f"Features: {X_train.shape[1]} (ultra-advanced features)")
            
            # Train XGBoost
            try:
                xgb_model, xgb_accuracy = train_ultra_accurate_xgboost(
                    X_train, y_train, X_test, y_test, crop, mandi
                )
            except Exception as e:
                print(f"XGBoost training failed: {e}")
                xgb_accuracy = 0
            
            # Train LSTM
            try:
                lstm_model, lstm_scaler, lstm_accuracy = train_ultra_accurate_lstm(
                    X_train, y_train, X_test, y_test, crop, mandi
                )
            except Exception as e:
                print(f"LSTM training failed: {e}")
                lstm_accuracy = 0
            
            # Store results
            results.append({
                'crop': crop,
                'mandi': mandi,
                'xgb_accuracy': xgb_accuracy,
                'lstm_accuracy': lstm_accuracy,
                'avg_accuracy': (xgb_accuracy + lstm_accuracy) / 2
            })
            
            print(f"\nResults for {crop} - {mandi}:")
            print(f"XGBoost Accuracy: {xgb_accuracy:.2f}%")
            print(f"LSTM Accuracy: {lstm_accuracy:.2f}%")
            print(f"Average Accuracy: {(xgb_accuracy + lstm_accuracy) / 2:.2f}%")
            
            # Check if target achieved
            if xgb_accuracy >= 85 and lstm_accuracy >= 85:
                print("🎉 TARGET ACHIEVED: Both models above 85%!")
            elif xgb_accuracy >= 85:
                print("🎉 XGBoost target achieved!")
            elif lstm_accuracy >= 85:
                print("🎉 LSTM target achieved!")
            else:
                print("⚠️ Target not yet achieved. Consider additional optimizations.")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('ultra_accurate_model_results.csv', index=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print("ULTRA-ACCURATE MODEL TRAINING SUMMARY")
    print(f"{'='*60}")
    
    for model_type in ['xgb_accuracy', 'lstm_accuracy']:
        avg_acc = results_df[model_type].mean()
        max_acc = results_df[model_type].max()
        min_acc = results_df[model_type].min()
        
        model_name = "XGBoost" if "xgb" in model_type else "LSTM"
        print(f"\n{model_name}:")
        print(f"  Average Accuracy: {avg_acc:.2f}%")
        print(f"  Maximum Accuracy: {max_acc:.2f}%")
        print(f"  Minimum Accuracy: {min_acc:.2f}%")
        print(f"  Models above 85%: {(results_df[model_type] > 85).sum()}/{len(results_df)}")
    
    return results_df

if __name__ == "__main__":
    results = boost_accuracy_to_85_percent()
    
    print(f"\n✅ Training Complete!")
    print(f"Results saved to: ultra_accurate_model_results.csv")
    print(f"Ultra-accurate models saved in: app/data/processed/")
