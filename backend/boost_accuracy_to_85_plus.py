#!/usr/bin/env python3
"""
Boost Model Accuracy to 85%+ 
This script enhances XGBoost, LSTM, and Ensemble models to achieve >85% accuracy.
"""

import os
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

def create_enhanced_features(features, targets, dates):
    """Create enhanced feature set for better accuracy"""
    enhanced_features = []
    
    for i in range(len(features)):
        if i < 30:  # Need at least 30 days of history
            # Pad with zeros for early days
            padded_features = np.zeros(30)
            padded_features[-i-1:] = features[max(0, i-29):i+1]
            enhanced_features.append(padded_features)
        else:
            # Use last 30 days
            recent_features = features[i-29:i+1]
            enhanced_features.append(recent_features)
    
    enhanced_features = np.array(enhanced_features)
    
    # Add additional engineered features
    additional_features = []
    for i in range(len(enhanced_features)):
        if i < 30:
            # Early days - use basic features
            # Price momentum (if available)
            if len(targets) > i and i > 0:
                momentum = targets[i] - targets[i-1] if i > 0 else 0
            else:
                momentum = 0
            # Volatility (if available)
            if len(targets) > i and i > 4:
                volatility = np.std(targets[max(0, i-4):i+1])
            else:
                volatility = 0
            # Trend (if available)
            if len(targets) > i and i > 6:
                x_trend = np.arange(7)
                y_trend = targets[max(0, i-6):i+1]
                if len(y_trend) == 7:
                    trend_slope = np.polyfit(x_trend, y_trend, 1)[0]
                else:
                    trend_slope = 0
            else:
                trend_slope = 0
            # Seasonal features
            seasonal_day = np.sin(2 * np.pi * (i % 365) / 365)
            seasonal_week = np.sin(2 * np.pi * (i % 7) / 7)
            
            # Add these features
            additional_features.append([momentum, volatility, trend_slope, seasonal_day, seasonal_week])
        else:
            # Full history available
            # Price momentum
            momentum = targets[i] - targets[i-1] if i > 0 else 0
            # Volatility (7-day rolling)
            volatility = np.std(targets[max(0, i-6):i+1])
            # Trend (7-day linear regression slope)
            if i > 6:
                x_trend = np.arange(7)
                y_trend = targets[i-6:i+1]
                trend_slope = np.polyfit(x_trend, y_trend, 1)[0]
            else:
                trend_slope = 0
            # Seasonal features
            seasonal_day = np.sin(2 * np.pi * (i % 365) / 365)
            seasonal_week = np.sin(2 * np.pi * (i % 7) / 7)
            
            # Add these features
            additional_features.append([momentum, volatility, trend_slope, seasonal_day, seasonal_week])
    
    # Pad additional features for early days
    while len(additional_features) < len(enhanced_features):
        additional_features.insert(0, [0, 0, 0, 0, 0])
    
    additional_features = np.array(additional_features)
    
    # Combine features
    final_features = np.column_stack([enhanced_features, additional_features])
    
    return final_features

def train_enhanced_xgboost(X_train, y_train, X_val, y_val):
    """Train enhanced XGBoost with hyperparameter tuning"""
    print("🚀 Training Enhanced XGBoost...")
    
    # Define parameter grid for tuning
    param_grid = {
        'n_estimators': [200, 300, 400],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5]
    }
    
    # Use TimeSeriesSplit for time series data
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Initialize XGBoost
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_xgb = grid_search.best_estimator_
    
    # Evaluate on validation set
    y_pred = best_xgb.predict(X_val)
    val_r2 = r2_score(y_val, y_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    # Calculate accuracy within 10%
    accuracy_10 = np.mean(np.abs((y_pred - y_val) / y_val) <= 0.1) * 100
    
    print(f"✅ Enhanced XGBoost - R²: {val_r2:.4f}, RMSE: {val_rmse:.2f}, Accuracy (10%): {accuracy_10:.2f}%")
    print(f"   Best parameters: {grid_search.best_params_}")
    
    return best_xgb, val_r2, accuracy_10

def train_enhanced_lstm(X_train, y_train, X_val, y_val):
    """Train enhanced LSTM with better architecture"""
    print("🧠 Training Enhanced LSTM...")
    
    # Reshape data for LSTM (samples, timesteps, features)
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_val_lstm = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
    
    # Build enhanced LSTM model
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(1, X_train.shape[1])),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    # Compile with optimized settings
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    # Callbacks for better training
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        X_train_lstm, y_train,
        validation_data=(X_val_lstm, y_val),
        epochs=200,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate on validation set
    y_pred = model.predict(X_val_lstm).flatten()
    val_r2 = r2_score(y_val, y_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    # Calculate accuracy within 10%
    accuracy_10 = np.mean(np.abs((y_pred - y_val) / y_val) <= 0.1) * 100
    
    print(f"✅ Enhanced LSTM - R²: {val_r2:.4f}, RMSE: {val_rmse:.2f}, Accuracy (10%): {accuracy_10:.2f}%")
    
    return model, val_r2, accuracy_10

def create_ensemble_model(xgb_model, lstm_model, X_train, y_train, X_val, y_val):
    """Create optimized ensemble model"""
    print("🔗 Creating Optimized Ensemble...")
    
    # Get predictions from both models
    xgb_pred_train = xgb_model.predict(X_train)
    lstm_pred_train = lstm_model.predict(X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))).flatten()
    
    xgb_pred_val = xgb_model.predict(X_val)
    lstm_pred_val = lstm_model.predict(X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))).flatten()
    
    # Optimize ensemble weights using validation set
    best_weights = None
    best_score = float('-inf')
    
    # Grid search for optimal weights
    for w1 in np.arange(0.1, 1.0, 0.1):
        w2 = 1.0 - w1
        ensemble_pred_val = w1 * xgb_pred_val + w2 * lstm_pred_val
        score = r2_score(y_val, ensemble_pred_val)
        
        if score > best_score:
            best_score = score
            best_weights = (w1, w2)
    
    print(f"   Optimal weights - XGBoost: {best_weights[0]:.2f}, LSTM: {best_weights[1]:.2f}")
    
    # Create final ensemble predictions
    ensemble_pred_val = best_weights[0] * xgb_pred_val + best_weights[1] * lstm_pred_val
    
    # Calculate ensemble metrics
    ensemble_r2 = r2_score(y_val, ensemble_pred_val)
    ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred_val))
    ensemble_accuracy_10 = np.mean(np.abs((ensemble_pred_val - y_val) / y_val) <= 0.1) * 100
    
    print(f"✅ Ensemble Model - R²: {ensemble_r2:.4f}, RMSE: {ensemble_rmse:.2f}, Accuracy (10%): {ensemble_accuracy_10:.2f}%")
    
    return best_weights, ensemble_r2, ensemble_accuracy_10

def boost_model_accuracy(crop, mandi):
    """Boost accuracy for a specific crop-mandi combination"""
    print(f"\n🌾 Boosting accuracy for {crop.title()} - {mandi.title()}")
    print("=" * 60)
    
    try:
        # Load existing data
        features_path = f"app/data/processed/features_{crop}_{mandi}.npy"
        target_path = f"app/data/processed/target_{crop}_{mandi}.npy"
        dates_path = f"app/data/processed/dates_{crop}_{mandi}.npy"
        
        if not all(os.path.exists(path) for path in [features_path, target_path, dates_path]):
            print(f"❌ Data not found for {crop}_{mandi}")
            return None
        
        # Load data
        features = np.load(features_path)
        targets = np.load(target_path)
        dates = np.load(dates_path)
        
        print(f"📊 Data loaded: {len(features)} samples, {features.shape[1]} features")
        
        # Create enhanced features
        enhanced_features = create_enhanced_features(features, targets, dates)
        print(f"🚀 Enhanced features created: {enhanced_features.shape}")
        
        # Split data (70% train, 15% validation, 15% test)
        train_split = int(len(enhanced_features) * 0.7)
        val_split = int(len(enhanced_features) * 0.85)
        
        X_train = enhanced_features[:train_split]
        y_train = targets[:train_split]
        X_val = enhanced_features[train_split:val_split]
        y_val = targets[train_split:val_split]
        X_test = enhanced_features[val_split:]
        y_test = targets[val_split:]
        
        print(f"📈 Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Train enhanced models
        xgb_model, xgb_r2, xgb_acc = train_enhanced_xgboost(X_train, y_train, X_val, y_val)
        lstm_model, lstm_r2, lstm_acc = train_enhanced_lstm(X_train, y_train, X_val, y_val)
        
        # Create ensemble
        weights, ensemble_r2, ensemble_acc = create_ensemble_model(
            xgb_model, lstm_model, X_train, y_train, X_val, y_val
        )
        
        # Final evaluation on test set
        print(f"\n📊 FINAL TEST SET EVALUATION")
        print("=" * 40)
        
        # XGBoost test performance
        xgb_test_pred = xgb_model.predict(X_test)
        xgb_test_r2 = r2_score(y_test, xgb_test_pred)
        xgb_test_acc = np.mean(np.abs((xgb_test_pred - y_test) / y_test) <= 0.1) * 100
        
        # LSTM test performance
        lstm_test_pred = lstm_model.predict(X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))).flatten()
        lstm_test_r2 = r2_score(y_test, lstm_test_pred)
        lstm_test_acc = np.mean(np.abs((lstm_test_pred - y_test) / y_test) <= 0.1) * 100
        
        # Ensemble test performance
        ensemble_test_pred = weights[0] * xgb_test_pred + weights[1] * lstm_test_pred
        ensemble_test_r2 = r2_score(y_test, ensemble_test_pred)
        ensemble_test_acc = np.mean(np.abs((ensemble_test_pred - y_test) / y_test) <= 0.1) * 100
        
        print(f"XGBoost Test - R²: {xgb_test_r2:.4f}, Accuracy (10%): {xgb_test_acc:.2f}%")
        print(f"LSTM Test    - R²: {lstm_test_r2:.4f}, Accuracy (10%): {lstm_test_acc:.2f}%")
        print(f"Ensemble Test- R²: {ensemble_test_r2:.4f}, Accuracy (10%): {ensemble_test_acc:.2f}%")
        
        # Save enhanced models
        print(f"\n💾 Saving enhanced models...")
        
        # Save XGBoost
        xgb_enhanced_path = f"app/data/processed/xgb_enhanced_{crop}_{mandi}.joblib"
        joblib.dump(xgb_model, xgb_enhanced_path)
        
        # Save LSTM
        lstm_enhanced_path = f"app/data/processed/lstm_enhanced_{crop}_{mandi}.h5"
        lstm_model.save(lstm_enhanced_path)
        
        # Save ensemble metadata
        ensemble_metadata = {
            'xgb_model_path': xgb_enhanced_path,
            'lstm_model_path': lstm_enhanced_path,
            'weights': weights,
            'xgb_r2': xgb_test_r2,
            'lstm_r2': lstm_test_r2,
            'ensemble_r2': ensemble_test_r2,
            'xgb_accuracy': xgb_test_acc,
            'lstm_accuracy': lstm_test_acc,
            'ensemble_accuracy': ensemble_test_acc
        }
        
        ensemble_path = f"app/data/processed/ensemble_enhanced_{crop}_{mandi}.joblib"
        joblib.dump(ensemble_metadata, ensemble_path)
        
        print(f"✅ Enhanced models saved successfully!")
        
        # Check if accuracy targets are met
        targets_met = []
        if xgb_test_acc >= 85:
            targets_met.append("✅ XGBoost ≥85%")
        else:
            targets_met.append(f"❌ XGBoost {xgb_test_acc:.1f}%")
            
        if lstm_test_acc >= 85:
            targets_met.append("✅ LSTM ≥85%")
        else:
            targets_met.append(f"❌ LSTM {lstm_test_acc:.1f}%")
            
        if ensemble_test_acc >= 85:
            targets_met.append("✅ Ensemble ≥85%")
        else:
            targets_met.append(f"❌ Ensemble {ensemble_test_acc:.1f}%")
        
        print(f"\n🎯 ACCURACY TARGETS:")
        for target in targets_met:
            print(f"   {target}")
        
        return {
            'crop': crop,
            'mandi': mandi,
            'xgb_accuracy': xgb_test_acc,
            'lstm_accuracy': lstm_test_acc,
            'ensemble_accuracy': ensemble_test_acc,
            'targets_met': len([t for t in targets_met if '✅' in t])
        }
        
    except Exception as e:
        print(f"❌ Error boosting {crop}_{mandi}: {e}")
        return None

def main():
    """Main function to boost all models"""
    print("🚀 BOOSTING MODEL ACCURACY TO 85%+")
    print("=" * 60)
    
    # Define crop-mandi combinations
    crop_mandi_combinations = [
        ('arecanut', 'sirsi'), ('arecanut', 'yellapur'), ('arecanut', 'siddapur'),
        ('arecanut', 'shimoga'), ('arecanut', 'sagar'), ('arecanut', 'kumta'),
        ('coconut', 'bangalore'), ('coconut', 'arasikere'), ('coconut', 'channarayapatna'),
        ('coconut', 'ramanagara'), ('coconut', 'sira'), ('coconut', 'tumkur')
    ]
    
    results = []
    
    for crop, mandi in crop_mandi_combinations:
        result = boost_model_accuracy(crop, mandi)
        if result:
            results.append(result)
    
    if not results:
        print("❌ No models could be boosted!")
        return
    
    # Summary report
    print(f"\n" + "=" * 60)
    print("📊 BOOSTING SUMMARY REPORT")
    print("=" * 60)
    
    df = pd.DataFrame(results)
    
    print(f"Total Models Processed: {len(df)}")
    print(f"Models with XGBoost ≥85%: {len(df[df['xgb_accuracy'] >= 85])}")
    print(f"Models with LSTM ≥85%: {len(df[df['lstm_accuracy'] >= 85])}")
    print(f"Models with Ensemble ≥85%: {len(df[df['ensemble_accuracy'] >= 85])}")
    print(f"Models with All Targets Met: {len(df[df['targets_met'] == 3])}")
    
    print(f"\nAverage Accuracies:")
    print(f"XGBoost: {df['xgb_accuracy'].mean():.2f}%")
    print(f"LSTM: {df['lstm_accuracy'].mean():.2f}%")
    print(f"Ensemble: {df['ensemble_accuracy'].mean():.2f}%")
    
    # Save summary
    summary_file = 'enhanced_models_summary.csv'
    df.to_csv(summary_file, index=False)
    print(f"\n💾 Summary saved to: {summary_file}")
    
    return df

if __name__ == "__main__":
    results_df = main()
