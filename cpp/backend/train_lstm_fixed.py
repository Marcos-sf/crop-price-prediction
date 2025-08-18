import numpy as np
import os
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd

mandis = {
    'arecanut': ['Sirsi', 'Yellapur', 'Siddapur', 'Shimoga', 'Sagar', 'Kumta'],
    'coconut': ['Bangalore', 'Arasikere', 'Channarayapatna', 'Ramanagara', 'Sira', 'Tumkur']
}

CUTOFF_DATE = np.datetime64('2023-01-01')

def train_lstm_model_fixed(X_train, y_train, X_test, y_test, crop, mandi):
    """Train LSTM model with proper target handling"""
    # Normalize features
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std
    
    # Normalize targets as well for better LSTM training
    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8
    y_train_norm = (y_train - y_mean) / y_std
    y_test_norm = (y_test - y_mean) / y_std
    
    # Reshape for LSTM: (samples, timesteps, features)
    X_train_lstm = X_train_norm.reshape((X_train_norm.shape[0], 1, X_train_norm.shape[1]))
    X_test_lstm = X_test_norm.reshape((X_test_norm.shape[0], 1, X_test_norm.shape[1]))
    
    # Build improved LSTM model
    model = Sequential([
        LSTM(64, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Early stopping to prevent overfitting
    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    # Train model on normalized targets
    model.fit(X_train_lstm, y_train_norm, epochs=200, batch_size=16, 
              validation_split=0.2, callbacks=[es], verbose=0)
    
    # Get predictions (normalized)
    y_pred_norm = model.predict(X_test_lstm).flatten()
    
    # Denormalize predictions to get actual prices
    y_pred_lstm = y_pred_norm * y_std + y_mean
    
    # Save model and normalization params (both feature and target)
    model_path = f"app/data/processed/lstm_{crop}_{mandi.lower()}.h5"
    norm_path = f"app/data/processed/lstm_norm_{crop}_{mandi.lower()}.joblib"
    model.save(model_path)
    
    # Save both feature and target normalization parameters
    norm_params = {
        'feature_mean': X_mean,
        'feature_std': X_std,
        'target_mean': y_mean,
        'target_std': y_std
    }
    joblib.dump(norm_params, norm_path)
    
    return y_pred_lstm, model_path, norm_path

# Main training loop
for crop, mandi_list in mandis.items():
    for mandi in mandi_list:
        print(f"\n{'='*60}")
        print(f"Training Fixed LSTM Model for {crop.title()} in {mandi.title()}")
        print(f"{'='*60}")
        
        # Load data
        features_path = f"app/data/processed/features_{crop}_{mandi.lower()}.npy"
        target_path = f"app/data/processed/target_{crop}_{mandi.lower()}.npy"
        dates_path = f"app/data/processed/dates_{crop}_{mandi.lower()}.npy"
        
        if not (os.path.exists(features_path) and os.path.exists(target_path) and os.path.exists(dates_path)):
            print(f"Missing data for {crop.title()} in {mandi}, skipping.")
            continue
        
        X = np.load(features_path)
        y = np.load(target_path)
        dates = np.load(dates_path)
        
        if len(X) < 10 or len(X) != len(y) or len(X) != len(dates):
            print(f"Data length mismatch for {crop.title()} in {mandi}, skipping.")
            continue
        
        # Out-of-sample split: train before cutoff, test from cutoff onwards
        mask_train = dates < CUTOFF_DATE
        mask_test = dates >= CUTOFF_DATE
        
        if mask_test.sum() < 5:
            print(f"Not enough test samples after {str(CUTOFF_DATE)} for {crop.title()} in {mandi}, skipping.")
            continue
        
        X_train, X_test = X[mask_train], X[mask_test]
        y_train, y_test = y[mask_train], y[mask_test]
        test_dates = dates[mask_test]
        
        # Check for sufficient training data
        if len(X_train) < 10:
            print(f"Not enough training samples before {str(CUTOFF_DATE)} for {crop.title()} in {mandi}, skipping.")
            continue
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(y_test)}")
        print(f"Test set date range: {str(test_dates.min())[:10]} to {str(test_dates.max())[:10]}")
        
        # Train LSTM model
        print("Training LSTM model...")
        y_pred_lstm, model_path, norm_path = train_lstm_model_fixed(X_train, y_train, X_test, y_test, crop, mandi)
        
        # Evaluate LSTM model
        rmse = mean_squared_error(y_test, y_pred_lstm, squared=False)
        mae = mean_absolute_error(y_test, y_pred_lstm)
        r2 = r2_score(y_test, y_pred_lstm)
        mape = np.mean(np.abs((y_test - y_pred_lstm) / y_test)) * 100
        accuracy = 100 - mape
        
        print(f"LSTM Performance:")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  R²: {r2:.3f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  Accuracy: {accuracy:.2f}%")
        
        print(f"LSTM model saved to {model_path}")
        print(f"Normalization parameters saved to {norm_path}")
        
        # Show sample predictions
        print(f"\nSample Predictions (first 5 test samples):")
        print(f"Actual:    {y_test[:5]}")
        print(f"LSTM:      {y_pred_lstm[:5]}")

print("\n" + "="*60)
print("FIXED LSTM TRAINING COMPLETED!")
print("="*60)
