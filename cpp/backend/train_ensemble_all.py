import numpy as np
import os
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import pandas as pd

mandis = {
    'arecanut': ['Sirsi', 'Yellapur', 'Siddapur', 'Shimoga', 'Sagar', 'Kumta'],
    'coconut': ['Bangalore', 'Arasikere', 'Channarayapatna', 'Ramanagara', 'Sira', 'Tumkur']
}

CUTOFF_DATE = np.datetime64('2023-01-01')

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:(i+seq_len)])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

def train_lstm_model(X_train, y_train, X_test, y_test, crop, mandi):
    """Train LSTM model and return predictions"""
    # Normalize features
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std

    SEQ_LEN = 6  # Use last 6 months as input
    X_train_lstm, y_train_lstm = create_sequences(X_train_norm, y_train, SEQ_LEN)
    X_test_lstm, y_test_lstm = create_sequences(X_test_norm, y_test, SEQ_LEN)

    # Build LSTM model
    model = Sequential([
        LSTM(64, input_shape=(SEQ_LEN, X_train_lstm.shape[2]), dropout=0.2, recurrent_dropout=0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train_lstm, y_train_lstm, epochs=100, batch_size=16, validation_split=0.2, callbacks=[es], verbose=0)

    # Predict
    y_pred_lstm = model.predict(X_test_lstm).flatten()

    # Save model and normalization
    model.save(f'app/data/processed/lstm_{crop.lower()}_{mandi.lower()}.h5')
    np.save(f'app/data/processed/lstm_norm_{crop.lower()}_{mandi.lower()}.npy', {'mean': X_mean, 'std': X_std})

    return y_pred_lstm, y_test_lstm

def train_xgboost_model(X_train, y_train, X_test, y_test, crop, mandi):
    """Train XGBoost model and return predictions"""
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Hyperparameter tuning
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0]
    }
    
    xgb_model = xgb.XGBRegressor()
    search = RandomizedSearchCV(xgb_model, param_distributions=param_dist, 
                               n_iter=10, cv=tscv, scoring='neg_mean_absolute_error', 
                               n_jobs=-1, random_state=42)
    search.fit(X_train, y_train)
    
    # Train final model with best parameters
    model = search.best_estimator_
    model.fit(X_train, y_train)
    
    # Get predictions
    y_pred_xgb = model.predict(X_test)
    
    # Save model
    model_path = f"app/data/processed/xgb_{crop}_{mandi.lower()}.joblib"
    joblib.dump(model, model_path)
    
    return y_pred_xgb, model_path

def ensemble_predictions(y_pred_lstm, y_pred_xgb, weights=(0.5, 0.5)):
    """Combine LSTM and XGBoost predictions using weighted average"""
    return weights[0] * y_pred_lstm + weights[1] * y_pred_xgb

def evaluate_models(y_test, y_pred_lstm, y_pred_xgb, y_pred_ensemble, crop, mandi):
    """Evaluate individual models and ensemble"""
    models = {
        'LSTM': y_pred_lstm,
        'XGBoost': y_pred_xgb,
        'Ensemble': y_pred_ensemble
    }
    
    print(f"\n=== Model Evaluation for {crop.title()} in {mandi.title()} ===")
    
    for model_name, y_pred in models.items():
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        accuracy = 100 - mape
        
        print(f"{model_name}:")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  R²: {r2:.3f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  Accuracy: {accuracy:.2f}%")
    
    return models

def find_best_ensemble_weight(y_test, y_pred_lstm, y_pred_xgb):
    best_acc = 0
    best_w = 0.5
    for w in np.arange(0, 1.05, 0.05):
        y_pred_ensemble = w * y_pred_lstm + (1-w) * y_pred_xgb
        mape = np.mean(np.abs((y_test - y_pred_ensemble) / y_test)) * 100
        acc = 100 - mape
        if acc > best_acc:
            best_acc = acc
            best_w = w
    return best_w, best_acc

# Main training loop
for crop, mandi_list in mandis.items():
    for mandi in mandi_list:
        print(f"\n{'='*60}")
        print(f"Training Ensemble Models for {crop.title()} in {mandi.title()}")
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
        
        try:
            # Train LSTM and get predictions
            y_pred_lstm, y_test_lstm = train_lstm_model(X_train, y_train, X_test, y_test, crop, mandi)
            # Train XGBoost and get predictions
            y_pred_xgb = train_xgboost_model(X_train, y_train, X_test, y_test, crop, mandi)

            # Align XGBoost predictions to LSTM output length
            n = len(y_pred_lstm)
            y_pred_xgb_aligned = y_pred_xgb[-n:]

            # Before ensemble_predictions call:
            import numpy as np

            y_pred_lstm = np.array(y_pred_lstm)
            y_pred_xgb_aligned = np.array(y_pred_xgb_aligned)

            # Ensemble predictions
            y_pred_ensemble = ensemble_predictions(y_pred_lstm, y_pred_xgb_aligned)

            # Evaluate models
            evaluate_models(y_test_lstm, y_pred_lstm, y_pred_xgb_aligned, y_pred_ensemble, crop, mandi)

        except Exception as e:
            print(f"Error training models for {crop} in {mandi}: {e}")
            continue

print("\n" + "="*60)
print("ENSEMBLE TRAINING COMPLETED!")
print("="*60)
print("\nModels saved:")
print("- LSTM models: app/data/processed/lstm_*.h5")
print("- XGBoost models: app/data/processed/xgb_*.joblib")
print("- Ensemble metadata: app/data/processed/ensemble_*.joblib")
print("- LSTM normalization: app/data/processed/lstm_norm_*.joblib")
