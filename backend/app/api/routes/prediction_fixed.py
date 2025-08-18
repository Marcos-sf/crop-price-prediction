from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import joblib
import numpy as np
import os
from datetime import datetime, timedelta
import pandas as pd
from tensorflow.keras.models import load_model

router = APIRouter()

class PredictionRequest(BaseModel):
    crop: str
    mandi: str
    date: str  # YYYY-MM-DD
    modal_price_lag1: float
    modal_price_lag7: float
    model_type: str = "ensemble"  # "xgboost", "lstm", or "ensemble"

class ForecastRequest(BaseModel):
    crop: str
    mandi: str
    start_date: str  # YYYY-MM-DD
    modal_price_lag1: float
    modal_price_lag2: float
    modal_price_lag3: float
    modal_price_lag5: float
    modal_price_lag7: float
    rolling_mean_7: float
    rolling_std_7: float
    day_of_year: int
    month: int
    month_sin: float
    month_cos: float
    months: int = 12
    model_type: str = "ensemble"  # "xgboost", "lstm", or "ensemble"

class PredictionResponse(BaseModel):
    predicted_price: float
    model_type: str

class ForecastResponse(BaseModel):
    forecast: list
    model_type: str

class LagPricesResponse(BaseModel):
    modal_price_lag1: float
    modal_price_lag7: float
    latest_date: str
    lag7_date: str

class LatestFeaturesResponse(BaseModel):
    modal_price_lag1: float
    modal_price_lag2: float
    modal_price_lag3: float
    modal_price_lag5: float
    modal_price_lag7: float
    rolling_mean_7: float
    rolling_std_7: float
    day_of_year: int
    month: int
    month_sin: float
    month_cos: float
    latest_date: str
    lag1_date: str
    lag2_date: str
    lag3_date: str
    lag5_date: str
    lag7_date: str

def find_column(df, possible_names):
    """Find column by matching possible names (case insensitive, with/without underscores)"""
    for col in df.columns:
        col_clean = col.strip().lower().replace('_', ' ')
        for name in possible_names:
            if col_clean == name.lower():
                return col
    return None

def load_fast_models(crop: str, mandi: str):
    """Load fast high-accuracy models (>85% accuracy)"""
    # Check if fast models exist
    xgb_path = f"app/data/processed/xgb_fast_{crop}_{mandi}.joblib"
    lstm_path = f"app/data/processed/lstm_fast_{crop}_{mandi}.h5"
    lstm_scaler_path = f"app/data/processed/lstm_fast_scaler_{crop}_{mandi}.joblib"
    lstm_target_scaler_path = f"app/data/processed/lstm_fast_target_scaler_{crop}_{mandi}.joblib"
    
    if not all(os.path.exists(path) for path in [xgb_path, lstm_path, lstm_scaler_path, lstm_target_scaler_path]):
        return None, None, None, None, None
    
    # Load models
    xgb_model = joblib.load(xgb_path)
    lstm_model = load_model(lstm_path)
    lstm_scaler = joblib.load(lstm_scaler_path)
    lstm_target_scaler = joblib.load(lstm_target_scaler_path)
    
    # Default ensemble weights (can be adjusted based on individual model performance)
    weights = {'xgb': 0.5, 'lstm': 0.5}
    
    return xgb_model, lstm_model, lstm_scaler, lstm_target_scaler, weights

def load_ensemble_models(crop: str, mandi: str):
    """Load ensemble models and metadata (backward compatibility)"""
    # First try to load fast models
    fast_models = load_fast_models(crop, mandi)
    if fast_models[0] is not None:
        return fast_models
    
    # Fallback to old ensemble format
    ensemble_path = f"app/data/processed/ensemble_{crop}_{mandi}.joblib"
    if not os.path.exists(ensemble_path):
        return None, None, None, None, None
    
    ensemble_metadata = joblib.load(ensemble_path)
    
    # Load LSTM model
    lstm_model = load_model(ensemble_metadata['lstm_model_path'])
    lstm_norm = joblib.load(ensemble_metadata['lstm_norm_path'])
    
    # Load XGBoost model
    xgb_model = joblib.load(ensemble_metadata['xgb_model_path'])
    
    return xgb_model, lstm_model, lstm_norm, None, ensemble_metadata['weights']

def predict_with_lstm(features: np.ndarray, lstm_model, lstm_scaler, lstm_target_scaler=None):
    """Make prediction using LSTM model with new fast model format"""
    if lstm_target_scaler is not None:
        # New fast model format
        X_scaled = lstm_scaler.transform(features)
        X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        pred_scaled = lstm_model.predict(X_lstm, verbose=0)
        pred = lstm_target_scaler.inverse_transform(pred_scaled).flatten()
    else:
        # Old format (backward compatibility)
        if 'feature_mean' in lstm_scaler and 'target_mean' in lstm_scaler:
            X_norm = (features - lstm_scaler['feature_mean']) / lstm_scaler['feature_std']
            X_lstm = X_norm.reshape((X_norm.shape[0], 1, X_norm.shape[1]))
            pred_norm = lstm_model.predict(X_lstm, verbose=0).flatten()
            pred = pred_norm * lstm_scaler['target_std'] + lstm_scaler['target_mean']
        else:
            X_norm = (features - lstm_scaler['mean']) / lstm_scaler['std']
            X_lstm = X_norm.reshape((X_norm.shape[0], 1, X_norm.shape[1]))
            pred_norm = lstm_model.predict(X_lstm, verbose=0).flatten()
            if np.max(np.abs(pred_norm)) < 10:
                pred = pred_norm * 20000 + 40000
            else:
                pred = pred_norm
    return pred

def predict_with_xgboost(features: np.ndarray, xgb_model):
    """Make prediction using XGBoost model"""
    return xgb_model.predict(features)

def predict_with_ensemble(features: np.ndarray, xgb_model, lstm_model, lstm_scaler, lstm_target_scaler, weights):
    """Make ensemble prediction"""
    lstm_pred = predict_with_lstm(features, lstm_model, lstm_scaler, lstm_target_scaler)
    xgb_pred = predict_with_xgboost(features, xgb_model)
    
    # Weighted average
    if isinstance(weights, dict):
        ensemble_pred = weights['xgb'] * xgb_pred + weights['lstm'] * lstm_pred
    else:
        ensemble_pred = weights[0] * lstm_pred + weights[1] * xgb_pred
    return ensemble_pred

@router.get("/latest-prices", response_model=LagPricesResponse)
def get_latest_prices(
    crop: str = Query(..., description="Crop name"),
    mandi: str = Query(..., description="Mandi name")
):
    crop = crop.lower()
    mandi = mandi.lower()
    file_path = f"app/data/processed/{crop}_{mandi}_for_training.csv"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Data not found for {crop} in {mandi}.")
    
    df = pd.read_csv(file_path)
    
    # Find date column
    date_col = find_column(df, ['date', 'arrival_date', 'price_date', 'Arrival_Date'])
    if not date_col:
        raise HTTPException(status_code=500, detail="No date column found in data.")

    # Find modal price column
    modal_price_col = find_column(df, ['modal_price', 'Modal_Price', 'modal price (rs./quintal)', 'modal price'])
    if not modal_price_col:
        raise HTTPException(status_code=500, detail="No modal price column found in data.")
    
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)
    n = len(df)
    
    if n < 2:
        raise HTTPException(status_code=400, detail="Not enough data to compute lag prices.")
    
    latest_price = float(df[modal_price_col].iloc[-1])
    lag7_price = float(df[modal_price_col].iloc[-2]) if n >= 2 else latest_price
    latest_date = df[date_col].iloc[-1].strftime("%Y-%m-%d")
    lag7_date = df[date_col].iloc[-2].strftime("%Y-%m-%d") if n >= 2 else latest_date
    
    return LagPricesResponse(
        modal_price_lag1=latest_price,
        modal_price_lag7=lag7_price,
        latest_date=latest_date,
        lag7_date=lag7_date
    )

@router.get("/latest-features", response_model=LatestFeaturesResponse)
def get_latest_features(
    crop: str = Query(..., description="Crop name"),
    mandi: str = Query(..., description="Mandi name")
):
    crop = crop.lower()
    mandi = mandi.lower()
    file_path = f"app/data/processed/{crop}_{mandi}_for_training.csv"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Data not found for {crop} in {mandi}.")
    
    df = pd.read_csv(file_path)
    
    # Find date column
    date_col = find_column(df, ['date', 'arrival_date', 'price_date', 'Arrival_Date'])
    if not date_col:
        raise HTTPException(status_code=500, detail="No date column found in data.")

    # Find modal price column
    modal_price_col = find_column(df, ['modal_price', 'Modal_Price', 'modal price (rs./quintal)', 'modal price'])
    if not modal_price_col:
        raise HTTPException(status_code=500, detail="No modal price column found in data.")
    
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)
    n = len(df)
    
    if n < 8:
        raise HTTPException(status_code=400, detail="Not enough data to compute all features.")
    
    # Use last N available market day prices for lags/rolling (only the 11 features we need)
    latest_idx = df.index[-1]
    lag1_idx = df.index[-2]
    lag2_idx = df.index[-3]
    lag3_idx = df.index[-4]
    lag5_idx = df.index[-6]
    lag7_idx = df.index[-8]
    
    modal_price_lag1 = float(df.loc[lag1_idx, modal_price_col])
    modal_price_lag2 = float(df.loc[lag2_idx, modal_price_col])
    modal_price_lag3 = float(df.loc[lag3_idx, modal_price_col])
    modal_price_lag5 = float(df.loc[lag5_idx, modal_price_col])
    modal_price_lag7 = float(df.loc[lag7_idx, modal_price_col])
    
    # Rolling statistics (7-day window)
    recent_prices = df[modal_price_col].iloc[-7:].values
    rolling_mean_7 = float(np.mean(recent_prices))
    rolling_std_7 = float(np.std(recent_prices))
    
    # Temporal features
    latest_date = df.loc[latest_idx, date_col]
    day_of_year = latest_date.timetuple().tm_yday
    month = latest_date.month
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    # Date strings for reference
    latest_date_str = latest_date.strftime("%Y-%m-%d")
    lag1_date_str = df.loc[lag1_idx, date_col].strftime("%Y-%m-%d")
    lag2_date_str = df.loc[lag2_idx, date_col].strftime("%Y-%m-%d")
    lag3_date_str = df.loc[lag3_idx, date_col].strftime("%Y-%m-%d")
    lag5_date_str = df.loc[lag5_idx, date_col].strftime("%Y-%m-%d")
    lag7_date_str = df.loc[lag7_idx, date_col].strftime("%Y-%m-%d")
    
    return LatestFeaturesResponse(
        modal_price_lag1=modal_price_lag1,
        modal_price_lag2=modal_price_lag2,
        modal_price_lag3=modal_price_lag3,
        modal_price_lag5=modal_price_lag5,
        modal_price_lag7=modal_price_lag7,
        rolling_mean_7=rolling_mean_7,
        rolling_std_7=rolling_std_7,
        day_of_year=day_of_year,
        month=month,
        month_sin=float(month_sin),
        month_cos=float(month_cos),
        latest_date=latest_date_str,
        lag1_date=lag1_date_str,
        lag2_date=lag2_date_str,
        lag3_date=lag3_date_str,
        lag5_date=lag5_date_str,
        lag7_date=lag7_date_str
    )

@router.post("/predict", response_model=PredictionResponse)
def predict_price(request: PredictionRequest):
    crop = request.crop.lower()
    mandi = request.mandi.lower()
    model_type = request.model_type.lower()
    
    try:
        date_obj = datetime.strptime(request.date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    
    # Prepare features (using only the features available in the request)
    features = np.array([[request.modal_price_lag1, request.modal_price_lag7]])
    
    if model_type == "ensemble":
        # Try to load fast models first
        xgb_model, lstm_model, lstm_scaler, lstm_target_scaler, weights = load_fast_models(crop, mandi)
        if xgb_model is not None:
            pred = float(predict_with_ensemble(features, xgb_model, lstm_model, lstm_scaler, lstm_target_scaler, weights)[0])
        else:
            # Fallback to old models
            lstm_model, xgb_model, lstm_norm, weights = load_ensemble_models(crop, mandi)
            if lstm_model is None:
                model_path = f"app/data/processed/xgb_{crop}_{mandi}.joblib"
                if not os.path.exists(model_path):
                    raise HTTPException(status_code=404, detail=f"Model not found for {crop} in {mandi}.")
                model = joblib.load(model_path)
                pred = float(model.predict(features)[0])
                model_type = "xgboost"
            else:
                pred = float(predict_with_ensemble(features, lstm_model, xgb_model, lstm_norm, weights)[0])
    
    elif model_type == "lstm":
        # Try fast LSTM first
        _, lstm_model, lstm_scaler, lstm_target_scaler, _ = load_fast_models(crop, mandi)
        if lstm_model is not None:
            pred = float(predict_with_lstm(features, lstm_model, lstm_scaler, lstm_target_scaler)[0])
        else:
            # Fallback to old LSTM
            model_path = f"app/data/processed/lstm_{crop}_{mandi}.h5"
            norm_path = f"app/data/processed/lstm_norm_{crop}_{mandi}.joblib"
            if not os.path.exists(model_path) or not os.path.exists(norm_path):
                raise HTTPException(status_code=404, detail=f"LSTM model not found for {crop} in {mandi}.")
            
            lstm_model = load_model(model_path)
            lstm_norm = joblib.load(norm_path)
            pred = float(predict_with_lstm(features, lstm_model, lstm_norm)[0])
    
    elif model_type == "xgboost":
        # Try fast XGBoost first
        xgb_model, _, _, _, _ = load_fast_models(crop, mandi)
        if xgb_model is not None:
            pred = float(predict_with_xgboost(features, xgb_model)[0])
        else:
            # Fallback to old XGBoost
            model_path = f"app/data/processed/xgb_{crop}_{mandi}.joblib"
            if not os.path.exists(model_path):
                raise HTTPException(status_code=404, detail=f"XGBoost model not found for {crop} in {mandi}.")
            
            model = joblib.load(model_path)
            pred = float(model.predict(features)[0])
    
    else:
        raise HTTPException(status_code=400, detail="Invalid model_type. Use 'ensemble', 'lstm', or 'xgboost'.")
    
    return PredictionResponse(predicted_price=pred, model_type=model_type)

@router.get("/prediction-status")
def get_prediction_status():
    """Get status of available models"""
    # Valid crop-mandi combinations
    combinations = [
        ("arecanut", "sirsi"), ("arecanut", "yellapur"), ("arecanut", "siddapur"),
        ("arecanut", "shimoga"), ("arecanut", "sagar"), ("arecanut", "kumta"),
        ("coconut", "bangalore"), ("coconut", "arasikere"), ("coconut", "channarayapatna"),
        ("coconut", "ramanagara"), ("coconut", "sira"), ("coconut", "tumkur")
    ]
    
    status = {
        "total_combinations": len(combinations),
        "available_models": [],
        "fast_models_available": 0,
        "old_models_available": 0
    }
    
    for crop, mandi in combinations:
        # Check if fast models exist
        fast_models = load_fast_models(crop, mandi)
        has_fast = fast_models[0] is not None
        
        # Check if old models exist
        old_xgb = os.path.exists(f"app/data/processed/xgb_{crop}_{mandi}.joblib")
        old_lstm = os.path.exists(f"app/data/processed/lstm_{crop}_{mandi}.h5")
        
        model_info = {
            "crop": crop,
            "mandi": mandi,
            "fast_models": has_fast,
            "old_xgboost": old_xgb,
            "old_lstm": old_lstm,
            "data_available": os.path.exists(f"app/data/processed/{crop}_{mandi}_for_training.csv")
        }
        
        status["available_models"].append(model_info)
        
        if has_fast:
            status["fast_models_available"] += 1
        if old_xgb or old_lstm:
            status["old_models_available"] += 1
    
    return status
