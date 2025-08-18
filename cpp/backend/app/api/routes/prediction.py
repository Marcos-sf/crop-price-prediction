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
    days: int = 7  # Number of days to forecast
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
    """Find column by exact match first, then case insensitive"""
    # First try exact matches
    for name in possible_names:
        if name in df.columns:
            return name
    
    # Then try case insensitive matches
    for col in df.columns:
        col_lower = col.strip().lower()
        for name in possible_names:
            if col_lower == name.lower():
                return col
    return None

def load_simple_model(crop: str, mandi: str):
    """Load simple model that matches current API structure"""
    model_path = f"app/data/processed/simple_{crop}_{mandi}.joblib"
    scaler_path = f"app/data/processed/simple_scaler_{crop}_{mandi}.joblib"
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            return model, scaler
        except Exception as e:
            print(f"Error loading simple model: {e}")
    return None, None

def load_advanced_models(crop: str, mandi: str):
    """Load new advanced high-accuracy models (>85% accuracy)"""
    xgb_path = f"app/data/processed/xgb_advanced_{crop}_{mandi}.joblib"
    lstm_path = f"app/data/processed/lstm_advanced_{crop}_{mandi}.h5"
    lstm_scaler_path = f"app/data/processed/lstm_advanced_scaler_{crop}_{mandi}.joblib"
    lstm_target_scaler_path = f"app/data/processed/lstm_advanced_target_scaler_{crop}_{mandi}.joblib"
    weights_path = f"app/data/processed/ensemble_advanced_weights_{crop}_{mandi}.joblib"
    
    if all(os.path.exists(path) for path in [xgb_path, lstm_path, lstm_scaler_path, lstm_target_scaler_path, weights_path]):
        try:
            xgb_model = joblib.load(xgb_path)
            lstm_model = load_model(lstm_path)
            lstm_scaler = joblib.load(lstm_scaler_path)
            lstm_target_scaler = joblib.load(lstm_target_scaler_path)
            weights = joblib.load(weights_path)
            
            return xgb_model, lstm_model, lstm_scaler, lstm_target_scaler, weights
        except Exception as e:
            print(f"Error loading advanced models: {e}")
    return None, None, None, None, None

def load_fast_models(crop: str, mandi: str):
    """Load high-accuracy models with priority order: advanced > fast > simple"""
    # First priority: New advanced models (>85% accuracy)
    xgb_model, lstm_model, lstm_scaler, lstm_target_scaler, weights = load_advanced_models(crop, mandi)
    if xgb_model is not None:
        return xgb_model, lstm_model, lstm_scaler, lstm_target_scaler, weights
    
    # Second priority: Simple models for API compatibility
    simple_model, simple_scaler = load_simple_model(crop, mandi)
    if simple_model is not None:
        return simple_model, None, simple_scaler, None, {'simple': 1.0}
    
    # Third priority: Original fast models
    xgb_path = f"app/data/processed/xgb_fast_{crop}_{mandi}.joblib"
    lstm_path = f"app/data/processed/lstm_fast_{crop}_{mandi}.h5"
    lstm_scaler_path = f"app/data/processed/lstm_fast_scaler_{crop}_{mandi}.joblib"
    lstm_target_scaler_path = f"app/data/processed/lstm_fast_target_scaler_{crop}_{mandi}.joblib"
    
    if all(os.path.exists(path) for path in [xgb_path, lstm_path, lstm_scaler_path, lstm_target_scaler_path]):
        try:
            xgb_model = joblib.load(xgb_path)
            lstm_model = load_model(lstm_path)
            lstm_scaler = joblib.load(lstm_scaler_path)
            lstm_target_scaler = joblib.load(lstm_target_scaler_path)
            weights = {'xgb': 0.5, 'lstm': 0.5}
            
            return xgb_model, lstm_model, lstm_scaler, lstm_target_scaler, weights
        except Exception as e:
            print(f"Error loading fast models: {e}")
    
    return None, None, None, None, None

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

def create_advanced_features_from_basic(basic_features, crop, mandi, target_size=62):
    """Create advanced features from basic API inputs for advanced models"""
    # This is a simplified version - in production you'd need historical data
    # For now, we'll create approximate features based on the basic inputs
    
    # Extract basic features
    lag1, lag2, lag3, lag5, lag7 = basic_features[0:5]
    rolling_mean_7, rolling_std_7 = basic_features[5:7]
    day_of_year, month, month_sin, month_cos = basic_features[7:11]
    
    # Create extended feature set (approximation for advanced models)
    advanced_features = []
    
    # Original lag features
    advanced_features.extend([lag1, lag2, lag3, lag5, lag7])
    
    # Extended lag features (approximate from existing)
    advanced_features.extend([lag7 * 0.95, lag7 * 0.9, lag7 * 0.85])  # lag14, lag21, lag30 approximations
    
    # Rolling statistics (multiple windows)
    for window_factor in [0.8, 1.0, 1.2, 1.5, 2.0]:  # Different window approximations
        advanced_features.extend([
            rolling_mean_7 * window_factor,  # rolling_mean
            rolling_std_7 * window_factor,   # rolling_std
            lag1 * 0.95 * window_factor,     # rolling_min approximation
            lag1 * 1.05 * window_factor,     # rolling_max approximation
            rolling_mean_7 * window_factor   # rolling_median approximation
        ])
    
    # Price change features
    advanced_features.extend([
        (lag1 - lag2) / lag2 if lag2 != 0 else 0,  # price_change_1d
        (lag1 - lag7) / lag7 if lag7 != 0 else 0,  # price_change_7d
        (lag1 - lag7) / lag7 if lag7 != 0 else 0   # price_change_30d approximation
    ])
    
    # Volatility features
    volatility = rolling_std_7 / rolling_mean_7 if rolling_mean_7 != 0 else 0
    advanced_features.extend([volatility, volatility * 1.2])
    
    # Temporal features
    advanced_features.extend([
        2025,  # year (current year)
        month, day_of_year % 31 + 1, day_of_year, 
        (day_of_year - 1) // 7 + 1,  # week_of_year
        (month - 1) // 3 + 1,  # quarter
        0,  # is_weekend (assume weekday)
        month_sin, month_cos,
        np.sin(2 * np.pi * (day_of_year % 31 + 1) / 31),  # day_sin
        np.cos(2 * np.pi * (day_of_year % 31 + 1) / 31),  # day_cos
        np.sin(2 * np.pi * day_of_year / 365),  # dayofyear_sin
        np.cos(2 * np.pi * day_of_year / 365)   # dayofyear_cos
    ])
    
    # Trend features (approximate)
    trend = (lag1 - lag7) / 7 if lag7 != 0 else 0
    advanced_features.extend([trend, trend * 4])  # 7d and 30d trends
    
    # Relative position features
    advanced_features.extend([
        lag1 / rolling_mean_7 if rolling_mean_7 != 0 else 1,  # price_vs_7d_mean
        lag1 / rolling_mean_7 if rolling_mean_7 != 0 else 1,  # price_vs_30d_mean
        lag1 / (rolling_mean_7 * 1.1) if rolling_mean_7 != 0 else 1,  # price_vs_7d_max
        lag1 / (rolling_mean_7 * 0.9) if rolling_mean_7 != 0 else 1   # price_vs_7d_min
    ])
    
    # Momentum features
    advanced_features.extend([
        lag1 - lag3,  # momentum_3d
        lag1 - lag7,  # momentum_7d
        lag1 - lag7   # momentum_30d approximation
    ])
    
    # Seasonal features (simplified)
    seasonal_month = rolling_mean_7  # Approximate seasonal pattern
    seasonal_quarter = rolling_mean_7
    advanced_features.extend([seasonal_month, seasonal_quarter])
    
    # Pad or trim to expected size (adjust based on your actual advanced model feature count)
    if len(advanced_features) < target_size:
        # Pad with mean values
        mean_val = np.mean(advanced_features)
        advanced_features.extend([mean_val] * (target_size - len(advanced_features)))
    elif len(advanced_features) > target_size:
        # Trim to target size
        advanced_features = advanced_features[:target_size]
    
    return np.array(advanced_features).reshape(1, -1)

def predict_with_ensemble(features: np.ndarray, xgb_model, lstm_model, lstm_scaler, lstm_target_scaler, weights, crop=None, mandi=None):
    """Make ensemble prediction - handles both complex models and simple models"""
    # Check if this is a simple model (single model, no LSTM)
    if isinstance(weights, dict) and 'simple' in weights:
        # Use simple model directly
        features_scaled = lstm_scaler.transform(features)
        return xgb_model.predict(features_scaled)
    
    # Check if we need to expand features for advanced models
    if features.shape[1] == 11 and lstm_model is not None:
        # This is likely an advanced model that needs more features
        # Detect required feature count from the scaler
        try:
            required_features = lstm_scaler.n_features_in_
        except:
            required_features = 62  # Default fallback
        
        # Create advanced features from basic inputs
        features = create_advanced_features_from_basic(features[0], crop, mandi, required_features)
    
    # Original ensemble logic for complex models
    if lstm_model is not None:
        lstm_pred = predict_with_lstm(features, lstm_model, lstm_scaler, lstm_target_scaler)
        xgb_pred = predict_with_xgboost(features, xgb_model)
        
        # Weighted average
        if isinstance(weights, dict):
            ensemble_pred = weights.get('xgb', 0.5) * xgb_pred + weights.get('lstm', 0.5) * lstm_pred
        else:
            ensemble_pred = 0.5 * xgb_pred + 0.5 * lstm_pred
        
        return ensemble_pred
    else:
        # Fallback to XGBoost only
        return predict_with_xgboost(features, xgb_model)

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
    
    # Prepare features for prediction
    features = np.array([
        request.modal_price_lag1, request.modal_price_lag2, request.modal_price_lag3,
        request.modal_price_lag5, request.modal_price_lag7, request.rolling_mean_7,
        request.rolling_std_7, request.day_of_year, request.month, request.month_sin, request.month_cos
    ]).reshape(1, -1)
    
    if model_type == "ensemble":
        # Try to load fast models first
        xgb_model, lstm_model, lstm_scaler, lstm_target_scaler, weights = load_fast_models(crop, mandi)
        ensemble_pred = predict_with_ensemble(features, xgb_model, lstm_model, lstm_scaler, lstm_target_scaler, weights, request.crop, request.mandi)[0]
        if xgb_model is not None:
            pred = float(ensemble_pred)
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

@router.post("/forecast", response_model=ForecastResponse)
def forecast_prices(request: ForecastRequest):
    """Generate price forecast for multiple days"""
    crop = request.crop.lower()
    mandi = request.mandi.lower()
    
    # Load models
    xgb_model, lstm_model, lstm_scaler, lstm_target_scaler, weights = load_fast_models(crop, mandi)
    
    if xgb_model is None:
        # Fallback to old models
        try:
            xgb_model = joblib.load(f"app/data/processed/xgb_{crop}_{mandi}.joblib")
            lstm_model = load_model(f"app/data/processed/lstm_{crop}_{mandi}.h5")
            
            # Try different scaler naming conventions
            try:
                lstm_scaler = joblib.load(f"app/data/processed/lstm_scaler_{crop}_{mandi}.joblib")
                lstm_target_scaler = joblib.load(f"app/data/processed/lstm_target_scaler_{crop}_{mandi}.joblib")
            except FileNotFoundError:
                # Fallback to old naming convention
                lstm_scaler = joblib.load(f"app/data/processed/lstm_norm_{crop}_{mandi}.joblib")
                lstm_target_scaler = None
            
            weights = {'xgb': 0.5, 'lstm': 0.5}
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=f"Models not found for {crop} in {mandi}: {str(e)}")
    
    # Prepare features for forecasting
    features = np.array([
        request.modal_price_lag1, request.modal_price_lag2, request.modal_price_lag3,
        request.modal_price_lag5, request.modal_price_lag7, request.rolling_mean_7,
        request.rolling_std_7, request.day_of_year, request.month, request.month_sin, request.month_cos
    ]).reshape(1, -1)
    
    # Generate monthly forecast for 12 months
    forecast = []
    current_features = features.copy()
    base_date = pd.Timestamp.now()
    
    # Use months parameter from request, default to 12
    months_to_forecast = getattr(request, 'months', 12)
    
    for month in range(months_to_forecast):
        # Calculate future date (monthly intervals)
        future_date = base_date + pd.DateOffset(months=month+1)
        
        # Update temporal features for the future month
        new_features = current_features[0].copy()
        new_features[7] = future_date.dayofyear  # day_of_year
        new_features[8] = future_date.month      # month
        new_features[9] = np.sin(2 * np.pi * future_date.month / 12)  # month_sin
        new_features[10] = np.cos(2 * np.pi * future_date.month / 12)  # month_cos
        current_features = new_features.reshape(1, -1)
        
        # Make ensemble prediction
        ensemble_pred = predict_with_ensemble(
            current_features, xgb_model, lstm_model, 
            lstm_scaler, lstm_target_scaler, weights
        )
        
        forecast.append({
            "month": future_date.strftime("%B %Y"),  # "August 2025", "September 2025", etc.
            "predicted_price": float(ensemble_pred[0]),
            "date": future_date.strftime("%Y-%m-%d"),
            "month_name": future_date.strftime("%B %Y"),
            "year": future_date.year,
            "month_number": future_date.month,
            "sequence": month + 1  # Keep sequence number for ordering if needed
        })
        
        # Update lag features with predicted price for next iteration
        if month < months_to_forecast - 1:
            # Shift lag prices (simulate time progression)
            new_features[1:5] = new_features[0:4]  # lag2-5 become lag1-4
            new_features[0] = ensemble_pred[0]     # new prediction becomes lag1
            
            # Update rolling statistics (simple approximation)
            new_features[5] = ensemble_pred[0]  # rolling_mean_7 approximation
            new_features[6] = abs(ensemble_pred[0] - new_features[1]) * 0.1  # rolling_std_7 approximation
    
    return ForecastResponse(forecast=forecast, model_type="ensemble")
