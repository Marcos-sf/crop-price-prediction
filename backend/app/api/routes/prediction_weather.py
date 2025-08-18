from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import joblib
import numpy as np
import os
from datetime import datetime, timedelta
import pandas as pd
from tensorflow.keras.models import load_model
import requests

router = APIRouter()

class WeatherPredictionRequest(BaseModel):
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
    # Weather features
    tavg: float = None
    tmin: float = None
    tmax: float = None
    prcp: float = None
    wspd: float = None
    pres: float = None
    model_type: str = "weather_ensemble"  # "weather_xgboost", "weather_lstm", or "weather_ensemble"

class WeatherForecastRequest(BaseModel):
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
    # Weather features
    tavg: float = None
    tmin: float = None
    tmax: float = None
    prcp: float = None
    wspd: float = None
    pres: float = None
    months: int = 12
    model_type: str = "weather_ensemble"

class WeatherPredictionResponse(BaseModel):
    predicted_price: float
    model_type: str
    weather_impact: dict
    confidence_score: float

class WeatherForecastResponse(BaseModel):
    forecast: list
    model_type: str
    weather_insights: dict

class WeatherLatestFeaturesResponse(BaseModel):
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
    # Weather features
    tavg: float
    tmin: float
    tmax: float
    prcp: float
    wspd: float
    pres: float
    latest_date: str
    weather_summary: dict

def find_column(df, possible_names):
    """Find column by exact match first, then case insensitive"""
    for name in possible_names:
        if name in df.columns:
            return name
    
    for col in df.columns:
        col_lower = col.strip().lower()
        for name in possible_names:
            if col_lower == name.lower():
                return col
    return None

def fetch_current_weather(lat, lon):
    """Fetch current weather data for a location"""
    try:
        # Using OpenWeatherMap API (you can replace with your preferred weather API)
        api_key = "5a955c4200def2ed3234e4144393c036"  # Replace with your API key
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": api_key,
            "units": "metric"
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                "tavg": data["main"]["temp"],
                "tmin": data["main"]["temp_min"],
                "tmax": data["main"]["temp_max"],
                "prcp": data.get("rain", {}).get("1h", 0),
                "wspd": data["wind"]["speed"],
                "pres": data["main"]["pressure"]
            }
        else:
            return None
    except Exception as e:
        print(f"Error fetching weather: {e}")
        return None

def create_weather_enhanced_features(request, historical_data=None):
    """Create weather-enhanced features for prediction"""
    # Base features
    base_features = [
        request.modal_price_lag1,
        request.modal_price_lag2,
        request.modal_price_lag3,
        request.modal_price_lag5,
        request.modal_price_lag7,
        request.rolling_mean_7,
        request.rolling_std_7,
        request.day_of_year,
        request.month,
        request.month_sin,
        request.month_cos
    ]
    
    # Weather features (use provided values or defaults)
    weather_features = [
        request.tavg or 25.0,  # Default temperature
        request.tmin or 20.0,
        request.tmax or 30.0,
        request.prcp or 0.0,
        request.wspd or 5.0,
        request.pres or 1013.0
    ]
    
    # Additional derived features
    temp_range = (request.tmax or 30.0) - (request.tmin or 20.0)
    weather_features.append(temp_range)
    
    # Combine all features
    all_features = base_features + weather_features
    
    return np.array(all_features).reshape(1, -1)

def predict_with_weather_xgboost(features, crop, mandi):
    """Predict using weather-enhanced XGBoost model"""
    try:
        model_path = f"app/data/processed/xgb_weather_{crop.lower()}_{mandi.lower()}.joblib"
        if not os.path.exists(model_path):
            # Fallback to regular model
            model_path = f"app/data/processed/xgb_{crop.lower()}_{mandi.lower()}.joblib"
        
        model = joblib.load(model_path)
        prediction = model.predict(features)[0]
        return prediction
    except Exception as e:
        print(f"Error in weather XGBoost prediction: {e}")
        return None

def predict_with_weather_lstm(features, crop, mandi):
    """Predict using weather-enhanced LSTM model"""
    try:
        model_path = f"app/data/processed/lstm_weather_{crop.lower()}_{mandi.lower()}.h5"
        scaler_path = f"app/data/processed/lstm_weather_scaler_{crop.lower()}_{mandi.lower()}.joblib"
        
        if not os.path.exists(model_path):
            # Fallback to regular model
            model_path = f"app/data/processed/lstm_{crop.lower()}_{mandi.lower()}.h5"
            scaler_path = f"app/data/processed/lstm_scaler_{crop.lower()}_{mandi.lower()}.joblib"
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = load_model(model_path)
            scaler = joblib.load(scaler_path)
            
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Reshape for LSTM if needed
            if len(features_scaled.shape) == 2:
                features_scaled = features_scaled.reshape(1, 1, features_scaled.shape[1])
            
            prediction = model.predict(features_scaled)[0][0]
            return prediction
        else:
            return None
    except Exception as e:
        print(f"Error in weather LSTM prediction: {e}")
        return None

@router.get("/weather-latest-features", response_model=WeatherLatestFeaturesResponse)
def get_weather_latest_features(
    crop: str = Query(..., description="Crop name"),
    mandi: str = Query(..., description="Mandi name")
):
    """Get latest features including weather data"""
    crop = crop.lower()
    mandi = mandi.lower()
    
    # Try weather-enhanced data first
    weather_file = f"app/data/processed/{crop}_{mandi}_with_weather.csv"
    regular_file = f"app/data/processed/{crop}_{mandi}_for_training.csv"
    
    if os.path.exists(weather_file):
        df = pd.read_csv(weather_file)
        has_weather = True
    elif os.path.exists(regular_file):
        df = pd.read_csv(regular_file)
        has_weather = False
    else:
        raise HTTPException(status_code=404, detail=f"Data not found for {crop} in {mandi}.")
    
    # Find columns
    date_col = find_column(df, ['date', 'arrival_date', 'price_date', 'Arrival_Date'])
    modal_price_col = find_column(df, ['modal_price', 'Modal_Price', 'modal price (rs./quintal)', 'modal price'])
    
    if not date_col or not modal_price_col:
        raise HTTPException(status_code=500, detail="Required columns not found in data.")
    
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col, modal_price_col])
    df = df.sort_values(date_col)
    
    if len(df) < 8:
        raise HTTPException(status_code=400, detail="Not enough data to compute features.")
    
    # Get latest data
    latest_idx = df.index[-1]
    lag1_idx = df.index[-2]
    lag2_idx = df.index[-3]
    lag3_idx = df.index[-4]
    lag5_idx = df.index[-6]
    lag7_idx = df.index[-8]
    
    # Price features
    modal_price_lag1 = float(df.loc[lag1_idx, modal_price_col])
    modal_price_lag2 = float(df.loc[lag2_idx, modal_price_col])
    modal_price_lag3 = float(df.loc[lag3_idx, modal_price_col])
    modal_price_lag5 = float(df.loc[lag5_idx, modal_price_col])
    modal_price_lag7 = float(df.loc[lag7_idx, modal_price_col])
    
    recent_prices = df[modal_price_col].iloc[-7:].values
    rolling_mean_7 = float(np.mean(recent_prices))
    rolling_std_7 = float(np.std(recent_prices))
    
    # Temporal features
    latest_date = df.loc[latest_idx, date_col]
    day_of_year = latest_date.timetuple().tm_yday
    month = latest_date.month
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    # Weather features
    weather_data = {
        "tavg": 25.0, "tmin": 20.0, "tmax": 30.0,
        "prcp": 0.0, "wspd": 5.0, "pres": 1013.0
    }
    
    if has_weather:
        weather_cols = ['tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'pres']
        for col in weather_cols:
            if col in df.columns:
                value = df.loc[latest_idx, col]
                if pd.notna(value):
                    weather_data[col] = float(value)
    
    # Weather summary
    weather_summary = {
        "temperature_range": f"{weather_data['tmin']:.1f}°C - {weather_data['tmax']:.1f}°C",
        "precipitation": f"{weather_data['prcp']:.1f} mm",
        "wind_speed": f"{weather_data['wspd']:.1f} m/s",
        "pressure": f"{weather_data['pres']:.0f} hPa"
    }
    
    return WeatherLatestFeaturesResponse(
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
        tavg=weather_data["tavg"],
        tmin=weather_data["tmin"],
        tmax=weather_data["tmax"],
        prcp=weather_data["prcp"],
        wspd=weather_data["wspd"],
        pres=weather_data["pres"],
        latest_date=latest_date.strftime("%Y-%m-%d"),
        weather_summary=weather_summary
    )

@router.post("/weather-predict", response_model=WeatherPredictionResponse)
def weather_predict_price(request: WeatherPredictionRequest):
    """Predict price with weather features"""
    crop = request.crop.lower()
    mandi = request.mandi.lower()
    model_type = request.model_type.lower()
    
    try:
        date_obj = datetime.strptime(request.date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    
    # Create features
    features = create_weather_enhanced_features(request)
    
    # Make predictions
    predictions = {}
    
    if model_type in ["weather_xgboost", "weather_ensemble"]:
        xgb_pred = predict_with_weather_xgboost(features, crop, mandi)
        if xgb_pred is not None:
            predictions["xgboost"] = xgb_pred
    
    if model_type in ["weather_lstm", "weather_ensemble"]:
        lstm_pred = predict_with_weather_lstm(features, crop, mandi)
        if lstm_pred is not None:
            predictions["lstm"] = lstm_pred
    
    if not predictions:
        raise HTTPException(status_code=500, detail="No models available for prediction.")
    
    # Combine predictions
    if model_type == "weather_ensemble" and len(predictions) > 1:
        # Weighted ensemble
        weights = {"xgboost": 0.6, "lstm": 0.4}
        final_prediction = sum(predictions[model] * weights[model] for model in predictions.keys())
        used_model = "weather_ensemble"
    else:
        final_prediction = list(predictions.values())[0]
        used_model = list(predictions.keys())[0]
    
    # Calculate weather impact
    weather_impact = {
        "temperature_effect": "neutral",
        "precipitation_effect": "neutral",
        "wind_effect": "neutral"
    }
    
    if request.tavg:
        if request.tavg > 30:
            weather_impact["temperature_effect"] = "high_temp_negative"
        elif request.tavg < 15:
            weather_impact["temperature_effect"] = "low_temp_negative"
    
    if request.prcp:
        if request.prcp > 10:
            weather_impact["precipitation_effect"] = "high_rain_negative"
        elif request.prcp < 1:
            weather_impact["precipitation_effect"] = "low_rain_positive"
    
    # Calculate confidence score
    confidence_score = 0.85  # Base confidence
    if len(predictions) > 1:
        # Higher confidence if multiple models agree
        pred_values = list(predictions.values())
        variance = np.var(pred_values)
        if variance < 1000:  # Low variance = high agreement
            confidence_score = 0.95
        elif variance > 5000:  # High variance = low agreement
            confidence_score = 0.75
    
    return WeatherPredictionResponse(
        predicted_price=float(final_prediction),
        model_type=used_model,
        weather_impact=weather_impact,
        confidence_score=confidence_score
    )

@router.post("/weather-forecast", response_model=WeatherForecastResponse)
def weather_forecast_prices(request: WeatherForecastRequest):
    """Generate weather-enhanced price forecast"""
    crop = request.crop.lower()
    mandi = request.mandi.lower()
    model_type = request.model_type.lower()
    
    forecast = []
    current_date = datetime.strptime(request.start_date, "%Y-%m-%d")
    
    # Mandi coordinates for weather fetching
    mandi_coords = {
        "sirsi": (14.6197, 74.8354),
        "yellapur": (14.9621, 74.7097),
        "siddapur": (14.3432, 74.8945),
        "shimoga": (13.9299, 75.5681),
        "sagar": (14.1667, 75.0333),
        "kumta": (14.4251, 74.4189),
        "bangalore": (12.9716, 77.5946),
        "arasikere": (13.3132, 76.2577),
        "channarayapatna": (12.9066, 76.3886),
        "ramanagara": (12.7217, 77.2812),
        "sira": (13.7411, 76.9042),
        "tumkur": (13.3409, 77.1017)
    }
    
    lat, lon = mandi_coords.get(mandi, (12.9716, 77.5946))  # Default to Bangalore
    
    for month in range(request.months):
        # Update features for each month
        forecast_date = current_date + timedelta(days=30*month)
        
        # Fetch weather for forecast date (simplified - using current weather)
        weather_data = fetch_current_weather(lat, lon) or {
            "tavg": 25.0, "tmin": 20.0, "tmax": 30.0,
            "prcp": 0.0, "wspd": 5.0, "pres": 1013.0
        }
        
        # Create forecast request
        forecast_request = WeatherPredictionRequest(
            crop=request.crop,
            mandi=request.mandi,
            date=forecast_date.strftime("%Y-%m-%d"),
            modal_price_lag1=request.modal_price_lag1,
            modal_price_lag2=request.modal_price_lag2,
            modal_price_lag3=request.modal_price_lag3,
            modal_price_lag5=request.modal_price_lag5,
            modal_price_lag7=request.modal_price_lag7,
            rolling_mean_7=request.rolling_mean_7,
            rolling_std_7=request.rolling_std_7,
            day_of_year=forecast_date.timetuple().tm_yday,
            month=forecast_date.month,
            month_sin=np.sin(2 * np.pi * forecast_date.month / 12),
            month_cos=np.cos(2 * np.pi * forecast_date.month / 12),
            tavg=weather_data["tavg"],
            tmin=weather_data["tmin"],
            tmax=weather_data["tmax"],
            prcp=weather_data["prcp"],
            wspd=weather_data["wspd"],
            pres=weather_data["pres"],
            model_type=model_type
        )
        
        # Make prediction
        prediction_response = weather_predict_price(forecast_request)
        
        forecast.append({
            "date": forecast_date.strftime("%Y-%m-%d"),
            "predicted_price": prediction_response.predicted_price,
            "weather_conditions": weather_data,
            "confidence": prediction_response.confidence_score
        })
    
    # Weather insights
    weather_insights = {
        "temperature_trend": "stable",
        "precipitation_forecast": "low",
        "optimal_growing_conditions": "favorable"
    }
    
    return WeatherForecastResponse(
        forecast=forecast,
        model_type=model_type,
        weather_insights=weather_insights
    ) 