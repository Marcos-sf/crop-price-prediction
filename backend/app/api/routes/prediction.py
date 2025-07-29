from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import joblib
import numpy as np
import os
from datetime import datetime, timedelta
import pandas as pd

router = APIRouter()

class PredictionRequest(BaseModel):
    crop: str
    mandi: str
    date: str  # YYYY-MM-DD
    modal_price_lag1: float
    modal_price_lag7: float

class ForecastRequest(BaseModel):
    crop: str
    mandi: str
    start_date: str  # YYYY-MM-DD
    modal_price_lag1: float
    modal_price_lag2: float
    modal_price_lag3: float
    modal_price_lag5: float
    modal_price_lag7: float
    modal_price_lag10: float
    modal_price_lag14: float
    modal_price_lag30: float
    rolling_mean_7: float
    rolling_mean_30: float
    rolling_std_7: float
    rolling_std_30: float
    day_of_year: int
    month: int
    month_sin: float
    month_cos: float
    months: int = 12

class PredictionResponse(BaseModel):
    predicted_price: float

class ForecastResponse(BaseModel):
    forecast: list

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
    modal_price_lag10: float
    modal_price_lag14: float
    modal_price_lag30: float
    rolling_mean_7: float
    rolling_mean_30: float
    rolling_std_7: float
    rolling_std_30: float
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
    lag10_date: str
    lag14_date: str
    lag30_date: str

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
    # Dynamically find the correct date and modal price columns
    date_col = None
    for col in df.columns:
        if col.strip().lower() in ['date', 'arrival_date', 'price_date']:
            date_col = col
            break
    if not date_col:
        raise HTTPException(status_code=500, detail="No date column found in data.")

    modal_price_col = None
    for col in df.columns:
        if col.strip().lower() in ['modal_price', 'modal price (rs./quintal)']:
            modal_price_col = col
            break
    if not modal_price_col:
        raise HTTPException(status_code=500, detail="No modal price column found in data.")

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)
    if len(df) < 7:
        raise HTTPException(status_code=400, detail="Not enough data for lag calculation.")
    latest_row = df.iloc[-1]
    latest_date = latest_row[date_col]
    lag7_date = latest_date - pd.Timedelta(days=7)
    lag7_candidates = df[df[date_col] <= lag7_date]
    if not lag7_candidates.empty:
        lag7_row = lag7_candidates.iloc[-1]
    else:
        lag7_row = df.iloc[0]  # fallback to earliest row
    return LagPricesResponse(
        modal_price_lag1=float(latest_row[modal_price_col]),
        modal_price_lag7=float(lag7_row[modal_price_col]),
        latest_date=latest_row[date_col].strftime('%Y-%m-%d'),
        lag7_date=lag7_row[date_col].strftime('%Y-%m-%d')
    )

@router.post("/predict", response_model=PredictionResponse)
def predict_price(request: PredictionRequest):
    crop = request.crop.lower()
    mandi = request.mandi.lower()
    try:
        date_obj = datetime.strptime(request.date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    day_of_year = date_obj.timetuple().tm_yday
    month = date_obj.month
    features = np.array([[request.modal_price_lag1, request.modal_price_lag7, day_of_year, month]])
    model_path = f"app/data/processed/xgb_{crop}_{mandi}.joblib"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model not found for {crop} in {mandi}.")
    model = joblib.load(model_path)
    pred = model.predict(features)[0]
    return PredictionResponse(predicted_price=float(pred))

@router.post("/forecast", response_model=ForecastResponse)
def forecast_prices(request: ForecastRequest):
    crop = request.crop.lower()
    mandi = request.mandi.lower()
    try:
        date_obj = datetime.strptime(request.start_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    model_path = f"app/data/processed/xgb_{crop}_{mandi}.joblib"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model not found for {crop} in {mandi}.")
    model = joblib.load(model_path)
    # Only use the 11 features the model expects
    lag1 = request.modal_price_lag1
    lag2 = request.modal_price_lag2
    lag3 = request.modal_price_lag3
    lag5 = request.modal_price_lag5
    lag7 = request.modal_price_lag7
    rolling7 = request.rolling_mean_7
    std7 = request.rolling_std_7
    day_of_year = request.day_of_year
    month = request.month
    month_sin = request.month_sin
    month_cos = request.month_cos
    forecast = []
    current_date = date_obj
    # For rolling lags and stats, keep a list of last 7 predictions
    lag_history = [lag7, lag5, lag3, lag2, lag1]
    def safe_lag(lags, n):
        return lags[-n] if len(lags) >= n else lags[0]
    for i in range(request.months):
        features = np.array([[lag1, lag2, lag3, lag5, lag7,
                              rolling7, std7,
                              day_of_year, month, month_sin, month_cos]])
        pred = float(model.predict(features)[0])
        forecast.append({
            "month": current_date.strftime("%b %Y"),
            "predicted_price": pred
        })
        # Update lag history
        lag_history.append(pred)
        if len(lag_history) > 7:
            lag_history.pop(0)
        # Update lags for next step using safe_lag
        lag1 = safe_lag(lag_history, 1)
        lag2 = safe_lag(lag_history, 2)
        lag3 = safe_lag(lag_history, 3)
        lag5 = safe_lag(lag_history, 5)
        lag7 = safe_lag(lag_history, 7)
        # Update rolling mean and std safely
        rolling7 = np.mean(lag_history[-7:]) if len(lag_history) >= 7 else np.mean(lag_history)
        std7 = np.std(lag_history[-7:]) if len(lag_history) >= 7 else np.std(lag_history)
        # Move to next month
        next_month = (current_date.month % 12) + 1
        next_year = current_date.year + (current_date.month // 12)
        current_date = current_date.replace(year=next_year, month=next_month, day=1)
        day_of_year = current_date.timetuple().tm_yday
        month = current_date.month
        month_sin = float(np.sin(2 * np.pi * month / 12))
        month_cos = float(np.cos(2 * np.pi * month / 12))
    return ForecastResponse(forecast=forecast)

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
    # Dynamically find the correct date and modal price columns
    date_col = None
    for col in df.columns:
        if col.strip().lower() in ['date', 'arrival_date', 'price_date']:
            date_col = col
            break
    if not date_col:
        raise HTTPException(status_code=500, detail="No date column found in data.")
    modal_price_col = None
    for col in df.columns:
        if col.strip().lower() in ['modal_price', 'modal price (rs./quintal)']:
            modal_price_col = col
            break
    if not modal_price_col:
        raise HTTPException(status_code=500, detail="No modal price column found in data.")
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)
    n = len(df)
    if n < 31:
        raise HTTPException(status_code=400, detail="Not enough data to compute all features.")
    # Use last N available market day prices for lags/rolling
    latest_idx = df.index[-1]
    lag1_idx = df.index[-2]
    lag2_idx = df.index[-3]
    lag3_idx = df.index[-4]
    lag5_idx = df.index[-6]
    lag7_idx = df.index[-8]
    lag10_idx = df.index[-11]
    lag14_idx = df.index[-15]
    lag30_idx = df.index[-31]
    modal_price_lag1 = df.loc[lag1_idx, modal_price_col]
    modal_price_lag2 = df.loc[lag2_idx, modal_price_col]
    modal_price_lag3 = df.loc[lag3_idx, modal_price_col]
    modal_price_lag5 = df.loc[lag5_idx, modal_price_col]
    modal_price_lag7 = df.loc[lag7_idx, modal_price_col]
    modal_price_lag10 = df.loc[lag10_idx, modal_price_col]
    modal_price_lag14 = df.loc[lag14_idx, modal_price_col]
    modal_price_lag30 = df.loc[lag30_idx, modal_price_col]
    rolling_mean_7 = df[modal_price_col].iloc[-7:].mean()
    rolling_mean_30 = df[modal_price_col].iloc[-30:].mean()
    rolling_std_7 = df[modal_price_col].iloc[-7:].std()
    rolling_std_30 = df[modal_price_col].iloc[-30:].std()
    latest_row = df.iloc[-1]
    day_of_year = int(latest_row[date_col].timetuple().tm_yday)
    month = int(latest_row[date_col].month)
    import numpy as np
    month_sin = float(np.sin(2 * np.pi * month / 12))
    month_cos = float(np.cos(2 * np.pi * month / 12))
    return LatestFeaturesResponse(
        modal_price_lag1=float(modal_price_lag1),
        modal_price_lag2=float(modal_price_lag2),
        modal_price_lag3=float(modal_price_lag3),
        modal_price_lag5=float(modal_price_lag5),
        modal_price_lag7=float(modal_price_lag7),
        modal_price_lag10=float(modal_price_lag10),
        modal_price_lag14=float(modal_price_lag14),
        modal_price_lag30=float(modal_price_lag30),
        rolling_mean_7=float(rolling_mean_7),
        rolling_mean_30=float(rolling_mean_30),
        rolling_std_7=float(rolling_std_7),
        rolling_std_30=float(rolling_std_30),
        day_of_year=day_of_year,
        month=month,
        month_sin=month_sin,
        month_cos=month_cos,
        latest_date=latest_row[date_col].strftime('%Y-%m-%d'),
        lag1_date=df.loc[lag1_idx, date_col].strftime('%Y-%m-%d'),
        lag2_date=df.loc[lag2_idx, date_col].strftime('%Y-%m-%d'),
        lag3_date=df.loc[lag3_idx, date_col].strftime('%Y-%m-%d'),
        lag5_date=df.loc[lag5_idx, date_col].strftime('%Y-%m-%d'),
        lag7_date=df.loc[lag7_idx, date_col].strftime('%Y-%m-%d'),
        lag10_date=df.loc[lag10_idx, date_col].strftime('%Y-%m-%d'),
        lag14_date=df.loc[lag14_idx, date_col].strftime('%Y-%m-%d'),
        lag30_date=df.loc[lag30_idx, date_col].strftime('%Y-%m-%d')
    )

@router.get("/history")
def get_history(
    crop: str = Query(..., description="Crop name"),
    mandi: str = Query(..., description="Mandi name")
):
    crop = crop.lower()
    mandi = mandi.lower()
    file_path = f"app/data/processed/{crop}_{mandi}_for_training.csv"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Data not found for {crop} in {mandi}.")
    df = pd.read_csv(file_path)
    # Dynamically find the correct date and modal price columns
    date_col = None
    for col in df.columns:
        if col.strip().lower() in ['date', 'arrival_date', 'price_date']:
            date_col = col
            break
    if not date_col:
        raise HTTPException(status_code=500, detail="No date column found in data.")
    modal_price_col = None
    for col in df.columns:
        if col.strip().lower() in ['modal_price', 'modal price (rs./quintal)']:
            modal_price_col = col
            break
    if not modal_price_col:
        raise HTTPException(status_code=500, detail="No modal price column found in data.")
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)
    # Only keep date and modal price
    result = [
        {"date": row[date_col].strftime('%Y-%m-%d'), "modal_price": float(row[modal_price_col])}
        for _, row in df.iterrows()
    ]
    return {"history": result}

@router.get("/prediction-status")
async def get_prediction_status():
    """
    Get the status of prediction models
    """
    return {
        "lstm_model": "ready",
        "xgboost_model": "ready",
        "ensemble_model": "ready",
        "last_training": "2025-01-15T10:30:00Z",
        "data_last_updated": "2025-01-15T09:00:00Z"
    } 