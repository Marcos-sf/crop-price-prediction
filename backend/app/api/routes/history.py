import os
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any

router = APIRouter()

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

@router.get("/history")
async def get_history(
    crop: str = Query(..., description="Crop name"),
    mandi: str = Query(..., description="Mandi name")
):
    """
    Return real historical mandi price data for given crop and mandi.
    """
    try:
        crop = crop.lower()
        mandi = mandi.lower()
        
        # Try different file paths for historical data
        possible_paths = [
            f"app/data/processed/{crop}_{mandi}_for_training.csv",
            f"app/data/processed/{crop}_{mandi}_with_weather.csv",
            f"app/data/processed/{crop}_{mandi}_filtered.csv"
        ]
        
        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break
        
        if not file_path:
            raise HTTPException(status_code=404, detail=f"No historical data found for {crop} in {mandi}")

        # Load CSV
        df = pd.read_csv(file_path)
        
        # Find date column
        date_col = find_column(df, ['date', 'arrival_date', 'price_date', 'Arrival_Date'])
        if not date_col:
            raise HTTPException(status_code=500, detail="No date column found in data")

        # Find modal price column
        modal_price_col = find_column(df, ['modal_price', 'Modal_Price', 'modal price (rs./quintal)', 'modal price'])
        if not modal_price_col:
            raise HTTPException(status_code=500, detail="No modal price column found in data")
        
        # Convert date column to datetime
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col, modal_price_col])
        df = df.sort_values(date_col)
        
        # Convert to dict format expected by frontend
        history = []
        for _, row in df.iterrows():
            history.append({
                "date": row[date_col].strftime("%Y-%m-%d"),
                "modal_price": float(row[modal_price_col])
            })
        
        return {"history": history}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load history: {str(e)}")

@router.get("/history-summary")
async def get_history_summary(
    crop: str = Query(..., description="Crop name"),
    mandi: str = Query(..., description="Mandi name")
):
    """
    Return summary statistics for historical data.
    """
    try:
        crop = crop.lower()
        mandi = mandi.lower()
        
        # Get historical data
        history_response = await get_history(crop=crop, mandi=mandi)
        history = history_response["history"]
        
        if not history:
            return {"summary": {}}
        
        # Calculate summary statistics
        prices = [item["modal_price"] for item in history]
        
        summary = {
            "total_records": len(history),
            "date_range": {
                "start": history[0]["date"],
                "end": history[-1]["date"]
            },
            "price_stats": {
                "min": min(prices),
                "max": max(prices),
                "mean": sum(prices) / len(prices),
                "median": sorted(prices)[len(prices) // 2]
            },
            "volatility": ((max(prices) - min(prices)) / (sum(prices) / len(prices))) * 100
        }
        
        return {"summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate summary: {str(e)}")
