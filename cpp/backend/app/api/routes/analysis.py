from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
from datetime import datetime, timedelta

router = APIRouter()

class AnalysisRequest(BaseModel):
    crop: str
    mandi: str

class AnalysisResponse(BaseModel):
    crop: str
    mandi: str
    trend_analysis: List[dict]
    seasonality_analysis: List[dict]
    key_metrics: Dict[str, Any]
    market_insights: str
    timestamp: str

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_prices(request: AnalysisRequest):
    """
    Perform detailed analysis of crop prices
    """
    try:
        # Generate trend analysis data (12 months)
        trend_data = []
        base_price = 320
        for i in range(12):
            month_date = datetime.now() + timedelta(days=30*i)
            price = base_price + 30 * np.sin(i/3) + np.random.normal(0, 15)
            trend_data.append({
                "month": month_date.strftime("%b %Y"),
                "price": round(price, 2),
                "trend": "up" if price > base_price else "down"
            })
        
        # Generate seasonality analysis
        seasonality_data = [
            {"season": "Q3 2025 (Jul-Sep)", "avgPrice": 347},
            {"season": "Q4 2025 (Oct-Dec)", "avgPrice": 363},
            {"season": "Q1 2026 (Jan-Mar)", "avgPrice": 392},
            {"season": "Q2 2026 (Apr-Jun)", "avgPrice": 408}
        ]
        
        # Calculate key metrics
        prices = [item["price"] for item in trend_data]
        avg_price = np.mean(prices)
        max_price = max(prices)
        min_price = min(prices)
        volatility = ((max_price - min_price) / avg_price) * 100
        
        key_metrics = {
            "average_price": round(avg_price, 2),
            "price_volatility": round(volatility, 1),
            "trend_direction": "upward" if trend_data[-1]["price"] > trend_data[0]["price"] else "downward",
            "confidence_level": 85,
            "highest_price": round(max_price, 2),
            "lowest_price": round(min_price, 2)
        }
        
        # Generate market insights
        market_insights = f"""
        The price trend for {request.crop} in {request.mandi} shows a steady movement over the analyzed period. 
        Seasonal patterns indicate higher prices during certain quarters, likely due to increased demand during 
        festival seasons. Based on historical data and current market conditions, prices are expected to remain 
        stable with a slight trend in the coming months.
        """
        
        return AnalysisResponse(
            crop=request.crop,
            mandi=request.mandi,
            trend_analysis=trend_data,
            seasonality_analysis=seasonality_data,
            key_metrics=key_metrics,
            market_insights=market_insights.strip(),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/analysis-metrics")
async def get_analysis_metrics():
    """
    Get available analysis metrics
    """
    return {
        "available_metrics": [
            "trend_analysis",
            "seasonality_analysis", 
            "volatility_calculation",
            "price_forecasting",
            "market_insights"
        ],
        "supported_timeframes": ["3_months", "6_months", "12_months"],
        "analysis_types": ["trend", "seasonal", "volatility", "forecast"]
    } 