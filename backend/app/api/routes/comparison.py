from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
from datetime import datetime, timedelta

router = APIRouter()

class ComparisonRequest(BaseModel):
    crop: str
    mandis: List[str]
    months: int = 12

class ComparisonResponse(BaseModel):
    crop: str
    mandis: List[str]
    comparison_data: List[dict]
    summary_statistics: Dict[str, Dict[str, float]]
    timestamp: str

@router.post("/compare", response_model=ComparisonResponse)
async def compare_prices(request: ComparisonRequest):
    """
    Compare prices for the same crop across multiple mandis
    """
    try:
        if len(request.mandis) < 2:
            raise HTTPException(status_code=400, detail="At least 2 mandis required for comparison")
        
        if len(request.mandis) > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 mandis allowed for comparison")
        
        # Generate comparison data
        comparison_data = []
        base_prices = {
            "sirsi": 320,
            "shimoga": 315,
            "chikmagalur": 330,
            "madikeri": 325,
            "tiptur": 310,
            "hassan": 318,
            "sullia": 335,
            "tumkur": 312
        }
        
        # Generate data for each month
        for i in range(request.months):
            month_date = datetime.now() + timedelta(days=30*i)
            month_data = {"month": month_date.strftime("%b %Y")}
            
            for mandi in request.mandis:
                base_price = base_prices.get(mandi.lower(), 320)
                price = base_price + 20 * np.sin(i/3) + np.random.normal(0, 10)
                month_data[mandi] = round(price, 2)
            
            comparison_data.append(month_data)
        
        # Calculate summary statistics for each mandi
        summary_statistics = {}
        for mandi in request.mandis:
            prices = [item[mandi] for item in comparison_data]
            summary_statistics[mandi] = {
                "average_price": round(np.mean(prices), 2),
                "highest_price": round(max(prices), 2),
                "lowest_price": round(min(prices), 2),
                "price_volatility": round(((max(prices) - min(prices)) / np.mean(prices)) * 100, 1)
            }
        
        return ComparisonResponse(
            crop=request.crop,
            mandis=request.mandis,
            comparison_data=comparison_data,
            summary_statistics=summary_statistics,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

@router.get("/comparison-metrics")
async def get_comparison_metrics():
    """
    Get available comparison metrics
    """
    return {
        "max_mandis": 5,
        "min_mandis": 2,
        "available_metrics": [
            "price_comparison",
            "average_price",
            "price_volatility",
            "trend_comparison",
            "market_ranking"
        ],
        "supported_timeframes": ["3_months", "6_months", "12_months"]
    } 