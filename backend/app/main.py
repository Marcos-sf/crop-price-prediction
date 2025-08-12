from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import routers
from app.api.routes import prediction, analysis, comparison
from app.api.routes.news import router as news_router

app = FastAPI(
    title="Crop Price Prediction API",
    description="API for predicting crop prices using LSTM and XGBoost models",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(prediction.router, prefix="/api/v1", tags=["predictions"])
app.include_router(analysis.router, prefix="/api/v1", tags=["analysis"])
app.include_router(comparison.router, prefix="/api/v1", tags=["comparison"])
app.include_router(news_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Crop Price Prediction API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "crop-price-prediction-api"}

@app.get("/api/v1/crops")
async def get_crops():
    """Get list of available crops with trained models"""
    crops = [
        {"id": "arecanut", "name": "Arecanut"},
        {"id": "coconut", "name": "Coconut"}
    ]
    return {"crops": crops}

@app.get("/api/v1/mandis")
async def get_mandis(crop: str = None):
    """Get list of available mandis for a specific crop"""
    
    # Valid crop-mandi combinations based on trained models
    valid_combinations = {
        "arecanut": [
            {"id": "sirsi", "name": "Sirsi"},
            {"id": "yellapur", "name": "Yellapur"},
            {"id": "siddapur", "name": "Siddapur"},
            {"id": "shimoga", "name": "Shimoga"},
            {"id": "sagar", "name": "Sagar"},
            {"id": "kumta", "name": "Kumta"}
        ],
        "coconut": [
            {"id": "bangalore", "name": "Bangalore"},
            {"id": "arasikere", "name": "Arasikere"},
            {"id": "channarayapatna", "name": "Channarayapatna"},
            {"id": "ramanagara", "name": "Ramanagara"},
            {"id": "sira", "name": "Sira"},
            {"id": "tumkur", "name": "Tumkur"}
        ]
    }
    
    if crop and crop in valid_combinations:
        return {"mandis": valid_combinations[crop]}
    
    # Return all mandis if no crop specified
    all_mandis = []
    for mandis in valid_combinations.values():
        all_mandis.extend(mandis)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_mandis = []
    for mandi in all_mandis:
        if mandi["id"] not in seen:
            seen.add(mandi["id"])
            unique_mandis.append(mandi)
    
    return {"mandis": unique_mandis}

@app.get("/api/v1/crop-mandi-combinations")
async def get_valid_combinations():
    """Get all valid crop-mandi combinations with trained models"""
    combinations = [
        # Arecanut combinations
        {"crop": "arecanut", "mandi": "sirsi", "crop_name": "Arecanut", "mandi_name": "Sirsi"},
        {"crop": "arecanut", "mandi": "yellapur", "crop_name": "Arecanut", "mandi_name": "Yellapur"},
        {"crop": "arecanut", "mandi": "siddapur", "crop_name": "Arecanut", "mandi_name": "Siddapur"},
        {"crop": "arecanut", "mandi": "shimoga", "crop_name": "Arecanut", "mandi_name": "Shimoga"},
        {"crop": "arecanut", "mandi": "sagar", "crop_name": "Arecanut", "mandi_name": "Sagar"},
        {"crop": "arecanut", "mandi": "kumta", "crop_name": "Arecanut", "mandi_name": "Kumta"},
        
        # Coconut combinations
        {"crop": "coconut", "mandi": "bangalore", "crop_name": "Coconut", "mandi_name": "Bangalore"},
        {"crop": "coconut", "mandi": "arasikere", "crop_name": "Coconut", "mandi_name": "Arasikere"},
        {"crop": "coconut", "mandi": "channarayapatna", "crop_name": "Coconut", "mandi_name": "Channarayapatna"},
        {"crop": "coconut", "mandi": "ramanagara", "crop_name": "Coconut", "mandi_name": "Ramanagara"},
        {"crop": "coconut", "mandi": "sira", "crop_name": "Coconut", "mandi_name": "Sira"},
        {"crop": "coconut", "mandi": "tumkur", "crop_name": "Coconut", "mandi_name": "Tumkur"}
    ]
    return {"combinations": combinations}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
