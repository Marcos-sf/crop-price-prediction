from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import routers
from app.api.routes import prediction, analysis, comparison

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

@app.get("/")
async def root():
    return {"message": "Crop Price Prediction API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "crop-price-prediction-api"}

@app.get("/api/v1/crops")
async def get_crops():
    """Get list of available crops"""
    crops = [
        {"id": "arecanut", "name": "Arecanut"},
        {"id": "elaichi", "name": "Elaichi (Cardamom)"},
        {"id": "pepper", "name": "Pepper"},
        {"id": "coconut", "name": "Coconut"}
    ]
    return {"crops": crops}

@app.get("/api/v1/mandis")
async def get_mandis():
    """Get list of available mandis"""
    mandis = [
        {"id": "sirsi", "name": "Sirsi"},
        {"id": "shimoga", "name": "Shimoga"},
        {"id": "chikmagalur", "name": "Chikmagalur"},
        {"id": "madikeri", "name": "Madikeri"},
        {"id": "tiptur", "name": "Tiptur"},
        {"id": "hassan", "name": "Hassan"},
        {"id": "sullia", "name": "Sullia"},
        {"id": "tumkur", "name": "Tumkur"}
    ]
    return {"mandis": mandis}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
