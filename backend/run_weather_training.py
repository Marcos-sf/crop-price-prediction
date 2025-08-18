#!/usr/bin/env python3
"""
Weather-Enhanced Model Training Script
=====================================

This script trains models with weather features integrated for better prediction accuracy.

Usage:
    python run_weather_training.py

Features:
- Uses weather data (temperature, precipitation, wind, pressure)
- Creates weather-enhanced XGBoost and LSTM models
- Includes weather lags and rolling averages
- Provides fallback to regular models if weather data unavailable
"""

import os
import sys
import time
from weather_integrated_training import main as train_weather_models

def main():
    """Main function to run weather-enhanced training"""
    print("🌤️  Weather-Enhanced Crop Price Prediction Model Training")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("app/data/processed"):
        print("❌ Error: app/data/processed directory not found!")
        print("Please run this script from the backend directory.")
        sys.exit(1)
    
    # Check for weather data
    weather_files = [f for f in os.listdir("app/data/processed") if "_with_weather.csv" in f]
    if weather_files:
        print(f"✅ Found {len(weather_files)} weather-enhanced data files")
        for file in weather_files[:5]:  # Show first 5
            print(f"   - {file}")
        if len(weather_files) > 5:
            print(f"   ... and {len(weather_files) - 5} more")
    else:
        print("⚠️  No weather-enhanced data files found")
        print("   Models will be trained with regular data only")
    
    print("\n🚀 Starting weather-enhanced model training...")
    print("This may take several minutes depending on data size and model complexity.")
    
    start_time = time.time()
    
    try:
        # Run the training
        train_weather_models()
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"\n✅ Weather-enhanced training completed successfully!")
        print(f"⏱️  Total training time: {training_time:.2f} seconds")
        
        # Check for generated models
        weather_models = []
        for file in os.listdir("app/data/processed"):
            if "weather_" in file and (file.endswith(".joblib") or file.endswith(".h5")):
                weather_models.append(file)
        
        if weather_models:
            print(f"\n📊 Generated {len(weather_models)} weather-enhanced models:")
            for model in weather_models:
                print(f"   - {model}")
        else:
            print("\n⚠️  No weather-enhanced models were generated")
            print("   Check the training logs for any errors")
        
        print("\n🎯 Next Steps:")
        print("1. Test the weather-enhanced API endpoints:")
        print("   - GET /api/v1/weather-latest-features")
        print("   - POST /api/v1/weather-predict")
        print("   - POST /api/v1/weather-forecast")
        print("2. Compare accuracy with regular models")
        print("3. Update frontend to use weather features")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print("Check the logs above for more details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 