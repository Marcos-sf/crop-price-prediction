#!/usr/bin/env python3
"""
Evaluate Current Model Accuracy
This script evaluates the current XGBoost models and generates updated accuracy statistics.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
import warnings
warnings.filterwarnings('ignore')

def evaluate_model_performance(crop, mandi):
    """Evaluate performance of a specific crop-mandi model"""
    try:
        # Load model
        xgb_path = f"app/data/processed/xgb_{crop}_{mandi}.joblib"
        if not os.path.exists(xgb_path):
            print(f"❌ Model not found: {xgb_path}")
            return None
            
        model = joblib.load(xgb_path)
        
        # Load data
        features_path = f"app/data/processed/features_{crop}_{mandi}.npy"
        target_path = f"app/data/processed/target_{crop}_{mandi}.npy"
        
        if not os.path.exists(features_path) or not os.path.exists(target_path):
            print(f"❌ Data not found for {crop}_{mandi}")
            return None
            
        features = np.load(features_path)
        targets = np.load(target_path)
        
        # Split data (80% train, 20% test)
        split_idx = int(len(features) * 0.8)
        X_train, X_test = features[:split_idx], features[split_idx:]
        y_train, y_test = targets[:split_idx], targets[split_idx:]
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Calculate accuracy within different thresholds
        accuracy_5 = np.mean(np.abs((y_test_pred - y_test) / y_test) <= 0.05) * 100
        accuracy_10 = np.mean(np.abs((y_test_pred - y_test) / y_test) <= 0.10) * 100
        accuracy_15 = np.mean(np.abs((y_test_pred - y_test) / y_test) <= 0.15) * 100
        accuracy_20 = np.mean(np.abs((y_test_pred - y_test) / y_test) <= 0.20) * 100
        
        # Calculate MAPE
        mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100
        
        return {
            'crop': crop,
            'mandi': mandi,
            'model_type': 'XGBoost',
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'accuracy_5': accuracy_5,
            'accuracy_10': accuracy_10,
            'accuracy_15': accuracy_15,
            'accuracy_20': accuracy_20,
            'mape': mape,
            'train_samples': len(y_train),
            'test_samples': len(y_test),
            'overfitting_score': train_r2 - test_r2
        }
        
    except Exception as e:
        print(f"❌ Error evaluating {crop}_{mandi}: {e}")
        return None

def main():
    """Main evaluation function"""
    print("🚀 Evaluating Current Model Performance")
    print("=" * 50)
    
    # Define crop-mandi combinations
    crop_mandi_combinations = [
        ('arecanut', 'sirsi'), ('arecanut', 'yellapur'), ('arecanut', 'siddapur'),
        ('arecanut', 'shimoga'), ('arecanut', 'sagar'), ('arecanut', 'kumta'),
        ('coconut', 'bangalore'), ('coconut', 'arasikere'), ('coconut', 'channarayapatna'),
        ('coconut', 'ramanagara'), ('coconut', 'sira'), ('coconut', 'tumkur')
    ]
    
    results = []
    
    for crop, mandi in crop_mandi_combinations:
        print(f"🔍 Evaluating {crop.title()} - {mandi.title()}...")
        result = evaluate_model_performance(crop, mandi)
        if result:
            results.append(result)
            print(f"   ✅ R²: {result['test_r2']:.4f}, Accuracy (10%): {result['accuracy_10']:.2f}%")
        else:
            print(f"   ❌ Failed to evaluate")
    
    if not results:
        print("❌ No models could be evaluated!")
        return
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Display summary statistics
    print("\n" + "=" * 50)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 50)
    
    print(f"Total Models Evaluated: {len(df)}")
    print(f"Average Test R² Score: {df['test_r2'].mean():.4f}")
    print(f"Average Test RMSE: {df['test_rmse'].mean():.2f}")
    print(f"Average Test MAE: {df['test_mae'].mean():.2f}")
    print(f"Average Accuracy (5%): {df['accuracy_5'].mean():.2f}%")
    print(f"Average Accuracy (10%): {df['accuracy_10'].mean():.2f}%")
    print(f"Average Accuracy (15%): {df['accuracy_15'].mean():.2f}%")
    print(f"Average Accuracy (20%): {df['accuracy_20'].mean():.2f}%")
    print(f"Average MAPE: {df['mape'].mean():.2f}%")
    
    # Performance by crop type
    print("\n" + "=" * 50)
    print("🌾 PERFORMANCE BY CROP TYPE")
    print("=" * 50)
    
    crop_summary = df.groupby('crop').agg({
        'test_r2': ['mean', 'std', 'min', 'max'],
        'accuracy_10': ['mean', 'std', 'min', 'max'],
        'test_rmse': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    print(crop_summary)
    
    # High-performing models
    print("\n" + "=" * 50)
    print("🏆 HIGH-PERFORMING MODELS (≥85% accuracy)")
    print("=" * 50)
    
    high_accuracy = df[df['accuracy_10'] >= 85]
    if len(high_accuracy) > 0:
        for _, model in high_accuracy.iterrows():
            print(f"✅ {model['crop'].title()} - {model['mandi'].title()}: "
                  f"Accuracy (10%) = {model['accuracy_10']:.2f}%, R² = {model['test_r2']:.4f}")
    else:
        print("❌ No models achieved ≥85% accuracy")
    
    # Models needing improvement
    print("\n" + "=" * 50)
    print("⚠️ MODELS NEEDING IMPROVEMENT (<80% accuracy)")
    print("=" * 50)
    
    low_accuracy = df[df['accuracy_10'] < 80]
    if len(low_accuracy) > 0:
        for _, model in low_accuracy.iterrows():
            print(f"⚠️ {model['crop'].title()} - {model['mandi'].title()}: "
                  f"Accuracy (10%) = {model['accuracy_10']:.2f}%, R² = {model['test_r2']:.4f}")
    else:
        print("✅ All models achieved ≥80% accuracy")
    
    # Save results
    output_file = 'current_model_evaluation_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    # Display detailed results
    print("\n" + "=" * 50)
    print("📋 DETAILED RESULTS")
    print("=" * 50)
    
    display_cols = ['crop', 'mandi', 'test_r2', 'accuracy_10', 'accuracy_15', 'test_rmse', 'mape']
    print(df[display_cols].round(4).to_string(index=False))
    
    return df

if __name__ == "__main__":
    results_df = main()
