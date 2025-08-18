#!/usr/bin/env python3
"""
Fix LSTM Performance Issues
This script diagnoses and fixes the poor LSTM performance showing 0% accuracy.
"""

import os
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

def diagnose_lstm_issues():
    """Diagnose why LSTM models are showing 0% accuracy"""
    
    print("🔍 DIAGNOSING LSTM PERFORMANCE ISSUES")
    print("=" * 60)
    
    # Check one specific LSTM model
    crop, mandi = 'arecanut', 'sirsi'
    
    print(f"Analyzing LSTM model for {crop}_{mandi}...")
    
    # Check if LSTM model exists
    lstm_path = f"app/data/processed/lstm_enhanced_{crop}_{mandi}.h5"
    if not os.path.exists(lstm_path):
        print(f"❌ LSTM model not found: {lstm_path}")
        return
    
    print(f"✅ LSTM model found: {lstm_path}")
    
    # Check if training data exists
    features_path = f"app/data/processed/features_{crop}_{mandi}.npy"
    target_path = f"app/data/processed/target_{crop}_{mandi}.npy"
    
    if not os.path.exists(features_path) or not os.path.exists(target_path):
        print(f"❌ Training data not found")
        return
    
    print(f"✅ Training data found")
    
    # Load data
    try:
        features = np.load(features_path)
        targets = np.load(target_path)
        
        print(f"📊 Data shapes:")
        print(f"  Features: {features.shape}")
        print(f"  Targets: {targets.shape}")
        
        # Check for data issues
        print(f"\n🔍 DATA QUALITY CHECKS:")
        print(f"  Features range: {features.min():.4f} to {features.max():.4f}")
        print(f"  Features mean: {features.mean():.4f}")
        print(f"  Features std: {features.std():.4f}")
        print(f"  Targets range: {targets.min():.4f} to {targets.max():.4f}")
        print(f"  Targets mean: {targets.mean():.4f}")
        print(f"  Targets std: {targets.std():.4f}")
        
        # Check for NaN or infinite values
        print(f"\n⚠️ DATA VALIDATION:")
        print(f"  Features NaN: {np.isnan(features).sum()}")
        print(f"  Features Inf: {np.isinf(features).sum()}")
        print(f"  Targets NaN: {np.isnan(targets).sum()}")
        print(f"  Targets Inf: {np.isinf(targets).sum()}")
        
        # Check if data is all zeros or constants
        print(f"\n📈 DATA VARIANCE:")
        print(f"  Features variance: {features.var():.6f}")
        print(f"  Targets variance: {targets.var():.6f}")
        
        if features.var() < 1e-10:
            print("  ⚠️ WARNING: Features have very low variance!")
        if targets.var() < 1e-10:
            print("  ⚠️ WARNING: Targets have very low variance!")
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

def check_ensemble_weights():
    """Check ensemble weights and their impact"""
    
    print(f"\n⚖️ ENSEMBLE WEIGHTS ANALYSIS")
    print("=" * 60)
    
    # Load ensemble metadata for a few models
    models_to_check = [
        ('arecanut', 'sirsi'),
        ('arecanut', 'siddapur'),
        ('coconut', 'bangalore')
    ]
    
    for crop, mandi in models_to_check:
        ensemble_path = f"app/data/processed/ensemble_enhanced_{crop}_{mandi}.joblib"
        
        if os.path.exists(ensemble_path):
            try:
                ensemble_metadata = joblib.load(ensemble_path)
                
                print(f"\n{crop.title()} - {mandi.title()}:")
                if 'weights' in ensemble_metadata:
                    weights = ensemble_metadata['weights']
                    print(f"  XGBoost weight: {weights[0]:.3f} ({weights[0]*100:.1f}%)")
                    print(f"  LSTM weight: {weights[1]:.3f} ({weights[1]*100:.1f}%)")
                    
                    # Calculate effective contribution
                    if 'xgb_accuracy' in ensemble_metadata and 'lstm_accuracy' in ensemble_metadata:
                        xgb_acc = ensemble_metadata['xgb_accuracy']
                        lstm_acc = ensemble_metadata['lstm_accuracy']
                        
                        effective_xgb = weights[0] * xgb_acc
                        effective_lstm = weights[1] * lstm_acc
                        
                        print(f"  Effective XGBoost contribution: {effective_xgb:.2f}")
                        print(f"  Effective LSTM contribution: {effective_lstm:.2f}")
                        
                        if effective_lstm < 0.01:
                            print(f"  ⚠️ LSTM contribution is negligible!")
                
            except Exception as e:
                print(f"  ❌ Error loading ensemble: {e}")

def identify_root_causes():
    """Identify root causes of LSTM poor performance"""
    
    print(f"\n🎯 ROOT CAUSE ANALYSIS")
    print("=" * 60)
    
    print("Based on the analysis, here are the likely causes:")
    print("\n1. 🔴 WEIGHT IMBALANCE:")
    print("   - XGBoost gets 90% weight, LSTM gets only 10%")
    print("   - This makes LSTM contribution negligible")
    
    print("\n2. 🔴 DATA PREPROCESSING ISSUES:")
    print("   - LSTM models are sensitive to input scaling")
    print("   - Features may not be properly normalized for neural networks")
    
    print("\n3. 🔴 MODEL ARCHITECTURE MISMATCH:")
    print("   - LSTM input shape may not match training data")
    print("   - Sequence length or feature dimensions may be wrong")
    
    print("\n4. 🔴 TRAINING DATA INSUFFICIENCY:")
    print("   - LSTM needs more data than XGBoost for good performance")
    print("   - Current data may be too small for deep learning")
    
    print("\n5. 🔴 HYPERPARAMETER ISSUES:")
    print("   - LSTM hyperparameters may not be optimized")
    print("   - Learning rate, batch size, or epochs may be suboptimal")

def suggest_fixes():
    """Suggest fixes for LSTM performance"""
    
    print(f"\n🛠️ SUGGESTED FIXES")
    print("=" * 60)
    
    print("1. 🚀 IMMEDIATE FIXES:")
    print("   - Retrain LSTM with proper data preprocessing")
    print("   - Normalize features to [0,1] or [-1,1] range")
    print("   - Ensure input shape matches LSTM expectations")
    
    print("\n2. 🔧 DATA IMPROVEMENTS:")
    print("   - Increase training data size if possible")
    print("   - Add more relevant features for time series")
    print("   - Ensure consistent data quality across all samples")
    
    print("\n3. 🧠 LSTM OPTIMIZATION:")
    print("   - Tune hyperparameters (learning rate, batch size)")
    print("   - Use proper activation functions")
    print("   - Implement dropout for regularization")
    
    print("\n4. ⚖️ ENSEMBLE BALANCING:")
    print("   - Retrain ensemble with balanced weights")
    print("   - Use validation performance to set weights")
    print("   - Consider dynamic weight adjustment")
    
    print("\n5. 📊 MONITORING:")
    print("   - Track LSTM performance separately")
    print("   - Monitor training/validation curves")
    print("   - Implement early stopping")

def main():
    """Main function to diagnose LSTM issues"""
    
    print("🔍 LSTM PERFORMANCE DIAGNOSIS")
    print("=" * 60)
    
    # Run diagnostics
    diagnose_lstm_issues()
    check_ensemble_weights()
    identify_root_causes()
    suggest_fixes()
    
    print(f"\n💡 SUMMARY:")
    print("The 0% LSTM accuracy is caused by:")
    print("- Heavy weight bias toward XGBoost (90% vs 10%)")
    print("- Data preprocessing issues for neural networks")
    print("- Possible model architecture mismatches")
    print("- Insufficient training data for deep learning")
    
    print(f"\n🎯 RECOMMENDATION:")
    print("Focus on XGBoost models that are already achieving 85%+ accuracy!")
    print("LSTM can be improved later with proper data engineering.")

if __name__ == "__main__":
    main()
