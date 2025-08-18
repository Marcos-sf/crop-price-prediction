#!/usr/bin/env python3
"""
Real-Time Model Performance Monitoring
This script provides real-time monitoring of model accuracy and performance.
"""

import os
import time
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_enhanced_models(crop, mandi):
    """Load enhanced models for a specific crop-mandi combination"""
    try:
        # Load enhanced XGBoost
        xgb_path = f"app/data/processed/xgb_enhanced_{crop}_{mandi}.joblib"
        if os.path.exists(xgb_path):
            xgb_model = joblib.load(xgb_path)
            print(f"✅ Loaded enhanced XGBoost for {crop}_{mandi}")
        else:
            # Fallback to original model
            xgb_path = f"app/data/processed/xgb_{crop}_{mandi}.joblib"
            if os.path.exists(xgb_path):
                xgb_model = joblib.load(xgb_path)
                print(f"⚠️ Loaded original XGBoost for {crop}_{mandi}")
            else:
                print(f"❌ No XGBoost model found for {crop}_{mandi}")
                return None, None, None
        
        # Load enhanced LSTM
        lstm_path = f"app/data/processed/lstm_enhanced_{crop}_{mandi}.h5"
        if os.path.exists(lstm_path):
            from tensorflow.keras.models import load_model
            lstm_model = load_model(lstm_path)
            print(f"✅ Loaded enhanced LSTM for {crop}_{mandi}")
        else:
            print(f"⚠️ No enhanced LSTM found for {crop}_{mandi}")
            lstm_model = None
        
        # Load ensemble metadata
        ensemble_path = f"app/data/processed/ensemble_enhanced_{crop}_{mandi}.joblib"
        if os.path.exists(ensemble_path):
            ensemble_metadata = joblib.load(ensemble_path)
            print(f"✅ Loaded enhanced ensemble for {crop}_{mandi}")
        else:
            print(f"⚠️ No enhanced ensemble found for {crop}_{mandi}")
            ensemble_metadata = None
        
        return xgb_model, lstm_model, ensemble_metadata
        
    except Exception as e:
        print(f"❌ Error loading models for {crop}_{mandi}: {e}")
        return None, None, None

def get_latest_data(crop, mandi, days=7):
    """Get latest data for real-time evaluation"""
    try:
        features_path = f"app/data/processed/features_{crop}_{mandi}.npy"
        target_path = f"app/data/processed/target_{crop}_{mandi}.npy"
        dates_path = f"app/data/processed/dates_{crop}_{mandi}.npy"
        
        if not all(os.path.exists(path) for path in [features_path, target_path, dates_path]):
            return None, None, None
        
        features = np.load(features_path)
        targets = np.load(target_path)
        dates = np.load(dates_path)
        
        # Get latest N days
        if len(features) >= days:
            latest_features = features[-days:]
            latest_targets = targets[-days:]
            latest_dates = dates[-days:]
        else:
            latest_features = features
            latest_targets = targets
            latest_dates = dates
        
        return latest_features, latest_targets, latest_dates
        
    except Exception as e:
        print(f"❌ Error getting latest data for {crop}_{mandi}: {e}")
        return None, None, None

def evaluate_real_time_performance(crop, mandi):
    """Evaluate real-time performance of models"""
    try:
        # Load models
        xgb_model, lstm_model, ensemble_metadata = load_enhanced_models(crop, mandi)
        if xgb_model is None:
            return None
        
        # Get latest data
        features, targets, dates = get_latest_data(crop, mandi, days=30)
        if features is None:
            return None
        
        # Split for evaluation (80% train, 20% test)
        split_idx = int(len(features) * 0.8)
        X_test = features[split_idx:]
        y_test = targets[split_idx:]
        
        results = {
            'crop': crop,
            'mandi': mandi,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'test_samples': len(y_test)
        }
        
        # XGBoost evaluation
        if xgb_model is not None:
            try:
                y_pred_xgb = xgb_model.predict(X_test)
                xgb_r2 = np.corrcoef(y_test, y_pred_xgb)[0, 1] ** 2
                xgb_accuracy_10 = np.mean(np.abs((y_pred_xgb - y_test) / y_test) <= 0.1) * 100
                xgb_accuracy_15 = np.mean(np.abs((y_pred_xgb - y_test) / y_test) <= 0.15) * 100
                
                results.update({
                    'xgb_r2': xgb_r2,
                    'xgb_accuracy_10': xgb_accuracy_10,
                    'xgb_accuracy_15': xgb_accuracy_15,
                    'xgb_status': '✅ Active' if xgb_accuracy_10 >= 85 else '⚠️ Needs Improvement'
                })
            except Exception as e:
                results.update({
                    'xgb_r2': None,
                    'xgb_accuracy_10': None,
                    'xgb_accuracy_15': None,
                    'xgb_status': f'❌ Error: {str(e)[:50]}'
                })
        
        # LSTM evaluation
        if lstm_model is not None:
            try:
                X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
                y_pred_lstm = lstm_model.predict(X_test_lstm).flatten()
                lstm_r2 = np.corrcoef(y_test, y_pred_lstm)[0, 1] ** 2
                lstm_accuracy_10 = np.mean(np.abs((y_pred_lstm - y_test) / y_test) <= 0.1) * 100
                lstm_accuracy_15 = np.mean(np.abs((y_pred_lstm - y_test) / y_test) <= 0.15) * 100
                
                results.update({
                    'lstm_r2': lstm_r2,
                    'lstm_accuracy_10': lstm_accuracy_10,
                    'lstm_accuracy_15': lstm_accuracy_15,
                    'lstm_status': '✅ Active' if lstm_accuracy_10 >= 85 else '⚠️ Needs Improvement'
                })
            except Exception as e:
                results.update({
                    'lstm_r2': None,
                    'lstm_accuracy_10': None,
                    'lstm_accuracy_15': None,
                    'lstm_status': f'❌ Error: {str(e)[:50]}'
                })
        
        # Ensemble evaluation - use enhanced ensemble if available
        if ensemble_metadata is not None and 'weights' in ensemble_metadata:
            try:
                weights = ensemble_metadata['weights']
                if xgb_model is not None and lstm_model is not None:
                    # Use both models for ensemble
                    y_pred_ensemble = weights[0] * y_pred_xgb + weights[1] * y_pred_lstm
                elif xgb_model is not None:
                    # Use only XGBoost
                    y_pred_ensemble = y_pred_xgb
                else:
                    y_pred_ensemble = y_pred_lstm
                
                ensemble_r2 = np.corrcoef(y_test, y_pred_ensemble)[0, 1] ** 2
                ensemble_accuracy_10 = np.mean(np.abs((y_pred_ensemble - y_test) / y_test) <= 0.1) * 100
                ensemble_accuracy_15 = np.mean(np.abs((y_pred_ensemble - y_test) / y_test) <= 0.15) * 100
                
                results.update({
                    'ensemble_r2': ensemble_r2,
                    'ensemble_accuracy_10': ensemble_accuracy_10,
                    'ensemble_accuracy_15': ensemble_accuracy_15,
                    'ensemble_status': '✅ Active' if ensemble_accuracy_10 >= 85 else '⚠️ Needs Improvement'
                })
            except Exception as e:
                results.update({
                    'ensemble_r2': None,
                    'ensemble_accuracy_10': None,
                    'ensemble_accuracy_15': None,
                    'ensemble_status': f'❌ Error: {str(e)[:50]}'
                })
        elif xgb_model is not None:
            # Fallback: use XGBoost as ensemble if no ensemble metadata
            try:
                y_pred_ensemble = y_pred_xgb
                ensemble_r2 = np.corrcoef(y_test, y_pred_ensemble)[0, 1] ** 2
                ensemble_accuracy_10 = np.mean(np.abs((y_pred_ensemble - y_test) / y_test) <= 0.1) * 100
                ensemble_accuracy_15 = np.mean(np.abs((y_pred_ensemble - y_test) / y_test) <= 0.15) * 100
                
                results.update({
                    'ensemble_r2': ensemble_r2,
                    'ensemble_accuracy_10': ensemble_accuracy_10,
                    'ensemble_accuracy_15': ensemble_accuracy_15,
                    'ensemble_status': '✅ Active' if ensemble_accuracy_10 >= 85 else '⚠️ Needs Improvement'
                })
            except Exception as e:
                results.update({
                    'ensemble_r2': None,
                    'ensemble_accuracy_10': None,
                    'ensemble_accuracy_15': None,
                    'ensemble_status': f'❌ Error: {str(e)[:50]}'
                })
        
        return results
        
    except Exception as e:
        print(f"❌ Error in real-time evaluation for {crop}_{mandi}: {e}")
        return None

def display_real_time_dashboard(results_list):
    """Display real-time performance dashboard"""
    if not results_list:
        print("❌ No results to display")
        return
    
    df = pd.DataFrame(results_list)
    
    # Clear screen (works on most terminals)
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("🚀 REAL-TIME MODEL PERFORMANCE DASHBOARD")
    print("=" * 80)
    print(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Overall statistics
    total_models = len(df)
    xgb_active = len(df[df['xgb_status'] == '✅ Active'])
    lstm_active = len(df[df['lstm_status'] == '✅ Active'])
    ensemble_active = len(df[df['ensemble_status'] == '✅ Active'])
    
    print(f"📊 OVERALL STATUS:")
    print(f"   Total Models: {total_models}")
    print(f"   XGBoost ≥85%: {xgb_active}/{total_models} ({xgb_active/total_models*100:.1f}%)")
    print(f"   LSTM ≥85%: {lstm_active}/{total_models} ({lstm_active/total_models*100:.1f}%)")
    print(f"   Ensemble ≥85%: {ensemble_active}/{total_models} ({ensemble_active/total_models*100:.1f}%)")
    
    # Performance summary
    print(f"\n🎯 PERFORMANCE SUMMARY:")
    if 'xgb_accuracy_10' in df.columns:
        xgb_avg = df['xgb_accuracy_10'].dropna().mean()
        print(f"   Average XGBoost Accuracy (10%): {xgb_avg:.2f}%")
    
    if 'lstm_accuracy_10' in df.columns:
        lstm_avg = df['lstm_accuracy_10'].dropna().mean()
        print(f"   Average LSTM Accuracy (10%): {lstm_avg:.2f}%")
    
    if 'ensemble_accuracy_10' in df.columns:
        ensemble_avg = df['ensemble_accuracy_10'].dropna().mean()
        print(f"   Average Ensemble Accuracy (10%): {ensemble_avg:.2f}%")
    
    # Detailed results table
    print(f"\n📋 DETAILED RESULTS:")
    print("-" * 80)
    
    # Display columns based on available data
    display_cols = ['crop', 'mandi']
    if 'xgb_accuracy_10' in df.columns:
        display_cols.extend(['xgb_accuracy_10', 'xgb_status'])
    if 'lstm_accuracy_10' in df.columns:
        display_cols.extend(['lstm_accuracy_10', 'lstm_status'])
    if 'ensemble_accuracy_10' in df.columns:
        display_cols.extend(['ensemble_accuracy_10', 'ensemble_status'])
    
    # Format the display
    for _, row in df.iterrows():
        print(f"{row['crop'].title():<12} {row['mandi'].title():<15}", end="")
        
        if 'xgb_accuracy_10' in row and pd.notna(row['xgb_accuracy_10']):
            print(f"XGB: {row['xgb_accuracy_10']:6.1f}% {row['xgb_status']:<20}", end="")
        
        if 'lstm_accuracy_10' in row and pd.notna(row['lstm_accuracy_10']):
            print(f"LSTM: {row['lstm_accuracy_10']:6.1f}% {row['lstm_status']:<20}", end="")
        
        if 'ensemble_accuracy_10' in row and pd.notna(row['ensemble_accuracy_10']):
            print(f"ENS: {row['ensemble_accuracy_10']:6.1f}% {row['ensemble_status']:<20}", end="")
        
        print()  # New line
    
    print("-" * 80)
    
    # Models needing attention
    print(f"\n⚠️ MODELS NEEDING ATTENTION:")
    attention_models = []
    
    for _, row in df.iterrows():
        if ('xgb_status' in row and '⚠️' in str(row['xgb_status'])) or \
           ('lstm_status' in row and '⚠️' in str(row['lstm_status'])) or \
           ('ensemble_status' in row and '⚠️' in str(row['ensemble_status'])):
            attention_models.append(f"{row['crop'].title()} - {row['mandi'].title()}")
    
    if attention_models:
        for model in attention_models[:5]:  # Show first 5
            print(f"   • {model}")
        if len(attention_models) > 5:
            print(f"   ... and {len(attention_models) - 5} more")
    else:
        print("   ✅ All models performing well!")
    
    # Save results for historical tracking
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"real_time_results_{timestamp}.csv"
    df.to_csv(results_file, index=False)
    
    print(f"\n💾 Results saved to: {results_file}")

def continuous_monitoring(update_interval=60):
    """Continuous monitoring with specified update interval (in seconds)"""
    print("🔄 Starting continuous monitoring...")
    print(f"📡 Update interval: {update_interval} seconds")
    print("🛑 Press Ctrl+C to stop monitoring")
    print("=" * 60)
    
    try:
        while True:
            # Define crop-mandi combinations
            crop_mandi_combinations = [
                ('arecanut', 'sirsi'), ('arecanut', 'yellapur'), ('arecanut', 'siddapur'),
                ('arecanut', 'shimoga'), ('arecanut', 'sagar'), ('arecanut', 'kumta'),
                ('coconut', 'bangalore'), ('coconut', 'arasikere'), ('coconut', 'channarayapatna'),
                ('coconut', 'ramanagara'), ('coconut', 'sira'), ('coconut', 'tumkur')
            ]
            
            # Evaluate all models
            results = []
            for crop, mandi in crop_mandi_combinations:
                result = evaluate_real_time_performance(crop, mandi)
                if result:
                    results.append(result)
            
            # Display dashboard
            display_real_time_dashboard(results)
            
            # Wait for next update
            print(f"\n⏰ Next update in {update_interval} seconds...")
            time.sleep(update_interval)
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Monitoring error: {e}")

def main():
    """Main function"""
    print("🚀 REAL-TIME MODEL PERFORMANCE MONITORING")
    print("=" * 60)
    
    print("Choose monitoring mode:")
    print("1. Single evaluation")
    print("2. Continuous monitoring (updates every 60 seconds)")
    print("3. Custom interval monitoring")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        # Single evaluation
        crop_mandi_combinations = [
            ('arecanut', 'sirsi'), ('arecanut', 'yellapur'), ('arecanut', 'siddapur'),
            ('arecanut', 'shimoga'), ('arecanut', 'sagar'), ('arecanut', 'kumta'),
            ('coconut', 'bangalore'), ('coconut', 'arasikere'), ('coconut', 'channarayapatna'),
            ('coconut', 'ramanagara'), ('coconut', 'sira'), ('coconut', 'tumkur')
        ]
        
        results = []
        for crop, mandi in crop_mandi_combinations:
            result = evaluate_real_time_performance(crop, mandi)
            if result:
                results.append(result)
        
        display_real_time_dashboard(results)
        
    elif choice == "2":
        # Continuous monitoring with 60-second interval
        continuous_monitoring(60)
        
    elif choice == "3":
        # Custom interval
        try:
            interval = int(input("Enter update interval in seconds: "))
            continuous_monitoring(interval)
        except ValueError:
            print("❌ Invalid interval. Using default 60 seconds.")
            continuous_monitoring(60)
    
    else:
        print("❌ Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
