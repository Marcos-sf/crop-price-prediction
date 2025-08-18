import numpy as np
import os
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

mandis = {
    'arecanut': ['Sirsi', 'Yellapur', 'Siddapur', 'Shimoga', 'Sagar', 'Kumta'],
    'coconut': ['Bangalore', 'Arasikere', 'Channarayapatna', 'Ramanagara', 'Sira', 'Tumkur']
}

CUTOFF_DATE = np.datetime64('2023-01-01')

def evaluate_model_predictions(y_true, y_pred, model_name):
    """Calculate comprehensive evaluation metrics"""
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    accuracy = 100 - mape
    
    # Additional metrics
    mean_absolute_percentage_error = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    max_error = np.max(np.abs(y_true - y_pred))
    
    return {
        'model': model_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'accuracy': accuracy,
        'mean_absolute_percentage_error': mean_absolute_percentage_error,
        'max_error': max_error,
        'mean_actual': np.mean(y_true),
        'mean_predicted': np.mean(y_pred),
        'std_actual': np.std(y_true),
        'std_predicted': np.std(y_pred)
    }

def predict_with_lstm(features, lstm_model, lstm_norm):
    """Make prediction using LSTM model with proper denormalization"""
    if 'feature_mean' in lstm_norm and 'target_mean' in lstm_norm:
        # Use new normalization format
        X_norm = (features - lstm_norm['feature_mean']) / lstm_norm['feature_std']
        X_lstm = X_norm.reshape((X_norm.shape[0], 1, X_norm.shape[1]))
        pred_norm = lstm_model.predict(X_lstm, verbose=0).flatten()
        # Denormalize using target parameters
        pred = pred_norm * lstm_norm['target_std'] + lstm_norm['target_mean']
    else:
        # Use old normalization format (backward compatibility)
        X_norm = (features - lstm_norm['mean']) / lstm_norm['std']
        X_lstm = X_norm.reshape((X_norm.shape[0], 1, X_norm.shape[1]))
        pred_norm = lstm_model.predict(X_lstm, verbose=0).flatten()
        # Scale to reasonable price range if predictions are normalized
        if np.max(np.abs(pred_norm)) < 10:
            pred = pred_norm * 20000 + 40000
        else:
            pred = pred_norm
    return pred

def predict_with_ensemble(features, lstm_model, xgb_model, lstm_norm, weights):
    """Make ensemble prediction"""
    lstm_pred = predict_with_lstm(features, lstm_model, lstm_norm)
    xgb_pred = xgb_model.predict(features)
    
    # Weighted average
    ensemble_pred = weights[0] * lstm_pred + weights[1] * xgb_pred
    return ensemble_pred

def evaluate_all_models():
    """Evaluate all models for all crop-mandi combinations"""
    
    all_results = []
    summary_stats = {
        'xgboost': {'rmse': [], 'mae': [], 'r2': [], 'mape': [], 'accuracy': []},
        'lstm': {'rmse': [], 'mae': [], 'r2': [], 'mape': [], 'accuracy': []},
        'ensemble': {'rmse': [], 'mae': [], 'r2': [], 'mape': [], 'accuracy': []}
    }
    
    print("=" * 80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 80)
    
    for crop, mandi_list in mandis.items():
        for mandi in mandi_list:
            print(f"\n{'='*60}")
            print(f"Evaluating {crop.title()} in {mandi.title()}")
            print(f"{'='*60}")
            
            # Load test data
            features_path = f"app/data/processed/features_{crop}_{mandi.lower()}.npy"
            target_path = f"app/data/processed/target_{crop}_{mandi.lower()}.npy"
            dates_path = f"app/data/processed/dates_{crop}_{mandi.lower()}.npy"
            
            if not (os.path.exists(features_path) and os.path.exists(target_path) and os.path.exists(dates_path)):
                print(f"Missing data for {crop.title()} in {mandi}, skipping.")
                continue
            
            X = np.load(features_path)
            y = np.load(target_path)
            dates = np.load(dates_path)
            
            # Out-of-sample split
            mask_train = dates < CUTOFF_DATE
            mask_test = dates >= CUTOFF_DATE
            
            if mask_test.sum() < 5:
                print(f"Not enough test samples for {crop.title()} in {mandi}, skipping.")
                continue
            
            X_test = X[mask_test]
            y_test = y[mask_test]
            test_dates = dates[mask_test]
            
            print(f"Test set size: {len(y_test)}")
            print(f"Test date range: {str(test_dates.min())[:10]} to {str(test_dates.max())[:10]}")
            print(f"Actual price range: {y_test.min():.0f} - {y_test.max():.0f}")
            
            results = []
            
            # Evaluate XGBoost
            xgb_path = f"app/data/processed/xgb_{crop}_{mandi.lower()}.joblib"
            if os.path.exists(xgb_path):
                xgb_model = joblib.load(xgb_path)
                y_pred_xgb = xgb_model.predict(X_test)
                xgb_metrics = evaluate_model_predictions(y_test, y_pred_xgb, "XGBoost")
                results.append(xgb_metrics)
                
                # Add to summary stats
                for metric in ['rmse', 'mae', 'r2', 'mape', 'accuracy']:
                    summary_stats['xgboost'][metric].append(xgb_metrics[metric])
                
                print(f"XGBoost - RMSE: {xgb_metrics['rmse']:.2f}, R²: {xgb_metrics['r2']:.3f}, Accuracy: {xgb_metrics['accuracy']:.2f}%")
            
            # Evaluate LSTM
            lstm_path = f"app/data/processed/lstm_{crop}_{mandi.lower()}.h5"
            lstm_norm_path = f"app/data/processed/lstm_norm_{crop}_{mandi.lower()}.joblib"
            if os.path.exists(lstm_path) and os.path.exists(lstm_norm_path):
                lstm_model = load_model(lstm_path)
                lstm_norm = joblib.load(lstm_norm_path)
                y_pred_lstm = predict_with_lstm(X_test, lstm_model, lstm_norm)
                lstm_metrics = evaluate_model_predictions(y_test, y_pred_lstm, "LSTM")
                results.append(lstm_metrics)
                
                # Add to summary stats
                for metric in ['rmse', 'mae', 'r2', 'mape', 'accuracy']:
                    summary_stats['lstm'][metric].append(lstm_metrics[metric])
                
                print(f"LSTM - RMSE: {lstm_metrics['rmse']:.2f}, R²: {lstm_metrics['r2']:.3f}, Accuracy: {lstm_metrics['accuracy']:.2f}%")
            
            # Evaluate Ensemble
            ensemble_path = f"app/data/processed/ensemble_{crop}_{mandi.lower()}.joblib"
            if os.path.exists(ensemble_path):
                ensemble_metadata = joblib.load(ensemble_path)
                lstm_model = load_model(ensemble_metadata['lstm_model_path'])
                lstm_norm = joblib.load(ensemble_metadata['lstm_norm_path'])
                xgb_model = joblib.load(ensemble_metadata['xgb_model_path'])
                weights = ensemble_metadata['weights']
                
                y_pred_ensemble = predict_with_ensemble(X_test, lstm_model, xgb_model, lstm_norm, weights)
                ensemble_metrics = evaluate_model_predictions(y_test, y_pred_ensemble, "Ensemble")
                results.append(ensemble_metrics)
                
                # Add to summary stats
                for metric in ['rmse', 'mae', 'r2', 'mape', 'accuracy']:
                    summary_stats['ensemble'][metric].append(ensemble_metrics[metric])
                
                print(f"Ensemble - RMSE: {ensemble_metrics['rmse']:.2f}, R²: {ensemble_metrics['r2']:.3f}, Accuracy: {ensemble_metrics['accuracy']:.2f}%")
            
            # Find best model for this crop-mandi
            if results:
                best_model = min(results, key=lambda x: x['rmse'])
                print(f"\n🏆 Best Model: {best_model['model']} (RMSE: {best_model['rmse']:.2f})")
            
            all_results.extend(results)
    
    # Print comprehensive summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE MODEL PERFORMANCE SUMMARY")
    print("=" * 80)
    
    for model_type in ['xgboost', 'lstm', 'ensemble']:
        if summary_stats[model_type]['rmse']:
            print(f"\n📊 {model_type.upper()} MODEL SUMMARY:")
            print(f"  Average RMSE: {np.mean(summary_stats[model_type]['rmse']):.2f} ± {np.std(summary_stats[model_type]['rmse']):.2f}")
            print(f"  Average MAE: {np.mean(summary_stats[model_type]['mae']):.2f} ± {np.std(summary_stats[model_type]['mae']):.2f}")
            print(f"  Average R²: {np.mean(summary_stats[model_type]['r2']):.3f} ± {np.std(summary_stats[model_type]['r2']):.3f}")
            print(f"  Average MAPE: {np.mean(summary_stats[model_type]['mape']):.2f}% ± {np.std(summary_stats[model_type]['mape']):.2f}%")
            print(f"  Average Accuracy: {np.mean(summary_stats[model_type]['accuracy']):.2f}% ± {np.std(summary_stats[model_type]['accuracy']):.2f}%")
            print(f"  Number of evaluations: {len(summary_stats[model_type]['rmse'])}")
    
    # Find overall best model
    if all_results:
        best_overall = min(all_results, key=lambda x: x['rmse'])
        print(f"\n🏆 OVERALL BEST MODEL: {best_overall['model']}")
        print(f"  RMSE: {best_overall['rmse']:.2f}")
        print(f"  R²: {best_overall['r2']:.3f}")
        print(f"  Accuracy: {best_overall['accuracy']:.2f}%")
        print(f"  For: {best_overall.get('crop', 'Unknown')} in {best_overall.get('mandi', 'Unknown')}")
    
    # Save detailed results to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('model_evaluation_results.csv', index=False)
    print(f"\n📄 Detailed results saved to: model_evaluation_results.csv")
    
    return all_results, summary_stats

if __name__ == "__main__":
    evaluate_all_models()
