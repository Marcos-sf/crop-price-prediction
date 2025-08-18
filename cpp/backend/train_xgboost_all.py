import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt

mandis = {
    'arecanut': ['Sirsi', 'Yellapur', 'Siddapur', 'Shimoga', 'Sagar', 'Kumta'],
    'coconut': ['Bangalore', 'Arasikere', 'Channarayapatna', 'Ramanagara', 'Sira', 'Tumkur']
}

all_y_test = []
all_y_pred = []

CUTOFF_DATE = np.datetime64('2023-01-01')

for crop, mandi_list in mandis.items():
    for mandi in mandi_list:
        features_path = f"app/data/processed/features_{crop}_{mandi.lower()}.npy"
        target_path = f"app/data/processed/target_{crop}_{mandi.lower()}.npy"
        dates_path = f"app/data/processed/dates_{crop}_{mandi.lower()}.npy"
        if not (os.path.exists(features_path) and os.path.exists(target_path) and os.path.exists(dates_path)):
            print(f"Missing data for {crop.title()} in {mandi}, skipping.")
            continue
        X = np.load(features_path)
        y = np.load(target_path)
        dates = np.load(dates_path)
        if len(X) < 10 or len(X) != len(y) or len(X) != len(dates):
            print(f"Data length mismatch for {crop.title()} in {mandi}, skipping.")
            continue
        # Out-of-sample split: train before cutoff, test from cutoff onwards
        mask_train = dates < CUTOFF_DATE
        mask_test = dates >= CUTOFF_DATE
        if mask_test.sum() < 5:
            print(f"Not enough test samples after {str(CUTOFF_DATE)} for {crop.title()} in {mandi}, skipping.")
            continue
        X_train, X_test = X[mask_train], X[mask_test]
        y_train, y_test = y[mask_train], y[mask_test]
        test_dates = dates[mask_test]
        # Check for sufficient training data
        if len(X_train) < 10:
            print(f"Not enough training samples before {str(CUTOFF_DATE)} for {crop.title()} in {mandi}, skipping.")
            continue
        tscv = TimeSeriesSplit(n_splits=3)
        if len(X_train) < tscv.get_n_splits():
            print(f"Not enough training samples for TimeSeriesSplit for {crop.title()} in {mandi}, skipping.")
            continue
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0]
        }
        xgb_model = xgb.XGBRegressor()
        search = RandomizedSearchCV(xgb_model, param_distributions=param_dist, n_iter=10, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42)
        search.fit(X_train, y_train)
        print(f"Best params for {crop.title()} in {mandi.title()}: {search.best_params_}")
        model = search.best_estimator_
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # Print test set size and date range
        print(f"Test set size: {len(y_test)}")
        print(f"Test set date range: {str(test_dates.min())[:10]} to {str(test_dates.max())[:10]}")
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        accuracy = 100 - mape
        print(f"RMSE for {crop.title()} in {mandi.title()}: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R²: {r2:.3f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"Accuracy: {accuracy:.2f}%")
        model_path = f"app/data/processed/xgb_{crop}_{mandi.lower()}.joblib"
        joblib.dump(model, model_path)
        print(f"Saved model to {model_path}\n")
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)
        # Print first 10 actual and predicted values for the test set
        print(f"First 10 actual values: {y_test[:10]}")
        print(f"First 10 predicted values: {y_pred[:10]}")
        # Naive baseline: predict last training value for all test samples
        if len(y_train) > 0:
            naive_pred = np.full_like(y_test, y_train[-1])
            naive_mape = np.mean(np.abs((y_test - naive_pred) / y_test)) * 100
            naive_acc = 100 - naive_mape
            print(f"Naive baseline (last train value) MAPE: {naive_mape:.2f}% | Accuracy: {naive_acc:.2f}%")
        else:
            print("Naive baseline not available (no training data)")

if all_y_test and all_y_pred:
    all_y_test = np.array(all_y_test)
    all_y_pred = np.array(all_y_pred)
    overall_rmse = mean_squared_error(all_y_test, all_y_pred, squared=False)
    overall_mae = mean_absolute_error(all_y_test, all_y_pred)
    overall_r2 = r2_score(all_y_test, all_y_pred)
    overall_mape = np.mean(np.abs((all_y_test - all_y_pred) / all_y_test)) * 100
    overall_accuracy = 100 - overall_mape
    print("\n=== OVERALL MODEL PERFORMANCE ===")
    print(f"Overall RMSE: {overall_rmse:.2f}")
    print(f"Overall MAE: {overall_mae:.2f}")
    print(f"Overall R²: {overall_r2:.3f}")
    print(f"Overall MAPE: {overall_mape:.2f}%")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%") 