import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def create_simple_features(df, date_col, price_col):
    """Create simple features from basic data (same as training)"""
    # Sort by date
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Create lag features
    df['modal_price_lag1'] = df[price_col].shift(1)
    df['modal_price_lag2'] = df[price_col].shift(2)
    df['modal_price_lag3'] = df[price_col].shift(3)
    df['modal_price_lag5'] = df[price_col].shift(5)
    df['modal_price_lag7'] = df[price_col].shift(7)
    
    # Create rolling statistics
    df['rolling_mean_7'] = df[price_col].rolling(window=7).mean()
    df['rolling_std_7'] = df[price_col].rolling(window=7).std()
    
    # Create temporal features
    df[date_col] = pd.to_datetime(df[date_col])
    df['day_of_year'] = df[date_col].dt.dayofyear
    df['month'] = df[date_col].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Drop rows with NaN values (due to lag features)
    df = df.dropna()
    
    return df

def evaluate_model_accuracy(crop, mandi):
    """Evaluate accuracy of the simple model"""
    print(f"\n{'='*60}")
    print(f"EVALUATING MODEL: {crop.upper()}-{mandi.upper()}")
    print(f"{'='*60}")
    
    # Check if model exists
    model_path = f"app/data/processed/simple_{crop}_{mandi}.joblib"
    scaler_path = f"app/data/processed/simple_scaler_{crop}_{mandi}.joblib"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"❌ Model files not found for {crop}-{mandi}")
        return None
    
    # Load model and scaler
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print(f"✅ Model and scaler loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Load and prepare data
    data_path = f"app/data/processed/{crop}_{mandi}_for_training.csv"
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    print(f"📊 Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Find date and price columns
    date_col = None
    price_col = None
    
    for col in df.columns:
        col_lower = col.strip().lower()
        if 'date' in col_lower or 'arrival' in col_lower:
            date_col = col
        elif 'modal' in col_lower and 'price' in col_lower:
            price_col = col
    
    if not date_col or not price_col:
        print(f"❌ Could not find date or price columns")
        return None
    
    print(f"📅 Date column: {date_col}")
    print(f"💰 Price column: {price_col}")
    
    # Create features
    df_features = create_simple_features(df, date_col, price_col)
    print(f"🔧 Features created: {df_features.shape[0]} rows after feature engineering")
    
    # Define feature columns
    feature_cols = [
        'modal_price_lag1', 'modal_price_lag2', 'modal_price_lag3',
        'modal_price_lag5', 'modal_price_lag7', 'rolling_mean_7',
        'rolling_std_7', 'day_of_year', 'month', 'month_sin', 'month_cos'
    ]
    
    # Prepare data
    X = df_features[feature_cols].values
    y = df_features[price_col].values
    
    # Split data (same as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    accuracy = max(0, r2 * 100)
    
    # Calculate additional metrics
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"\n📈 MODEL PERFORMANCE METRICS:")
    print(f"{'─'*40}")
    print(f"📊 R² Score:           {r2:.4f}")
    print(f"🎯 Accuracy:           {accuracy:.2f}%")
    print(f"📉 Mean Absolute Error: ₹{mae:.2f}")
    print(f"📉 Root Mean Sq Error:  ₹{rmse:.2f}")
    print(f"📉 Mean Abs Perc Error: {mape:.2f}%")
    print(f"{'─'*40}")
    
    # Price range analysis
    price_min = y_test.min()
    price_max = y_test.max()
    price_mean = y_test.mean()
    
    print(f"\n💰 PRICE ANALYSIS:")
    print(f"{'─'*40}")
    print(f"💵 Min Price:    ₹{price_min:.2f}")
    print(f"💵 Max Price:    ₹{price_max:.2f}")
    print(f"💵 Mean Price:   ₹{price_mean:.2f}")
    print(f"💵 Price Range:  ₹{price_max - price_min:.2f}")
    print(f"{'─'*40}")
    
    # Performance assessment
    if accuracy >= 85:
        status = "🟢 EXCELLENT"
    elif accuracy >= 70:
        status = "🟡 GOOD"
    elif accuracy >= 50:
        status = "🟠 FAIR"
    else:
        status = "🔴 POOR"
    
    print(f"\n🏆 OVERALL ASSESSMENT: {status}")
    
    return {
        'crop': crop,
        'mandi': mandi,
        'accuracy': accuracy,
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'data_points': len(y_test),
        'price_range': price_max - price_min,
        'mean_price': price_mean
    }

def main():
    """Evaluate all models and create summary"""
    print("🔍 CROP PRICE PREDICTION MODEL ACCURACY ANALYSIS")
    print("=" * 80)
    
    combinations = [
        ("arecanut", "sirsi"), ("arecanut", "yellapur"), ("arecanut", "siddapur"),
        ("arecanut", "shimoga"), ("arecanut", "sagar"), ("arecanut", "kumta"),
        ("coconut", "bangalore"), ("coconut", "arasikere"), ("coconut", "channarayapatna"),
        ("coconut", "ramanagara"), ("coconut", "sira"), ("coconut", "tumkur")
    ]
    
    results = []
    
    for crop, mandi in combinations:
        result = evaluate_model_accuracy(crop, mandi)
        if result:
            results.append(result)
    
    # Summary analysis
    if results:
        print(f"\n{'='*80}")
        print("📊 SUMMARY ANALYSIS")
        print(f"{'='*80}")
        
        # Create summary DataFrame
        df_results = pd.DataFrame(results)
        
        print(f"\n🎯 ACCURACY SUMMARY:")
        print(f"{'─'*50}")
        for _, row in df_results.iterrows():
            status = "🟢" if row['accuracy'] >= 85 else "🟡" if row['accuracy'] >= 70 else "🟠" if row['accuracy'] >= 50 else "🔴"
            print(f"{status} {row['crop']}-{row['mandi']:<15}: {row['accuracy']:6.2f}%")
        
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"{'─'*50}")
        print(f"🏆 Average Accuracy:    {df_results['accuracy'].mean():.2f}%")
        print(f"📊 Best Accuracy:       {df_results['accuracy'].max():.2f}% ({df_results.loc[df_results['accuracy'].idxmax(), 'crop']}-{df_results.loc[df_results['accuracy'].idxmax(), 'mandi']})")
        print(f"📉 Lowest Accuracy:     {df_results['accuracy'].min():.2f}% ({df_results.loc[df_results['accuracy'].idxmin(), 'crop']}-{df_results.loc[df_results['accuracy'].idxmin(), 'mandi']})")
        print(f"🎯 Models ≥85% Accuracy: {len(df_results[df_results['accuracy'] >= 85])}/{len(df_results)}")
        print(f"🎯 Models ≥70% Accuracy: {len(df_results[df_results['accuracy'] >= 70])}/{len(df_results)}")
        
        print(f"\n💰 PRICE PREDICTION ANALYSIS:")
        print(f"{'─'*50}")
        print(f"💵 Avg Mean Abs Error:  ₹{df_results['mae'].mean():.2f}")
        print(f"💵 Avg RMSE:            ₹{df_results['rmse'].mean():.2f}")
        print(f"💵 Avg MAPE:            {df_results['mape'].mean():.2f}%")
        
        # Save results
        df_results.to_csv('model_accuracy_results.csv', index=False)
        print(f"\n💾 Results saved to: model_accuracy_results.csv")
        
        print(f"\n{'='*80}")
        print("✅ ACCURACY ANALYSIS COMPLETE!")
        print(f"{'='*80}")
    
    else:
        print("❌ No models could be evaluated")

if __name__ == "__main__":
    main()
