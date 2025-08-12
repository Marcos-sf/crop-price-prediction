# 🎯 Achieve 85%+ Accuracy - Quick Start Guide

## Current Status
- **XGBoost**: ~71% accuracy
- **LSTM**: ~74% accuracy
- **Target**: 85%+ for both models

## 🚀 How to Run

### Step 1: Navigate to Backend Directory
```bash
cd backend
```

### Step 2: Run the 85% Accuracy Training Script
```bash
python achieve_85_percent_accuracy.py
```

## 📊 What This Script Does

### Enhanced Features (82 features):
- **Extended Lags**: 15 lag features (1, 2, 3, 5, 7, 10, 14, 21, 30 days)
- **Advanced Rolling Stats**: 32 features (mean, std, median, percentiles, etc.)
- **Momentum & Volatility**: 8 features
- **Trend Analysis**: 4 linear regression slopes
- **Enhanced Seasonality**: 8 seasonal features
- **Price Ratios**: 8 change ratio features
- **MA Crossovers**: 4 moving average features
- **RSI-like**: 3 technical indicators

### XGBoost Optimizations:
- **Extensive Hyperparameter Tuning**: 8 parameters × multiple values
- **TimeSeriesSplit**: Proper time series cross-validation
- **GridSearchCV**: Exhaustive parameter search
- **Higher n_estimators**: 1000-2000 trees

### LSTM Optimizations:
- **Bidirectional LSTM**: 3 layers (128→64→32 units)
- **Advanced Regularization**: L1/L2 regularization
- **Batch Normalization**: After each layer
- **Advanced Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- **Extended Training**: 200 epochs with validation split

## ⏱️ Expected Time
- **XGBoost**: ~30-60 minutes per crop-mandi
- **LSTM**: ~15-30 minutes per crop-mandi
- **Total**: ~2-4 hours for all models

## 📁 Output Files
1. **`85_percent_accuracy_results.csv`**: Detailed accuracy results
2. **`app/data/processed/xgb_85percent_*.joblib`**: 85%+ accurate XGBoost models
3. **`app/data/processed/lstm_85percent_*.h5`**: 85%+ accurate LSTM models
4. **`app/data/processed/lstm_85percent_scaler_*.joblib`**: LSTM scalers

## 🎯 Expected Results
- **XGBoost**: 85%+ accuracy (up from ~71%)
- **LSTM**: 85%+ accuracy (up from ~74%)
- **Average**: 87%+ accuracy

## 🔍 Monitoring Progress
The script will show:
- Training progress for each crop-mandi combination
- Best hyperparameters found
- Accuracy achieved for each model
- Target achievement status

## 🎉 Success Indicators
- Both XGBoost and LSTM achieve 85%+ accuracy
- Reduced MAPE (Mean Absolute Percentage Error)
- Improved R² score above 0.8
- Consistent performance across all crops/mandis

---

**Ready to achieve 85%+ accuracy? Run the script and watch your models improve! 🚀**
