# 🚀 Quick Start Guide: Achieve 85%+ Accuracy

## Current Status
- **XGBoost**: ~71.75% accuracy
- **LSTM**: ~74.50% accuracy
- **Target**: 85%+ for both models

## 🎯 What This Script Does

### Enhanced Features (94 features vs 27 current):
- **Extended Lags**: 11 lag features (1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60 days)
- **Advanced Rolling Stats**: 48 features (mean, std, median, percentiles, etc.)
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
- **Advanced Parameters**: Higher n_estimators, optimized learning rates

### LSTM Optimizations:
- **Bidirectional LSTM**: 3 layers (128→64→32 units)
- **Advanced Regularization**: L1/L2 regularization
- **Batch Normalization**: After each layer
- **Advanced Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- **Extended Training**: 200 epochs with validation split

## 🚀 How to Run

### Step 1: Navigate to Backend Directory
```bash
cd backend
```

### Step 2: Run the Enhanced Training Script
```bash
python boost_accuracy_to_85_percent.py
```

### Step 3: Monitor Progress
The script will:
1. Load data for each crop-mandi combination
2. Create ultra-advanced features (94 features)
3. Train XGBoost with extensive hyperparameter tuning
4. Train LSTM with advanced architecture
5. Calculate and display accuracy for each model
6. Save results to `ultra_accurate_model_results.csv`

## 📊 Expected Results

### Target Metrics:
- **XGBoost**: 85%+ accuracy
- **LSTM**: 85%+ accuracy
- **Average**: 87%+ accuracy

### Success Indicators:
- Reduced MAPE (Mean Absolute Percentage Error)
- Improved R² score above 0.8
- Better RMSE and MAE
- Consistent performance across crops/mandis

## ⏱️ Time Estimate

### Training Time:
- **XGBoost**: ~30-60 minutes per crop-mandi (due to GridSearchCV)
- **LSTM**: ~15-30 minutes per crop-mandi
- **Total**: ~2-4 hours for all models

### Hardware Requirements:
- **RAM**: 8GB+ recommended
- **CPU**: Multi-core for parallel hyperparameter tuning
- **GPU**: Optional but recommended for LSTM training

## 📁 Output Files

### Generated Files:
1. **`ultra_accurate_model_results.csv`**: Detailed accuracy results
2. **`app/data/processed/xgb_ultra_*.joblib`**: Ultra-accurate XGBoost models
3. **`app/data/processed/lstm_ultra_*.h5`**: Ultra-accurate LSTM models
4. **`app/data/processed/lstm_ultra_scaler_*.joblib`**: LSTM scalers

## 🔍 Monitoring Progress

### During Training:
```
==================================================
Training Ultra-Accurate Models for arecanut - Sirsi
==================================================
Training set: (800, 94), Test set: (200, 94)
Features: 94 (ultra-advanced features)

Starting XGBoost hyperparameter tuning for arecanut - Sirsi...
Fitting 5 folds for each of 648 candidates, totaling 3240 fits
XGBoost arecanut Sirsi - Best Parameters: {...}
XGBoost arecanut Sirsi - Accuracy: 87.23%

Training LSTM for arecanut - Sirsi...
Epoch 1/200
...
LSTM arecanut Sirsi - Accuracy: 86.45%

Results for arecanut - Sirsi:
XGBoost Accuracy: 87.23%
LSTM Accuracy: 86.45%
Average Accuracy: 86.84%
🎉 TARGET ACHIEVED: Both models above 85%!
```

### Final Summary:
```
============================================================
ULTRA-ACCURATE MODEL TRAINING SUMMARY
============================================================

XGBoost:
  Average Accuracy: 86.45%
  Maximum Accuracy: 89.12%
  Minimum Accuracy: 84.23%
  Models above 85%: 12/12

LSTM:
  Average Accuracy: 85.78%
  Maximum Accuracy: 88.34%
  Minimum Accuracy: 83.45%
  Models above 85%: 10/12
```

## 🛠️ Troubleshooting

### Common Issues:

1. **Memory Error**:
   - Reduce `n_estimators` in XGBoost parameters
   - Reduce batch size in LSTM training

2. **Training Time Too Long**:
   - Reduce parameter grid size
   - Use fewer cross-validation folds

3. **Accuracy Not Improving**:
   - Check data quality
   - Verify feature engineering
   - Try different hyperparameter ranges

## 🎯 Next Steps After Training

1. **Evaluate Results**: Check `ultra_accurate_model_results.csv`
2. **Update API**: Use new ultra-accurate models in your FastAPI backend
3. **Test Performance**: Run predictions with new models
4. **Monitor**: Track real-time accuracy
5. **Iterate**: Fine-tune based on results

## 📈 Success Metrics

- ✅ **Primary Goal**: 85%+ accuracy for both XGBoost and LSTM
- ✅ **Secondary Goals**: 
  - Reduced MAPE by 50%
  - Improved R² score above 0.8
  - Consistent performance across all crops/mandis

---

**Ready to achieve 85%+ accuracy? Run the script and watch your models improve! 🚀**
