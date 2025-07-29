# Crop Price Prediction - Karnataka, India

A full-stack web application for predicting future crop prices across major mandis in Karnataka, India. The project uses machine learning (XGBoost) to forecast crop prices for the next 12 months with real-time data integration.

## 🌾 Features

- **Multi-Crop Support**: Predict prices for Arecanut, Coconut, and other major Karnataka crops
- **Multi-Mandi Analysis**: Compare prices across 8+ major mandis in Karnataka
- **12-Month Forecasting**: Predict future prices with monthly granularity
- **Interactive Dashboard**: Real-time charts and tables with toggle views
- **Detailed Analytics**: Historical trend analysis and seasonality patterns
- **Export Functionality**: Download predictions in CSV format
- **Real-time Data**: Integration with Agmarknet and weather APIs

## 🛠️ Tech Stack

### Frontend
- **React.js** - Main frontend framework
- **Material-UI (MUI)** - UI component library
- **Recharts** - Data visualization
- **React Router** - Navigation

### Backend
- **FastAPI** - Python web framework
- **Pydantic** - Data validation
- **Uvicorn** - ASGI server

### Machine Learning
- **XGBoost** - Gradient boosting for price prediction
- **Pandas** - Data manipulation
- **Scikit-learn** - ML utilities
- **NumPy** - Numerical computing

### Data Sources
- **Agmarknet** - Historical crop price data
- **Meteostat** - Weather data integration

## 📁 Project Structure

```
crop-price-prediction/
├── frontend/                 # React frontend
│   ├── src/
│   │   ├── pages/           # React components
│   │   ├── components/      # Reusable components
│   │   └── assets/          # Static assets
│   └── package.json
├── backend/                  # FastAPI backend
│   ├── app/
│   │   ├── api/            # API routes
│   │   ├── utils/          # Utility functions
│   │   └── main.py         # FastAPI app
│   ├── data/               # Data files
│   └── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 14+
- Git

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

### Data Processing
```bash
cd backend
python process_data.py
python prepare_all_training_data.py
python train_xgboost_all.py
```

## 📊 API Endpoints

- `GET /api/v1/crops` - List available crops
- `GET /api/v1/mandis` - List available mandis
- `GET /api/v1/history` - Get historical price data
- `GET /api/v1/latest-features` - Get latest computed features
- `POST /api/v1/forecast` - Generate 12-month price predictions

## 🎯 Model Performance

- **Overall Accuracy**: 92.73%
- **Evaluation Metrics**: RMSE, MAE, R², MAPE
- **True Out-of-Sample Testing**: Train on pre-2023 data, test on 2023+ data

## 📈 Features Used

- Price lags (1, 2, 3, 5, 7 days)
- Rolling statistics (7-day mean, standard deviation)
- Time-based features (day of year, month, seasonality)
- Weather data integration (planned)

## 🔧 Configuration

The application supports multiple crops and mandis. To add new crops:

1. Add crop data to `backend/data/raw/`
2. Update crop lists in frontend components
3. Retrain models with new data

## 📝 License

This project is for educational and research purposes.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

For questions or issues, please open an issue on GitHub.

---

**Note**: This project is specifically designed for Karnataka, India crop markets and uses real agricultural data from Agmarknet. 