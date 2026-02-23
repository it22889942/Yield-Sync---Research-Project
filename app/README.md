# YieldSync - Deployment Package

**Sri Lankan Agricultural Price Forecasting System**

This package contains everything needed to run the price prediction system. A UI designer can use this folder directly to build a frontend application.

---

## Package Contents

```
deployment_package/
├── app.py             # Streamlit UI (run this to test)
├── api.py             # Simple API wrapper for UI integration
├── config.py          # Configuration (crops, markets, horizons)
├── predictor.py       # Inference module (predictions)
├── trainer.py         # Training module (model retraining)
├── data_fetcher.py    # Weekly HARTI + weather data updates
├── requirements.txt   # Python dependencies
├── README.md          # This file
├── data/              # Data files
│   └── full_history_features_real_weather.csv
└── models/
    └── saved_models/
        └── price forcasting/   # Pre-trained price prediction models
```

## Start Guide

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit App

```bash
streamlit run app.py
```

This will open the working app at http://localhost:8501

---

## Supported Crops & Markets

| Crop                | Markets                                                                             | Model Type   |
| ------------------- | ----------------------------------------------------------------------------------- | ------------ |
| **Rice**      | Colombo, Anuradhapura, Moneragala, Dambulla, Ampara, Kandy, Kurunegala, Polonnaruwa | LSTM         |
| **Beetroot**  | Colombo, Thambuththegama, Bandarawela, Dambulla, Kandy, Nuwara Eliya                | RandomForest |
| **Radish**    | Colombo, Moneragala, Dambulla, Kandy, Meegoda                                       | RandomForest |
| **Red Onion** | Colombo, Puttalam, Mullaittivu, Vavuniya, Batticaloa, Dambulla, + 9 more            | LightGBM     |

---

## 📅 Forecast Horizons

- **7 days** (1 week)
- **14 days** (2 weeks)
- **30 days** (1 month)

> Note: Longer horizons (60/84 days) were removed due to poor accuracy (negative R²).

---

## API Reference

### YieldSyncPredictor Class

```python
from predictor import YieldSyncPredictor

predictor = YieldSyncPredictor(model_base_dir='path/to/models/saved_models')
```

#### `predict_price(data, crop, days_ahead, market)`

Predict future price for a crop.

**Parameters:**

- `data` (DataFrame): Historical data with columns `['Date', 'item', 'price', ...]`
- `crop` (str): One of `['Rice', 'Beetroot', 'Radish', 'Red Onion']`
- `days_ahead` (int): Forecast horizon (7, 14, or 30)
- `market` (str, optional): Market name for location-specific prediction

**Returns:**

```python
{
    'crop': 'Rice',
    'market': 'Colombo',
    'current_price': 145.00,
    'predicted_price': 152.50,
    'price_change_percent': 5.17,
    'confidence_interval': {'lower': 122.10, 'upper': 182.90},
    'days_ahead': 7,
    'horizon_used': 7,
    'model_type': 'LSTM'
}
```

#### `predict_demand(data, crop, days_ahead)`

Predict future demand for a crop.

**Returns:**

```python
{
    'crop': 'Rice',
    'current_demand': 1250.5,
    'predicted_demand': 1305.2,
    'demand_change_percent': 4.37,
    'days_ahead': 7,
    'horizon_used': 7
}
```

#### `get_recommendation(crop, current_price, predicted_price, ...)`

Get buy/sell recommendation based on price prediction.

**Returns:**

```python
{
    'decision': 'HOLD',
    'reasoning': 'Moderate profit opportunity: +5.17%',
    'profit_analysis': {
        'revenue_if_sell_now': 144995.00,
        'revenue_if_hold': 152445.00,
        'profit_difference': 7450.00,
        'profit_change_percent': 5.14,
        'spoilage_loss_percent': 0.0,
        'storage_cost_total': 50.00
    },
    'confidence': 0.75
}
```

---

## Data Format

### Price Data (`full_history_features_real_weather.csv`)

| Column         | Type     | Description                      |
| -------------- | -------- | -------------------------------- |
| Date           | datetime | Record date                      |
| item           | string   | Crop name (Rice, Beetroot, etc.) |
| market         | string   | Market name                      |
| price          | float    | Price per kg (LKR)               |
| temp           | float    | Temperature (°C)                |
| rainfall       | float    | Rainfall (mm)                    |
| humidity       | float    | Humidity (%)                     |
| wind_speed     | float    | Wind speed (km/h)                |
| sunshine_hours | float    | Sunshine hours                   |

### Demand Data (`full_history_demand_data.csv`)

| Column          | Type     | Description        |
| --------------- | -------- | ------------------ |
| Date            | datetime | Record date        |
| item            | string   | Crop name          |
| market          | string   | Market name        |
| quantity_tonnes | float    | Quantity in tonnes |
| price           | float    | Price per kg (LKR) |

---

## Model Configuration

Edit `config.py` to customize:

```python
# Change forecast horizons
FORECAST_HORIZONS = [7, 14, 30]

# Add/modify crop-market mappings
CROP_MARKETS = {
    'Rice': ['Colombo', 'Kandy', ...],
    ...
}

# Adjust model RMSE for confidence intervals
MODEL_RMSE = {
    'Rice': 15.5,
    'Beetroot': 22.3,
    ...
}
```

---

## Streamlit Example

```python
import streamlit as st
from predictor import YieldSyncPredictor
import pandas as pd

@st.cache_resource
def load_predictor():
    return YieldSyncPredictor()

st.title("YieldSync Price Forecast")

predictor = load_predictor()

crop = st.selectbox("Select Crop", predictor.get_available_crops())
market = st.selectbox("Select Market", predictor.get_available_markets(crop))
days = st.slider("Forecast Days", 7, 30, 7)

if st.button("Predict"):
    df = pd.read_csv('data/full_history_features_real_weather.csv')
    df['Date'] = pd.to_datetime(df['Date'])
  
    result = predictor.predict_price(df, crop, days, market)
  
    st.metric("Current Price", f"LKR {result['current_price']:.2f}")
    st.metric("Predicted Price", f"LKR {result['predicted_price']:.2f}", 
              f"{result['price_change_percent']:+.1f}%")
```

---

## Model Performance

| Crop      | Horizon | MAE (LKR) | R² Score |
| --------- | ------- | --------- | --------- |
| Rice      | 7 days  | 12.5      | 0.85      |
| Rice      | 14 days | 15.2      | 0.78      |
| Rice      | 30 days | 18.7      | 0.65      |
| Beetroot  | 7 days  | 18.3      | 0.82      |
| Beetroot  | 14 days | 22.1      | 0.74      |
| Radish    | 7 days  | 10.5      | 0.88      |
| Red Onion | 7 days  | 35.2      | 0.79      |

---
