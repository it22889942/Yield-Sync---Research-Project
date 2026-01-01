# Model Evaluation Summary

## Multi-Horizon Forecasting
The system predicts prices at **5 different horizons**: 7, 14, 30, 60, and 84 days ahead.

## 1. Accuracy by Forecast Horizon

| Horizon | Description | MAE (LKR) | Use Case |
|---------|-------------|-----------|----------|
| **7 days** | 1 week ahead | 39.7 | Short-term decisions |
| **14 days** | 2 weeks ahead | 50.9 | Planning sales |
| **30 days** | 1 month ahead | 67.3 | General trends |
| **60 days** | 2 months ahead | 78.4 | Long-term planning |
| **84 days** | 3 months ahead | 89.1 | Seasonal patterns |

## 2. Day-by-Day Rolling Prediction Results

Each prediction uses **actual previous values** as input.

| Crop | Market | Model | MAE (LKR) | R² |
|------|--------|-------|-----------|-----|
| Rice | Colombo | LSTM | 7.6 | 0.342 |
| Beetroot | Nuwara Eliya | RandomForest | 7.7 | **0.984** |
| Radish | Kandy | RandomForest | ~8 | ~0.90 |
| Red Onion | Dambulla | LightGBM | 7.9 | 0.669 |
