# Documentation Directory

## Overview

This directory contains additional documentation for the Yield-Sync project.

---

## Project Documentation

### Main Documentation

- **README.md** (root) - Project overview, installation, and usage
- **data/README.md** - Dataset descriptions and data dictionary
- **notebooks/README.md** - Notebook descriptions and execution guide
- **models/README.md** - Model files and loading instructions

---

## Research Background

### Problem Statement

Agricultural markets in Sri Lanka experience significant price volatility due to:

- Seasonal supply variations
- Weather impacts on production
- Festival-driven demand spikes
- Limited market information for farmers

### Solution Approach

This project applies machine learning to:

1. Forecast prices at multiple time horizons
2. Predict demand/volume patterns
3. Generate actionable sell/hold recommendations
4. Assess market risk

---

## Methodology

### Data Collection

- Historical price data from 15 markets (2020-2024)
- Weather data (temperature, rainfall, humidity)
- Seasonal indicators (cultivation periods, harvest seasons)
- Holiday calendar (public holidays, festivals)

### Feature Engineering

| Feature Type  | Examples                                             |
| ------------- | ---------------------------------------------------- |
| Lag Features  | price_lag_1, price_lag_7, price_lag_14, price_lag_28 |
| Rolling Stats | rolling_mean_7, rolling_std_7, rolling_mean_28       |
| Momentum      | price_momentum_7, price_momentum_28                  |
| Categorical   | market_encoded, item_encoded                         |

### Models Used

| Model          | Use Case                | Strengths                       |
| -------------- | ----------------------- | ------------------------------- |
| LightGBM       | Price/Volume prediction | Fast, handles tabular data well |
| LSTM           | Sequence modeling       | Captures temporal patterns      |
| Seasonal Naive | Baseline                | Simple benchmark                |

### Evaluation Metrics

- **R-squared (R2)**: Explained variance (higher is better)
- **MAE**: Mean Absolute Error in LKR
- **RMSE**: Root Mean Squared Error in LKR
- **MAPE**: Mean Absolute Percentage Error

---

## Findings

### Price Forecasting

1. LightGBM outperforms LSTM for this tabular dataset
2. Short-term forecasts (1-4 weeks) achieve R2 of 0.84-0.85
3. Accuracy decreases with longer horizons (as expected)
4. Lag features are the most important predictors

### Demand Patterns

1. Seasonal variations in volume by crop
2. Higher demand near major holidays
3. Price-demand relationship (elasticity)
4. Day-of-week effects on trading volume

### Decision Engine

1. Configurable thresholds for SELL/HOLD decisions
2. Confidence scores based on prediction certainty
3. Risk assessment using historical volatility
