
# Yield-Sync Research Project

## Crop Price Prediction and Demand Forecasting System

A comprehensive machine learning system for agricultural market intelligence in Sri Lanka. This project provides price predictions, demand forecasting, and actionable SELL/HOLD recommendations for farmers and traders.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Dataset](#dataset)
5. [Models](#models)
6. [Installation](#installation)
7. [Usage](#usage)

---

## Project Overview

Yield-Sync is a research project that combines historical market data, weather information, and seasonal patterns to forecast agricultural commodity prices and demand. The system helps stakeholders make informed decisions about when to sell crops for maximum profit.

### Objectives

- Predict crop prices at multiple time horizons (1, 2, 4, 8, 12 weeks ahead)
- Forecast demand/volume for supply chain planning
- Generate actionable SELL/HOLD/WAIT recommendations
- Assess market risk and uncertainty

### Crops Covered

- Rice
- Beetroot
- Raddish
- Red Onion

### Markets

15 major agricultural markets across Sri Lanka

---

## Features

### 1. Multi-Horizon Price Forecasting

- Predictions for 1 week to 12 weeks ahead
- LightGBM and LSTM models
- R-squared scores up to 0.85 for short-term forecasts

### 2. Demand Forecasting

- Volume prediction in Metric Tons (MT)
- Price-demand elasticity analysis
- Seasonal demand patterns

### 3. Decision and Alert Engine

- SELL/HOLD/WAIT recommendations with confidence scores
- Best selling time analysis
- Risk assessment for each crop
- Automated alerts for significant price movements

---

## Project Structure

```
Yield-Sync---Research-Project/
│
├── data/                          # Dataset files
│   ├── 2020_2024.csv             # Extended base dataset (5 years)
│   ├── weather_data.csv          # Weather information
│   ├── seasonal_indicators.csv   # Cultivation seasons
│   ├── festival_holidays.csv     # Sri Lankan holidays
│   ├── model_ready_data.csv      # Preprocessed dataset
│   └── README.md                 # Data documentation
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_EDA_Crop_Price_Demand.ipynb
│   ├── 02_Price_Prediction_Model.ipynb
│   ├── 03_Multi_Horizon_Forecasting.ipynb
│   ├── 04_Demand_Forecasting.ipynb
│   └── 05_Decision_Alert_Engine.ipynb
│
├── models/                        # Saved models
│   └── saved_models/
│       ├── multi_horizon/        # Price forecasting models
│       ├── demand_forecasting/   # Volume prediction models
│       └── decision_engine/      # Decision outputs
│
├── app/                          # Application code (future)
├── docs/                         # Documentation
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Dataset

### Source Data

- **Period**: January 1, 2020 to December 31, 2024 (5 years)
- **Records**: 109,620 daily observations
- **Granularity**: Daily prices per market per crop

### Features

| Category   | Features                                              |
| ---------- | ----------------------------------------------------- |
| Price Data | Daily prices (LKR), Volume (MT)                       |
| Weather    | Temperature, Rainfall, Humidity, Wind Speed           |
| Seasonal   | Maha/Yala seasons, Harvest periods, Peak seasons      |
| Holidays   | Public holidays, Festival periods, Demand multipliers |
| Engineered | Lag features, Rolling statistics, Price momentum      |

---

## Models

### Price Prediction Models

| Model    | Horizon  | R-squared | MAE (LKR) |
| -------- | -------- | --------- | --------- |
| LightGBM | 1 week   | 0.85      | 7.98      |
| LightGBM | 2 weeks  | 0.84      | 8.37      |
| LightGBM | 4 weeks  | 0.84      | 8.14      |
| LightGBM | 8 weeks  | 0.77      | 9.63      |
| LightGBM | 12 weeks | 0.70      | 10.96     |

### Decision Engine

| Recommendation | Condition                | Action                  |
| -------------- | ------------------------ | ----------------------- |
| STRONG SELL    | Price drops more than 7% | Sell immediately        |
| SELL           | Price drops 3-7%         | Sell soon               |
| WAIT           | Price stable (within 3%) | Monitor market          |
| HOLD           | Price rises 3-7%         | Wait for better price   |
| STRONG HOLD    | Price rises more than 7% | Hold for maximum profit |

---

## Installation

### Requirements

- Python 3.8 or higher
- Jupyter Notebook or VS Code with Jupyter extension

### Setup

```bash
# Clone the repository
git clone https://github.com/it22889942/Yield-Sync---Research-Project.git
cd Yield-Sync---Research-Project

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
lightgbm>=3.3.0
tensorflow>=2.10.0
joblib>=1.2.0
```

---

## Usage

### Running the Notebooks

Execute notebooks in order:

1. **01_EDA_Crop_Price_Demand.ipynb** - Exploratory data analysis
2. **02_Price_Prediction_Model.ipynb** - Single-step price prediction
3. **03_Multi_Horizon_Forecasting.ipynb** - Multi-horizon price forecasting
4. **04_Demand_Forecasting.ipynb** - Volume/demand prediction
5. **05_Decision_Alert_Engine.ipynb** - SELL/HOLD recommendations

### Loading Saved Models

```python
import lightgbm as lgb
import joblib

# Load price prediction model
model = lgb.Booster(model_file='models/saved_models/multi_horizon/lgb_1_week_model.txt')

# Load configuration
config = joblib.load('models/saved_models/multi_horizon/config.joblib')

# Make predictions
predictions = model.predict(features)
```
