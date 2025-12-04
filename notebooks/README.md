# Notebooks Directory

## Overview

This directory contains Jupyter notebooks for the Yield-Sync crop price prediction and demand forecasting project. Execute notebooks in sequential order for best results.

---

## Notebook Descriptions

### 01_EDA_Crop_Price_Demand.ipynb

**Purpose**: Exploratory Data Analysis

**Contents**:

- Dataset overview and statistics
- Price distribution analysis by crop
- Seasonal patterns visualization
- Market-wise price comparison
- Correlation analysis
- Missing value assessment

**Output**: Understanding of data characteristics and patterns

---

### 02_Price_Prediction_Model.ipynb

**Purpose**: Single-Step Price Prediction Models

**Contents**:

- Data preprocessing and feature engineering
- Random Forest model training
- Gradient Boosting model training
- LSTM neural network implementation
- Model comparison and evaluation
- Feature importance analysis

**Models Trained**:

| Model             | R-squared | MAE (LKR) |
| ----------------- | --------- | --------- |
| Random Forest     | 0.93      | ~8.5      |
| Gradient Boosting | 0.93      | ~8.5      |
| LSTM              | 0.51      | ~15       |

**Output**: Saved models in `models/saved_models/`

---

### 03_Multi_Horizon_Forecasting.ipynb

**Purpose**: Multi-Horizon Price Forecasting

**Contents**:

- Target creation for multiple horizons (1, 2, 4, 8, 12 weeks)
- Seasonal naive baseline implementation
- LightGBM per-horizon models
- Multi-output LSTM model
- Performance comparison across horizons
- Visualization of results

**Horizons**:

| Horizon  | Days Ahead | LightGBM R-squared |
| -------- | ---------- | ------------------ |
| 1 week   | 7          | 0.85               |
| 2 weeks  | 14         | 0.84               |
| 4 weeks  | 28         | 0.84               |
| 8 weeks  | 56         | 0.77               |
| 12 weeks | 84         | 0.70               |

**Output**: Models saved in `models/saved_models/multi_horizon/`

---

### 04_Demand_Forecasting.ipynb

**Purpose**: Volume/Demand Prediction

**Contents**:

- Volume target creation for multiple horizons
- Price as feature (price-demand relationship)
- LightGBM demand models
- LSTM demand models per crop
- Demand pattern analysis (monthly, weekly, holiday effects)
- Price elasticity visualization

**Key Insight**: Price affects demand - higher prices typically reduce volume

**Output**: Models saved in `models/saved_models/demand_forecasting/`

---

### 05_Decision_Alert_Engine.ipynb

**Purpose**: SELL/HOLD Recommendations and Alerts

**Contents**:

- Load trained price prediction models
- Generate predictions for current market state
- Apply decision rules (configurable thresholds)
- Calculate confidence scores
- Generate alerts for significant price movements
- Find optimal selling time for each crop

**Decision Logic**:

| Recommendation | Condition                           |
| -------------- | ----------------------------------- |
| STRONG SELL    | Price expected to drop more than 7% |
| SELL           | Price expected to drop 3-7%         |
| WAIT           | Price stable (within 3%)            |
| HOLD           | Price expected to rise 3-7%         |
| STRONG HOLD    | Price expected to rise more than 7% |

**Output**:

- `recommendations.csv` - All recommendations
- `best_selling_times.csv` - Optimal timing analysis
- `risk_assessment.csv` - Risk scores per crop
- `alerts.csv` - Active market alerts

---

## Execution Order

```
1. 01_EDA_Crop_Price_Demand.ipynb      (Data exploration)
2. 02_Price_Prediction_Model.ipynb     (Basic models)
3. 03_Multi_Horizon_Forecasting.ipynb  (Multi-horizon price)
4. 04_Demand_Forecasting.ipynb         (Volume prediction)
5. 05_Decision_Alert_Engine.ipynb      (Recommendations)
```

---

## Requirements

### Python Libraries

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
lightgbm
tensorflow
joblib
```

### Hardware

- Minimum 8GB RAM recommended for LSTM training
- GPU optional but speeds up TensorFlow models

---

## Running Notebooks

### In VS Code

1. Open the notebook file
2. Select Python kernel
3. Run All Cells (or run cell by cell)

### In Jupyter

```bash
cd notebooks
jupyter notebook
```

Then open the desired notebook in browser.

## Output Locations

| Notebook | Output Directory                        |
| -------- | --------------------------------------- |
| 02       | models/saved_models/                    |
| 03       | models/saved_models/multi_horizon/      |
| 04       | models/saved_models/demand_forecasting/ |
| 05       | models/saved_models/decision_engine/    |
