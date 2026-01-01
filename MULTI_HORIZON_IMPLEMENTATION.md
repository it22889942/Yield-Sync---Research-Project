# Multi-Horizon Forecasting Implementation

## Overview
The YieldSync system now supports **true multi-horizon forecasting** with separate models trained for each time horizon. This replaces the previous single-model approach that only predicted 1-day ahead.

## Implementation Details

### 1. Forecast Horizons
Five prediction horizons are now supported:
- **7 days** (1 week)
- **14 days** (2 weeks)
- **30 days** (4 weeks / ~1 month)
- **60 days** (8 weeks / ~2 months)
- **84 days** (12 weeks / ~3 months)

### 2. Model Architecture
**Total Models: 40**
- 4 crops × 5 horizons × 2 types (price + demand) = 40 models

**Per Crop:**
- 5 demand forecasting models (one per horizon)
- 5 price forecasting models (one per horizon)

### 3. File Structure

#### Models Directory (`models/saved_models/`)

**Demand Models** (`demand forcasting/`):
```
demand_{Crop}_{horizon}day_{type}.ext
Examples:
- demand_Rice_7day_lstm.h5
- demand_Rice_14day_lstm.h5
- demand_Beetroot_30day_rf.pkl
- demand_Red_Onion_84day_lgb.pkl
```

**Price Models** (`price forcasting/`):
```
{crop}_{horizon}day_{type}.ext
{crop}_{horizon}day_config.joblib
{crop}_{horizon}day_lstm_scalers.joblib (for LSTM only)

Examples:
- rice_7day_lstm.h5
- rice_7day_config.joblib
- rice_7day_lstm_scalers.joblib
- beetroot_14day_rf.joblib
- red_onion_30day_lgbm.joblib
```

### 4. Code Changes

#### trainer.py (COMPLETE REWRITE)
**Key Changes:**
- Added `FORECAST_HORIZONS = [7, 14, 30, 60, 84]` constant
- Updated feature engineering functions to accept `horizon` parameter
- Changed target creation: `y.append(series[i + horizon])` instead of `y.append(series[i])`
- Modified `train_demand_model()` and `train_price_model()` to accept horizon parameter
- Updated `train_all_models()` to loop through all horizons

**Training Loop Structure:**
```python
for crop in crops:
    for horizon in FORECAST_HORIZONS:
        train_demand_model(df, crop, horizon)
        train_price_model(df, crop, horizon)
```

#### predictor.py (UPDATED)
**Key Changes:**
- Added `FORECAST_HORIZONS` constant
- Changed model storage to nested dictionaries:
  - `self.demand_models[crop][horizon]`
  - `self.price_models[crop][horizon]`
  - `self.price_scalers[crop][horizon]`
  - `self.price_configs[crop][horizon]`
- Updated `_load_all_models()` to load all 40 models
- Added `_select_horizon()` method to find closest available horizon
- Updated `predict_price()` and `predict_demand()` to:
  - Select appropriate horizon model based on `days_ahead`
  - Use the selected model for prediction
  - Return `horizon_used` in results

**Horizon Selection Logic:**
```python
def _select_horizon(self, days_ahead: int) -> int:
    """Find closest available horizon to requested days_ahead"""
    closest_horizon = min(self.FORECAST_HORIZONS, 
                         key=lambda h: abs(h - days_ahead))
    return closest_horizon
```

**Example:**
- Request 20 days → Uses 14-day model (closest)
- Request 45 days → Uses 30-day model (closest)
- Request 100 days → Uses 84-day model (max horizon)

### 5. Training Instructions

To train all 40 models:

```bash
cd "e:\Bashi Github\Yield-Sync---Research-Project"
python -m models.trainer
```

**Expected Output:**
```
Training multi-horizon models for 4 crops × 5 horizons × 2 types = 40 models

TRAINING DEMAND MODELS (20 models)
Rice:
  ✓ 7-day model trained
  ✓ 14-day model trained
  ✓ 30-day model trained
  ✓ 60-day model trained
  ✓ 84-day model trained
...

TRAINING PRICE MODELS (20 models)
Rice:
  ✓ 7-day model trained
  ✓ 14-day model trained
  ...

TRAINING COMPLETE
Total time: ~XX minutes
```

### 6. Usage Examples

#### In app.py (Streamlit):
```python
# User selects horizon in UI
horizon_days = st.selectbox("Forecast Horizon", [7, 14, 30, 60, 84])

# Prediction automatically uses correct model
price_pred = predictor.predict_price(df, crop, days_ahead=horizon_days)
demand_pred = predictor.predict_demand(df, crop, days_ahead=horizon_days)

# Result includes which model was used
st.write(f"Using {price_pred['horizon_used']}-day model")
```

#### Direct API Usage:
```python
from app.predictor import CropPredictor

predictor = CropPredictor()
data = pd.read_csv('data/full_history_features_real_weather.csv')

# 1-week forecast
result_7d = predictor.predict_price(data, 'Rice', days_ahead=7)
print(f"7-day prediction: ${result_7d['predicted_price']:.2f}")
print(f"Model used: {result_7d['horizon_used']}-day")

# 3-month forecast
result_84d = predictor.predict_price(data, 'Rice', days_ahead=84)
print(f"84-day prediction: ${result_84d['predicted_price']:.2f}")
print(f"Model used: {result_84d['horizon_used']}-day")
```

### 7. Benefits

**Accuracy Improvements:**
- Short-term forecasts (7-14 days) use models optimized for near-term patterns
- Long-term forecasts (60-84 days) use models that capture seasonal trends
- No more using 1-day model to extrapolate 12 weeks

**Reliability:**
- Each model trained on correct target offset
- Validation metrics specific to each horizon
- Uncertainty increases appropriately with horizon

**Flexibility:**
- Users can request any horizon
- System automatically selects best available model
- Easy to add new horizons (just add to FORECAST_HORIZONS list)

### 8. Model Performance Tracking

Each model saves training metrics:
- **Demand models:** RMSE, MAE, R²
- **Price models:** RMSE, MAE, R², MAPE

**Expected Performance:**
- 7-day models: Highest accuracy (lowest RMSE)
- 84-day models: Lower accuracy (higher RMSE, more uncertainty)

### 9. Next Steps

**Before Deployment:**
1. ✅ Update trainer.py (COMPLETE)
2. ✅ Update predictor.py (COMPLETE)
3. ⏳ Run full training: `python -m models.trainer`
4. ⏳ Validate all 40 models load correctly
5. ⏳ Test predictions at all horizons
6. ⏳ Update UI to show horizon selection clearly

**Future Enhancements:**
- Ensemble predictions (average 14-day and 30-day for 20-day forecast)
- Confidence intervals per horizon
- Model retraining scheduler
- A/B testing different horizons

## Technical Notes

**Feature Engineering:**
- All models use same lag features (60 days for Rice LSTM, etc.)
- Only **target** changes based on horizon
- Training: `y[i] = price[i + horizon]` instead of `y[i] = price[i + 1]`

**Model Naming Convention:**
- Demand: `demand_{Crop}_{horizon}day_{type}.ext`
- Price: `{crop_lowercase}_{horizon}day_{type}.ext`
- Config: `{crop_lowercase}_{horizon}day_config.joblib`
- Scalers: `{crop_lowercase}_{horizon}day_lstm_scalers.joblib`

**Crop Naming:**
- Rice, Beetroot, Radish, Red Onion (space preserved in demand models)
- rice, beetroot, radish, red_onion (lowercase with underscore in price models)

## Troubleshooting

**If models fail to load:**
1. Check file naming matches convention
2. Verify all 40 files exist in correct directories
3. Check console output for specific missing files

**If predictions fail:**
1. Verify requested crop exists in model configs
2. Check data has sufficient history (60+ days)
3. Ensure all weather features present

**If wrong horizon used:**
1. Check FORECAST_HORIZONS list matches trained models
2. Verify _select_horizon() logic
3. Check returned `horizon_used` value in result

---

**Implementation Status:** ✅ COMPLETE (Code Ready for Training)
**Last Updated:** 2025
**Author:** GitHub Copilot
