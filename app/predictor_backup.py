"""
Smart Farming Predictor - Production Module
Integrates Price & Demand Forecasting from new.ipynb and demand_forecasting.ipynb
"""

import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ProfitConfig:
    """Configuration for profit calculation"""
    transport_cost: float = 0.0  # LKR/kg
    storage_cost: float = 0.0    # LKR/kg/day
    spoilage_rate: float = 0.0   # %/day

# TensorFlow
try:
    from tensorflow.keras.models import load_model
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class SmartFarmingPredictor:
    """
    Production predictor integrating:
    - Price forecasting (per-crop optimized LSTM/RF/LightGBM)
    - Demand forecasting (per-crop optimized LSTM/RF/LightGBM)
    - Decision recommendations with confidence scoring
    
    Model Configuration (from notebooks):
    - Rice: LSTM (60-day consecutive)
    - Beetroot: RandomForest (7-day consecutive)
    - Radish: RandomForest (90-day consecutive)
    - Red Onion: LightGBM (45-day consecutive)
    """
    
    # Per-crop configurations matching notebooks exactly
    # Per-crop configurations matching notebooks exactly
    DEMAND_CONFIG = {
        'Rice': {
            'model_type': 'LSTM',
            'lag_days': 60,
            'univariate': True,
            'model_file': os.path.join('demand forcasting', 'demand_Rice_lstm.h5')
        },
        'Beetroot': {
            'model_type': 'RandomForest',
            'lag_days': 7,
            'univariate': False,
            'model_file': os.path.join('demand forcasting', 'demand_Beetroot_rf.pkl')
        },
        'Radish': {
            'model_type': 'RandomForest',
            'lag_days': 90,
            'univariate': False,
            'model_file': os.path.join('demand forcasting', 'demand_Radish_rf.pkl')
        },
        'Red Onion': {
            'model_type': 'LightGBM',
            'lag_days': 45,
            'univariate': False,
            'model_file': os.path.join('demand forcasting', 'demand_Red Onion_lgb.pkl')
        }
    }
    
    PRICE_CONFIG = {
        'Rice': {
            'model_type': 'LSTM',
            'lag_days': 60,
            'model_file': os.path.join('price forcasting', 'rice_lstm.h5')
        },
        'Beetroot': {
            'model_type': 'RandomForest',
            'lag_days': 7,
            'model_file': os.path.join('price forcasting', 'beetroot_rf.joblib')
        },
        'Radish': {
            'model_type': 'RandomForest',
            'lag_days': 90,
            'model_file': os.path.join('price forcasting', 'radish_rf.joblib')
        },
        'Red Onion': {
            'model_type': 'LightGBM',
            'lag_days': 45,
            'model_file': os.path.join('price forcasting', 'red_onion_lgbm.joblib')
        }
    }

    # Estimated RMSE for each crop (based on validation set performance)
    # Used for calculating prediction intervals
    MODEL_RMSE = {
        'Rice': 15.5,
        'Beetroot': 22.3,
        'Radish': 12.8,
        'Red Onion': 45.2
    }
    
    def __init__(self, model_base_dir: str = '../'):
        """
        Initialize predictor and load all models.
        
        Args:
            model_base_dir: Path to root directory (containing saved_models/)
        """
        self.model_base_dir = model_base_dir
        self.demand_models = {}
        self.price_models = {}
        self.demand_scalers = {}  # Scalers for demand prediction
        self.price_scalers = {}   # Scalers for price prediction
        
        self._load_all_models()
    
    def _load_all_models(self):
        """Load all 8 models (4 demand + 4 price) from saved_models directory."""
        model_dir = os.path.join(self.model_base_dir, 'saved_models')
        
        if not os.path.exists(model_dir):
            print(f"⚠️ Model directory not found: {model_dir}")
            return
        
        # Load demand models
        print("\n" + "="*60)
        print("LOADING DEMAND MODELS")
        print("="*60)
        for crop, config in self.DEMAND_CONFIG.items():
            try:
                filepath = os.path.join(model_dir, config['model_file'])
                if config['model_type'] == 'LSTM':
                    self.demand_models[crop] = load_model(filepath, compile=False)
                else:
                    self.demand_models[crop] = joblib.load(filepath)
                print(f"✓ {crop:12} ({config['model_type']:12} {config['lag_days']:3}d) loaded")
            except Exception as e:
                print(f"✗ {crop:12} - {str(e)[:45]}")
        
        # Load price models
        print("\n" + "="*60)
        print("LOADING PRICE MODELS")
        print("="*60)
        for crop, config in self.PRICE_CONFIG.items():
            try:
                filepath = os.path.join(model_dir, config['model_file'])
                if config['model_type'] == 'LSTM':
                    self.price_models[crop] = load_model(filepath, compile=False)
                else:
                    self.price_models[crop] = joblib.load(filepath)
                print(f"✓ {crop:12} ({config['model_type']:12} {config['lag_days']:3}d) loaded")
            except Exception as e:
                print(f"✗ {crop:12} - {str(e)[:45]}")
    
    def _create_demand_features(self, data: pd.DataFrame, crop: str) -> Optional[np.ndarray]:
        """
        Create demand lag features (consecutive days) matching demand_forecasting.ipynb.
        Returns feature vector ready for model prediction.
        """
        config = self.DEMAND_CONFIG[crop]
        lag_days = config['lag_days']
        
        # Filter crop data and sort by date
        crop_data = data[data['item'] == crop].copy().sort_values('Date')
        
        if len(crop_data) < lag_days:
            return None, f"Insufficient data: need {lag_days} days, have {len(crop_data)}"
        
        # Get last lag_days of quantity data
        latest_qty = crop_data['quantity_tonnes'].values[-lag_days:]
        
        # Univariate features (qty lags only) for Rice LSTM
        if config['univariate']:
            feature_vector = latest_qty.reshape(1, -1)
        else:
            # Multivariate: quantity + price lags
            latest_price = crop_data['price'].values[-lag_days:]
            feature_vector = np.concatenate([latest_qty, latest_price]).reshape(1, -1)
        
        return feature_vector, None
    
    def _create_price_features(self, data: pd.DataFrame, crop: str) -> Optional[np.ndarray]:
        """
        Create price lag features (consecutive days) matching new.ipynb.
        Returns feature vector ready for model prediction.
        """
        config = self.PRICE_CONFIG[crop]
        lag_days = config['lag_days']
        
        # Filter crop data and sort by date
        crop_data = data[data['item'] == crop].copy().sort_values('Date')
        
        if len(crop_data) < lag_days:
            return None, f"Insufficient data: need {lag_days} days, have {len(crop_data)}"
        
        # Get last lag_days of price data
        latest_price = crop_data['price'].values[-lag_days:]
        feature_vector = latest_price.reshape(1, -1)
        
        return feature_vector, None
    
    def predict_demand(self, current_data: pd.DataFrame, crop: str, days_ahead: int = 7) -> Dict:
        """
        Predict future quantity demanded for crop.
        
        Args:
            current_data: DataFrame with columns ['Date', 'item', 'market', 'quantity_tonnes', 'price', ...]
            crop: Crop name ('Rice', 'Beetroot', 'Radish', 'Red Onion')
            days_ahead: Forecast horizon (not used in actual prediction, for reference)
        
        Returns:
            Dict with predicted demand or error message
        """
        if crop not in self.demand_models:
            return {'error': f'Demand model for {crop} not loaded'}
        
        # Get crop data and validate
        crop_data = current_data[current_data['item'] == crop].copy().sort_values('Date')
        if len(crop_data) == 0:
            return {'error': f'No data found for crop {crop}'}
        
        current_demand = crop_data['quantity_tonnes'].iloc[-1]
        
        # Create features
        feature_vector, error = self._create_demand_features(current_data, crop)
        if error:
            return {'error': error}
        
        # Scale features using MinMaxScaler
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(feature_vector)
        
        # Model prediction
        config = self.DEMAND_CONFIG[crop]
        if config['model_type'] == 'LSTM':
            # Reshape for LSTM: [samples, timesteps, features]
            features_scaled = features_scaled.reshape((1, config['lag_days'], 1))
            predicted = self.demand_models[crop].predict(features_scaled, verbose=0)
            predicted_demand = float(predicted[0][0])
        else:
            # RandomForest or LightGBM
            predicted = self.demand_models[crop].predict(features_scaled)
            predicted_demand = float(predicted[0])
        
        # Calculate change percentage
        demand_change_pct = ((predicted_demand - current_demand) / current_demand * 100) if current_demand > 0 else 0
        
        return {
            'crop': crop,
            'current_demand': float(current_demand),
            'predicted_demand': predicted_demand,
            'demand_change_percent': demand_change_pct,
            'days_ahead': days_ahead,
            'model_type': config['model_type'],
            'lag_days': config['lag_days']
        }
    
    def predict_price(self, current_data: pd.DataFrame, crop: str, days_ahead: int = 7) -> Dict:
        """
        Predict future price for crop.
        
        Args:
            current_data: DataFrame with columns ['Date', 'item', 'price', ...]
            crop: Crop name ('Rice', 'Beetroot', 'Radish', 'Red Onion')
            days_ahead: Forecast horizon (not used in actual prediction, for reference)
        
        Returns:
            Dict with predicted price or error message
        """
        if crop not in self.price_models:
            return {'error': f'Price model for {crop} not loaded'}
        
        # Get crop data and validate
        crop_data = current_data[current_data['item'] == crop].copy().sort_values('Date')
        if len(crop_data) == 0:
            return {'error': f'No data found for crop {crop}'}
        
        current_price = crop_data['price'].iloc[-1]
        
        # Create features
        feature_vector, error = self._create_price_features(current_data, crop)
        if error:
            return {'error': error}
        
        # Scale features
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(feature_vector)
        
        # Model prediction
        config = self.PRICE_CONFIG[crop]
        if config['model_type'] == 'LSTM':
            # Reshape for LSTM: [samples, timesteps, features]
            features_scaled = features_scaled.reshape((1, config['lag_days'], 1))
            predicted = self.price_models[crop].predict(features_scaled, verbose=0)
            predicted_price = float(predicted[0][0])
        else:
            # RandomForest or LightGBM
            predicted = self.price_models[crop].predict(features_scaled)
            predicted_price = float(predicted[0])
        
        # Calculate change percentage
        price_change_pct = ((predicted_price - current_price) / current_price * 100) if current_price > 0 else 0
        
        # Calculate Prediction Intervals (95% CI)
        # Using fixed RMSE approximation: value +/- 1.96 * RMSE
        rmse = self.MODEL_RMSE.get(crop, 20.0)
        margin = 1.96 * rmse
        lower_bound = predicted_price - margin
        upper_bound = predicted_price + margin
        
        return {
            'crop': crop,
            'current_price': float(current_price),
            'predicted_price': predicted_price,
            'price_change_percent': price_change_pct,
            'confidence_interval': {
                'lower': lower_bound,
                'upper': upper_bound,
                'margin': margin
            },
            'days_ahead': days_ahead,
            'model_type': config['model_type'],
            'lag_days': config['lag_days']
        }
    
    def get_recommendation(self, price_change_pct: float, demand_change_pct: float, 
                          current_price: float, predicted_price: float,
                          profit_config: ProfitConfig = None,
                          days_ahead: int = 7) -> Tuple[str, float, str]:
        """
        Generate trading recommendation based on PROFITABILITY.
        Considers price change, transport cost, storage cost, and spoilage.
        
        Args:
            price_change_pct: Expected price change percentage
            demand_change_pct: Expected demand change percentage
            current_price: Current market price per kg
            predicted_price: Forecasted price per kg
            profit_config: storage/transport/spoilage costs
            days_ahead: Number of days to hold
        
        Returns:
            (recommendation, confidence_score, reasoning_text)
        """
        if profit_config is None:
            profit_config = ProfitConfig()
            
        # --- 1. Economic Analysis ---
        # Assume a standard batch size for calculation (e.g. 1000 kg)
        BATCH_KG = 1000.0
        
        # Scenario A: Sell NOW
        revenue_now = (current_price * BATCH_KG) - profit_config.transport_cost
        
        # Scenario B: Sell LATER
        # Spoilage reduces the sellable quantity
        spoilage_factor = 1.0 - (profit_config.spoilage_rate / 100.0 * days_ahead)
        qty_later = BATCH_KG * max(0.0, spoilage_factor)
        
        # Holding costs
        storage_cost_total = profit_config.storage_cost * days_ahead * BATCH_KG
        
        revenue_later = (predicted_price * qty_later) - profit_config.transport_cost - storage_cost_total
        
        # Net Benefit of Waiting
        profit_delta = revenue_later - revenue_now
        profit_delta_pct = (profit_delta / revenue_now * 100) if revenue_now > 0 else 0
        
        # --- 2. Signal Generation ---
        # Thresholds for profit-based decision
        PROFIT_HOLD_THRESHOLD = 2.0   # If we make >2% more by waiting -> HOLD
        LOSS_SELL_THRESHOLD = -2.0    # If we lose >2% by waiting -> SELL
        
        recommendation = "WAIT"
        reasoning_parts = []
        
        if profit_delta_pct >= PROFIT_HOLD_THRESHOLD:
            # It is profitable to wait
            if profit_delta_pct > 10.0:
                recommendation = "STRONG HOLD"
            else:
                recommendation = "HOLD"
            reasoning_parts.append(f"Wait for profit: +{profit_delta_pct:.1f}% expected")
            
        elif profit_delta_pct <= LOSS_SELL_THRESHOLD:
            # It is better to sell now (waiting incurs loss)
            if profit_delta_pct < -10.0:
                recommendation = "STRONG SELL"
            else:
                recommendation = "SELL"
            reasoning_parts.append(f"Sell now to avoid loss: {profit_delta_pct:.1f}% if held")
            
        else:
            # Neutral / Risks outweigh small gains
            recommendation = "WAIT"
            reasoning_parts.append(f"Marginal gain/loss: {profit_delta_pct:+.1f}%")

        # --- 3. Demand Context ---
        if abs(demand_change_pct) > 2.0:
             d_dir = "rising" if demand_change_pct > 0 else "falling"
             reasoning_parts.append(f"Demand {d_dir} ({demand_change_pct:+.1f}%)")

        reasoning = " | ".join(reasoning_parts)

        # --- 4. Confidence Score ---
        # Base confidence on magnitude of price movement model detected
        # (The profit model correctness depends on the price prediction accuracy)
        model_confidence = min(abs(price_change_pct) / 10.0, 0.9)
        
        # Adjust based on economic clarity
        # If profit delta is huge, we are more confident in the economic decision
        if abs(profit_delta_pct) > 5.0:
            confidence = min(model_confidence + 0.2, 1.0)
        else:
            confidence = model_confidence
            
        return recommendation, confidence, reasoning
    
    def get_all_predictions(self, current_data: pd.DataFrame, profit_config: ProfitConfig = None) -> Dict:
        """
        Generate all predictions and recommendations for all crops.
        
        Args:
            current_data: Full dataset with all crops
            profit_config: Optional profit configuration
        
        Returns:
            Dict with predictions for each crop
        """
        results = {}
        
        for crop in self.DEMAND_CONFIG.keys():
            demand_pred = self.predict_demand(current_data, crop)
            price_pred = self.predict_price(current_data, crop)
            
            if 'error' not in demand_pred and 'error' not in price_pred:
                demand_change = demand_pred['demand_change_percent']
                price_change = price_pred['price_change_percent']
                
                rec, conf, reason = self.get_recommendation(
                    price_change, 
                    demand_change,
                    price_pred['current_price'],
                    price_pred['predicted_price'],
                    profit_config=profit_config
                )
                
                results[crop] = {
                    'demand': demand_pred,
                    'price': price_pred,
                    'recommendation': {
                        'action': rec,
                        'confidence': conf,
                        'reasoning': reason,
                        'profit_config_used': profit_config
                    }
                }
            else:
                results[crop] = {
                    'error': 'Prediction failed',
                    'demand_error': demand_pred.get('error'),
                    'price_error': price_pred.get('error')
                }
        
        return results



# =================================================================================
# WRAPPER FOR V2.0 APP INTERFACE
# =================================================================================
class YieldSyncPredictor(SmartFarmingPredictor):
    """
    Wrapper for V2.0 App Interface.
    Simplifies the API to match the new Streamlit requirements.
    """
    def __init__(self, model_base_dir: Optional[str] = None):
        # Auto-detect path if not provided
        if model_base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(base_dir)
            model_base_dir = os.path.join(project_root, 'models')
        super().__init__(model_base_dir=model_base_dir)

    def get_recommendation(
        self, 
        crop: str, 
        price_history: List[float], 
        volume_history: List[float], 
        current_date: datetime, 
        quantity_kg: float, 
        days_since_harvest: int
    ) -> Dict:
        """
        V2.0 API for getting recommendations.
        """
        # 1. Prepare Data
        current_price = price_history[-1]
        
        # Construct dataframe for predictions
        # We assume daily intervals ending at current_date
        # Ensure we have enough history for the longest lag (90 days)
        # Pad with the first value if history is too short
        df_len = 120 # Safe buffer
        
        hist_len = len(price_history)
        if hist_len < df_len:
             # simple pad
             price_history = [price_history[0]] * (df_len - hist_len) + price_history
             volume_history = [volume_history[0]] * (df_len - hist_len) + volume_history
        
        # Recalculate length after padding
        hist_len = len(price_history)
        dates = [current_date - pd.Timedelta(days=hist_len-1-i) for i in range(hist_len)]

        df = pd.DataFrame({
            'Date': dates,
            'item': [crop] * hist_len,
            'price': price_history,
            'volume_MT': volume_history,
            # Add dummy weather/other cols required by feature engineering
            'temperature_avg_C': [27.5] * hist_len,
            'rainfall_mm': [5.0] * hist_len,
            'humidity_percent': [75.0] * hist_len,
            'wind_speed': [10.0] * hist_len, 
            'sunshine_hours': [5.0] * hist_len,
            'is_holiday': [0] * hist_len,
            'is_public_holiday': [0] * hist_len, # Match column name
            'demand_multiplier': [1.0] * hist_len,
            'season_encoded': [0] * hist_len,
            'harvest_period': [0] * hist_len,
            # Additional columns that might be checked
            'quantity_tonnes': volume_history # alias volume_MT
        })
        
        predictions = {}
        demand_predictions = {}
        
        # 2. Get Forecasts for standard horizons
        horizons = [7, 14, 30, 60]
        labels = ['1 Week', '2 Weeks', '1 Month', '2 Months']
        
        # Profit Config Defaults for V2.0
        profit_config = ProfitConfig(
            transport_cost=5.0,
            storage_cost=1.0,
            spoilage_rate=0.5
        )

        best_decision = "WAIT"
        best_profit = -float('inf')
        best_horizon = 0
        best_price = current_price
        best_reason = "No profitable opportunity found."
        
        # Get Price Predictions
        for days, label in zip(horizons, labels):
            try:
                pred = self.predict_price(df, crop, days)
                if 'predicted_price' in pred:
                    p_price = pred['predicted_price']
                    predictions[label] = round(p_price, 2)
                    
                    # Profit Logic
                    spoilage_loss_kg = quantity_kg * (profit_config.spoilage_rate / 100 * days)
                    remaining_qty = max(0, quantity_kg - spoilage_loss_kg)
                    
                    revenue_later = (p_price * remaining_qty) - profit_config.transport_cost
                    revenue_now = (current_price * quantity_kg) - profit_config.transport_cost
                    storage_total = profit_config.storage_cost * quantity_kg * days
                    
                    net_revenue_later = revenue_later - storage_total
                    profit_gain = net_revenue_later - revenue_now
                    
                    if profit_gain > best_profit:
                        best_profit = profit_gain
                        best_horizon = days
                        best_price = p_price
                        
                        price_change_pct = ((p_price - current_price) / current_price) * 100
                        if profit_gain > (revenue_now * 0.05): # 5% gain threshold
                            best_decision = "HOLD" if days < 30 else "STRONG HOLD"
                            best_reason = f"Price expected to rise {price_change_pct:.1f}% in {label}. Net profit +{profit_gain:.0f} LKR."
                        elif profit_gain < (revenue_now * -0.02): # 2% loss threshold
                            best_decision = "SELL"
                            best_reason = f"Price drop or high costs expected. Selling now prevents loss."
                        else:
                            best_decision = "WAIT"
                            best_reason = "Market stable. No significant advantage to holding."
            except Exception as e:
                predictions[label] = current_price
        
        # Get Demand Predictions (Optional)
        try:
             for days, label in zip(horizons, labels):
                 d_pred = self.predict_demand(df, crop, days)
                 if 'predicted_quantity' in d_pred:
                     demand_predictions[label] = round(d_pred['predicted_quantity'], 1)
        except Exception:
            pass
        
        return {
            'decision': best_decision,
            'confidence': 85.0 if best_decision != "WAIT" else 60.0,
            'reasoning': best_reason,
            'expected_profit_per_kg': (best_profit / quantity_kg) if quantity_kg > 0 else 0,
            'expected_profit_total': best_profit,
            'best_hold_days': best_horizon,
            'best_time': f"in {best_horizon} days",
            'best_price': best_price,
            'current_price': current_price,
            'predictions': predictions,
            'demand_predictions': demand_predictions
        }

# Example usage
if __name__ == '__main__':
    print("YieldSyncPredictor Ready")

