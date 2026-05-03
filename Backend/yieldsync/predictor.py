"""
YieldSync Price Forecasting - Inference Module
==============================================
This module provides price prediction capabilities for Sri Lankan crops.

Usage:
    from predictor import YieldSyncPredictor
    
    predictor = YieldSyncPredictor()
    result = predictor.predict_price(data_df, crop='Rice', days_ahead=7, market='Colombo')
    
Author: YieldSync Research Team
License: MIT
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

# =============================================================================
# OPTIONAL IMPORTS (graceful degradation if not installed)
# =============================================================================
try:
    from tensorflow.keras.models import load_model
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. LSTM models will not work.")

# Import configuration
try:
    from .config import CROP_MARKETS, TARGET_CROPS, FORECAST_HORIZONS, MODEL_RMSE, PERISHABILITY
except ImportError:
    # Fallback defaults if config not found
    CROP_MARKETS = {
        'Rice': ['Colombo', 'Anuradhapura', 'Dambulla', 'Kandy'],
        'Beetroot': ['Colombo', 'Dambulla', 'Bandarawela'],
        'Radish': ['Colombo', 'Dambulla', 'Kandy'],
        'Red Onion': ['Colombo', 'Dambulla', 'Jaffna']
    }
    TARGET_CROPS = ['Rice', 'Beetroot', 'Radish', 'Red Onion']
    FORECAST_HORIZONS = [7, 14, 30]
    MODEL_RMSE = {'Rice': 15.5, 'Beetroot': 22.3, 'Radish': 12.8, 'Red Onion': 45.2}
    PERISHABILITY = {'Rice': 180, 'Beetroot': 7, 'Radish': 5, 'Red Onion': 30}


# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class ProfitConfig:
    """Configuration for profit calculation"""
    transport_cost: float = 5.0   # LKR/kg
    storage_cost: float = 1.0     # LKR/kg/day
    spoilage_rate: float = 1.0    # %/day


# =============================================================================
# MAIN PREDICTOR CLASS
# =============================================================================
class YieldSyncPredictor:
    """
    Production-ready price predictor for Sri Lankan agricultural commodities.
    
    Supports:
    - Multi-horizon forecasting (7, 14, 30 days)
    - Per-market models (location-specific predictions)
    - Per-crop optimized models (LSTM/RandomForest/LightGBM)
    - Confidence intervals for predictions
    
    Model Configuration:
    - Rice: LSTM (60-day lag, univariate)
    - Beetroot: RandomForest (7-day lag, multivariate with weather)
    - Radish: RandomForest (90-day lag, multivariate)
    - Red Onion: LightGBM (45-day lag, multivariate)
    """
    
    # Per-crop model configurations
    PRICE_CONFIG = {
        'Rice': {'model_type': 'LSTM', 'lag_days': 60, 'univariate': True},
        'Beetroot': {'model_type': 'RandomForest', 'lag_days': 7, 'univariate': False},
        'Radish': {'model_type': 'RandomForest', 'lag_days': 90, 'univariate': False},
        'Red Onion': {'model_type': 'LightGBM', 'lag_days': 45, 'univariate': False}
    }
    
    # Demand forecasting disabled - no proper dataset available
    
    # Weather features for multivariate models
    WEATHER_FEATURES = ['temp', 'rainfall', 'humidity', 'wind_speed', 'sunshine_hours']
    
    def __init__(self, model_base_dir: str = None):
        """
        Initialize predictor and load models.
        
        Args:
            model_base_dir: Path to models/saved_models directory.
                           Auto-detects if not provided.
        """
        if model_base_dir is None:
            # Auto-detect model path
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_base_dir = os.path.join(base_dir, 'models', 'saved_models')
            
            # If not found, try parent directory
            if not os.path.exists(model_base_dir):
                parent_dir = os.path.dirname(base_dir)
                model_base_dir = os.path.join(parent_dir, 'models', 'saved_models')
        
        self.model_base_dir = model_base_dir
        
        # Model storage (price only - demand forecasting removed)
        self.price_models = {}          # {crop: {horizon: model}}
        self.price_models_per_market = {} # {crop: {market: {horizon: model}}}
        self.price_scalers = {}          # {crop: {horizon: scaler}}
        self.price_scalers_per_market = {}
        self.price_configs = {}
        self.price_configs_per_market = {}
        
        self.has_per_market_models = False
        
        # Load all models
        self._load_models()
    
    def _load_models(self):
        """Load all trained models from disk."""
        price_dir = os.path.join(self.model_base_dir, 'price forcasting')
        
        print("\n" + "="*50)
        print("Loading YieldSync Models...")
        print("="*50)
        
        # Load per-market price models
        per_market_count = 0
        for crop, markets in CROP_MARKETS.items():
            config = self.PRICE_CONFIG.get(crop)
            if not config:
                continue
            
            self.price_models_per_market[crop] = {}
            self.price_scalers_per_market[crop] = {}
            self.price_configs_per_market[crop] = {}
            
            crop_slug = crop.lower().replace(' ', '_')
            
            for market in markets:
                market_slug = market.lower().replace(' ', '_')
                self.price_models_per_market[crop][market] = {}
                self.price_scalers_per_market[crop][market] = {}
                self.price_configs_per_market[crop][market] = {}
                
                for horizon in FORECAST_HORIZONS:
                    model = self._load_single_model(
                        price_dir, crop, config, horizon, market_slug
                    )
                    if model:
                        self.price_models_per_market[crop][market][horizon] = model
                        per_market_count += 1
                        
                        # Load scalers for LSTM
                        if config['model_type'] == 'LSTM':
                            scalers_file = f'{crop_slug}_{market_slug}_{horizon}day_lstm_scalers.joblib'
                            scalers_path = os.path.join(price_dir, scalers_file)
                            if os.path.exists(scalers_path):
                                self.price_scalers_per_market[crop][market][horizon] = joblib.load(scalers_path)
                        
                        # Load config
                        config_file = f'{crop_slug}_{market_slug}_{horizon}day_config.joblib'
                        config_path = os.path.join(price_dir, config_file)
                        if os.path.exists(config_path):
                            self.price_configs_per_market[crop][market][horizon] = joblib.load(config_path)
        
        if per_market_count > 0:
            self.has_per_market_models = True
            print(f"✓ Loaded {per_market_count} per-market price models")
        
        # Load generic price models (fallback)
        generic_count = 0
        for crop, config in self.PRICE_CONFIG.items():
            self.price_models[crop] = {}
            self.price_scalers[crop] = {}
            
            crop_slug = crop.lower().replace(' ', '_')
            
            for horizon in FORECAST_HORIZONS:
                model = self._load_single_model(price_dir, crop, config, horizon)
                if model:
                    self.price_models[crop][horizon] = model
                    generic_count += 1
                    
                    # Load scalers
                    if config['model_type'] == 'LSTM':
                        scalers_file = f'{crop_slug}_{horizon}day_lstm_scalers.joblib'
                        scalers_path = os.path.join(price_dir, scalers_file)
                        if os.path.exists(scalers_path):
                            self.price_scalers[crop][horizon] = joblib.load(scalers_path)
        
        if generic_count > 0:
            print(f"✓ Loaded {generic_count} generic price models (fallback)")
        
        # Demand models removed - no proper dataset available
        # Keeping empty dict for API compatibility
        print("ℹ Demand forecasting disabled (no dataset)")
        
        print("="*50 + "\n")
    
    def _load_single_model(self, price_dir: str, crop: str, config: dict, 
                          horizon: int, market_slug: str = None):
        """Load a single price model."""
        crop_slug = crop.lower().replace(' ', '_')
        
        try:
            # Construct filename
            if market_slug:
                prefix = f'{crop_slug}_{market_slug}_{horizon}day'
            else:
                prefix = f'{crop_slug}_{horizon}day'
            
            if config['model_type'] == 'LSTM':
                model_file = f'{prefix}_lstm.h5'
            elif config['model_type'] == 'RandomForest':
                model_file = f'{prefix}_rf.joblib'
            elif config['model_type'] == 'LightGBM':
                model_file = f'{prefix}_lgbm.joblib'
            else:
                return None
            
            model_path = os.path.join(price_dir, model_file)
            
            if os.path.exists(model_path):
                if config['model_type'] == 'LSTM':
                    return load_model(model_path, compile=False)
                else:
                    return joblib.load(model_path)
        except Exception as e:
            pass  # Silent fail, will use fallback
        
        return None
    
    def _select_horizon(self, days_ahead: int) -> int:
        """Select closest available forecast horizon."""
        return min(FORECAST_HORIZONS, key=lambda h: abs(h - days_ahead))
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features to data."""
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['month'] = df['Date'].dt.month
        df['quarter'] = df['Date'].dt.quarter
        df['day_of_year'] = df['Date'].dt.dayofyear
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Sri Lankan seasons: Maha (Oct-Mar), Yala (Apr-Sep)
        df['season_encoded'] = df['month'].apply(lambda x: 1 if x in [10,11,12,1,2,3] else 0)
        df['harvest_period'] = df['month'].apply(lambda x: 1 if x in [1,2,3,8,9] else 0)
        
        return df
    
    def _create_price_features(self, data: pd.DataFrame, crop: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Create feature vector for price prediction."""
        config = self.PRICE_CONFIG[crop]
        lag_days = config['lag_days']
        
        # Filter and sort
        crop_data = data[data['item'] == crop].copy().sort_values('Date')
        
        if len(crop_data) < lag_days:
            return None, f"Need {lag_days} days of data, have {len(crop_data)}"
        
        latest = crop_data.iloc[-lag_days:]
        
        # LSTM: price only
        if config['model_type'] == 'LSTM':
            return latest['price'].values.reshape(1, -1), None
        
        # RF/LightGBM: price + weather
        all_features = [latest['price'].values]
        
        for col in self.WEATHER_FEATURES:
            if col in latest.columns:
                vals = latest[col].fillna(latest[col].mean()).values
                all_features.append(vals)
            else:
                all_features.append(np.zeros(lag_days))
        
        return np.concatenate(all_features).reshape(1, -1), None
    
    def predict_price(self, data: pd.DataFrame, crop: str, 
                     days_ahead: int = 7, market: str = None) -> Dict:
        """
        Predict future price for a crop.
        
        Args:
            data: DataFrame with columns ['Date', 'item', 'price', ...]
            crop: Crop name ('Rice', 'Beetroot', 'Radish', 'Red Onion')
            days_ahead: Forecast horizon (7, 14, or 30 days)
            market: Optional market name for location-specific prediction
        
        Returns:
            Dict containing:
            - predicted_price: Forecasted price (LKR/kg)
            - current_price: Current market price
            - price_change_percent: Expected % change
            - confidence_interval: {lower, upper} bounds
            - horizon_used: Actual model horizon used
        """
        if crop not in TARGET_CROPS:
            return {'error': f'Unknown crop: {crop}. Supported: {TARGET_CROPS}'}
        
        horizon = self._select_horizon(days_ahead)
        
        # Try per-market model first
        model = None
        scalers = None
        
        if market and self.has_per_market_models:
            if (crop in self.price_models_per_market and
                market in self.price_models_per_market.get(crop, {}) and
                horizon in self.price_models_per_market[crop].get(market, {})):
                model = self.price_models_per_market[crop][market][horizon]
                scalers = self.price_scalers_per_market.get(crop, {}).get(market, {}).get(horizon)
        
        # Fallback to generic model
        if model is None:
            if crop not in self.price_models or horizon not in self.price_models.get(crop, {}):
                return {'error': f'No model for {crop} {horizon}-day horizon'}
            model = self.price_models[crop][horizon]
            scalers = self.price_scalers.get(crop, {}).get(horizon)
        
        # Get current price
        crop_data = data[data['item'] == crop].copy().sort_values('Date')
        if len(crop_data) == 0:
            return {'error': f'No data for {crop}'}
        
        current_price = float(crop_data['price'].iloc[-1])
        config = self.PRICE_CONFIG[crop]
        
        # Create features
        features, error = self._create_price_features(data, crop)
        if error:
            return {'error': error}
        
        try:
            if config['model_type'] == 'LSTM':
                # LSTM needs scaling
                if scalers and 'y' in scalers:
                    scaler_y = scalers['y']
                    prices_col = features.reshape(-1, 1)
                    scaled = scaler_y.transform(prices_col)
                    scaled = scaled.reshape((1, config['lag_days'], 1))
                    pred_scaled = model.predict(scaled, verbose=0)
                    predicted = float(scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0])
                else:
                    scaler = MinMaxScaler()
                    scaled = scaler.fit_transform(features.reshape(-1, 1))
                    scaled = scaled.reshape((1, config['lag_days'], 1))
                    pred = model.predict(scaled, verbose=0)
                    predicted = float(pred[0][0])
            else:
                # RF/LightGBM - no scaling
                pred = model.predict(features)
                predicted = float(pred[0])
            
            predicted = max(0, predicted)  # Ensure non-negative
            
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}
        
        # Calculate metrics
        change_pct = ((predicted - current_price) / current_price * 100) if current_price > 0 else 0
        
        # Confidence interval (95%)
        rmse = MODEL_RMSE.get(crop, 20.0)
        margin = 1.96 * rmse
        
        return {
            'crop': crop,
            'market': market,
            'current_price': current_price,
            'predicted_price': round(predicted, 2),
            'price_change_percent': round(change_pct, 2),
            'confidence_interval': {
                'lower': round(max(0, predicted - margin), 2),
                'upper': round(predicted + margin, 2)
            },
            'days_ahead': days_ahead,
            'horizon_used': horizon,
            'model_type': config['model_type']
        }
    
    def predict_demand(self, data: pd.DataFrame, crop: str, days_ahead: int = 7) -> Dict:
        """
        Predict future demand for a crop.
        
        NOTE: Demand forecasting has been disabled due to lack of proper dataset.
        This method returns an error message.
        
        Args:
            data: DataFrame with columns ['Date', 'item', 'quantity_tonnes', ...]
            crop: Crop name
            days_ahead: Forecast horizon
        
        Returns:
            Dict with error message
        """
        return {
            'error': 'Demand forecasting is disabled - no proper dataset available',
            'crop': crop,
            'days_ahead': days_ahead
        }
    
    def get_recommendation(self, crop: str, current_price: float, predicted_price: float,
                          days_ahead: int = 7, quantity_kg: float = 1000,
                          profit_config: ProfitConfig = None) -> Dict:
        """
        Get trading recommendation based on price prediction and economics.
        
        Args:
            crop: Crop name
            current_price: Current price (LKR/kg)
            predicted_price: Predicted price (LKR/kg)
            days_ahead: Holding period
            quantity_kg: Batch size in kg
            profit_config: Storage/transport costs
        
        Returns:
            Dict with decision, reasoning, and profit analysis
        """
        if profit_config is None:
            # Set spoilage based on crop perishability
            shelf_life = PERISHABILITY.get(crop, 30)
            if shelf_life <= 7:
                spoilage = 2.0
            elif shelf_life <= 14:
                spoilage = 1.0
            else:
                spoilage = 0.3
            profit_config = ProfitConfig(spoilage_rate=spoilage)
        
        # Calculate economics
        revenue_now = (current_price * quantity_kg) - profit_config.transport_cost
        
        # Spoilage reduces sellable quantity
        spoilage_factor = max(0, 1.0 - (profit_config.spoilage_rate / 100.0 * days_ahead))
        qty_later = quantity_kg * spoilage_factor
        storage_total = profit_config.storage_cost * days_ahead * quantity_kg
        
        revenue_later = (predicted_price * qty_later) - profit_config.transport_cost - storage_total
        
        profit_delta = revenue_later - revenue_now
        profit_delta_pct = (profit_delta / revenue_now * 100) if revenue_now > 0 else 0
        
        # Decision logic
        if profit_delta_pct >= 10.0:
            decision = "STRONG HOLD"
            reasoning = f"Wait for +{profit_delta_pct:.1f}% profit"
        elif profit_delta_pct >= 2.0:
            decision = "HOLD"
            reasoning = f"Moderate profit opportunity: +{profit_delta_pct:.1f}%"
        elif profit_delta_pct <= -10.0:
            decision = "STRONG SELL"
            reasoning = f"Sell now to avoid {abs(profit_delta_pct):.1f}% loss"
        elif profit_delta_pct <= -2.0:
            decision = "SELL"
            reasoning = f"Sell now, holding would lose {abs(profit_delta_pct):.1f}%"
        else:
            decision = "NEUTRAL"
            reasoning = f"Price stable ({profit_delta_pct:+.1f}%), sell when convenient"
        
        return {
            'decision': decision,
            'reasoning': reasoning,
            'profit_analysis': {
                'revenue_if_sell_now': round(revenue_now, 2),
                'revenue_if_hold': round(revenue_later, 2),
                'profit_difference': round(profit_delta, 2),
                'profit_change_percent': round(profit_delta_pct, 2),
                'spoilage_loss_percent': round((1 - spoilage_factor) * 100, 2),
                'storage_cost_total': round(storage_total, 2)
            },
            'confidence': min(abs(profit_delta_pct) / 10.0 + 0.5, 0.95)
        }
    
    def get_available_markets(self, crop: str) -> List[str]:
        """Get list of markets with models for a crop."""
        return CROP_MARKETS.get(crop, [])
    
    def get_available_crops(self) -> List[str]:
        """Get list of supported crops."""
        return TARGET_CROPS


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================
def quick_predict(crop: str, market: str = None, days_ahead: int = 7, 
                  data_path: str = None) -> Dict:
    """
    Quick prediction without manual data loading.
    
    Args:
        crop: Crop name
        market: Market name (optional)
        days_ahead: Forecast horizon
        data_path: Path to CSV data (auto-detects if not provided)
    
    Returns:
        Prediction result dict
    """
    # Auto-detect data path
    if data_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_dir, 'data', 'full_history_features_real_weather.csv')
        
        if not os.path.exists(data_path):
            parent_dir = os.path.dirname(base_dir)
            data_path = os.path.join(parent_dir, 'data', 'full_history_features_real_weather.csv')
    
    if not os.path.exists(data_path):
        return {'error': f'Data file not found: {data_path}'}
    
    # Load data
    data = pd.read_csv(data_path)
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Initialize predictor and predict
    predictor = YieldSyncPredictor()
    return predictor.predict_price(data, crop, days_ahead, market)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print("YieldSync Predictor Module")
    print("=" * 40)
    print(f"Supported crops: {TARGET_CROPS}")
    print(f"Forecast horizons: {FORECAST_HORIZONS} days")
    print("\nExample usage:")
    print("  from predictor import YieldSyncPredictor, quick_predict")
    print("  result = quick_predict('Rice', market='Colombo', days_ahead=7)")
    print("  print(result)")
