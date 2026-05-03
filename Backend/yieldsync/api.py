"""
YieldSync API Interface
=======================
Simple wrapper for UI integration.

This module provides a clean API for frontend/backend developers
to integrate price forecasting into their applications.

Example Usage:
    from api import YieldSyncAPI
    
    api = YieldSyncAPI()
    
    # Get prediction
    result = api.predict('Rice', 'Colombo', 7)
    print(result['predicted_price'])
    
    # Get recommendation
    rec = api.get_recommendation('Rice', 'Colombo', 7, quantity_kg=500)
    print(rec['decision'])  # 'HOLD', 'SELL', etc.
"""

import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

from .predictor import YieldSyncPredictor
from .config import TARGET_CROPS, CROP_MARKETS, FORECAST_HORIZONS, PERISHABILITY



class YieldSyncAPI:
    """
    Simple API wrapper for UI integration.
    
    Attributes:
        crops: List of supported crops
        horizons: List of forecast horizons (days)
    """
    
    def __init__(self, data_path: str = None, model_path: str = None):
        """
        Initialize the API.
        
        Args:
            data_path: Path to price data CSV (auto-detects if not provided)
            model_path: Path to models directory (auto-detects if not provided)
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Auto-detect paths
        if data_path is None:
            data_path = os.path.join(base_dir, 'data', 'full_history_features_real_weather.csv')
            if not os.path.exists(data_path):
                parent = os.path.dirname(base_dir)
                data_path = os.path.join(parent, 'data', 'full_history_features_real_weather.csv')
        
        if model_path is None:
            model_path = os.path.join(base_dir, 'models', 'saved_models')
            if not os.path.exists(model_path):
                parent = os.path.dirname(base_dir)
                model_path = os.path.join(parent, 'models', 'saved_models')
        
        self.data_path = data_path
        self._predictor = YieldSyncPredictor(model_path)
        self._data = None
        self._last_loaded = None
        
        # Public attributes
        self.crops = TARGET_CROPS
        self.horizons = FORECAST_HORIZONS
    
    def _load_data(self, force_reload: bool = False):
        """Load data if not already loaded or if force_reload."""
        if self._data is None or force_reload:
            self._data = pd.read_csv(self.data_path)
            self._data['Date'] = pd.to_datetime(self._data['Date'])
            self._last_loaded = datetime.now()
        return self._data
    
    def get_markets(self, crop: str) -> List[str]:
        """Get available markets for a crop."""
        return CROP_MARKETS.get(crop, [])
    
    def get_current_price(self, crop: str, market: str = None) -> Dict:
        """
        Get the current (latest) price for a crop.
        
        Args:
            crop: Crop name
            market: Optional market name
        
        Returns:
            Dict with current_price, date, market
        """
        data = self._load_data()
        
        crop_data = data[data['item'] == crop].copy()
        if market:
            crop_data = crop_data[crop_data['market'] == market]
        
        if crop_data.empty:
            return {'error': f'No data for {crop}'}
        
        latest = crop_data.loc[crop_data['Date'].idxmax()]
        
        return {
            'crop': crop,
            'market': latest['market'] if 'market' in latest else market,
            'current_price': float(latest['price']),
            'date': str(latest['Date'].date())
        }
    
    def predict(self, crop: str, market: str = None, days_ahead: int = 7) -> Dict:
        """
        Get price prediction for a crop.
        
        Args:
            crop: Crop name ('Rice', 'Beetroot', 'Radish', 'Red Onion')
            market: Market name (optional, uses default if not provided)
            days_ahead: Days to forecast (7, 14, or 30)
        
        Returns:
            Dict containing:
            - current_price: Today's price (LKR/kg)
            - predicted_price: Forecasted price (LKR/kg)
            - price_change_percent: Expected change (%)
            - confidence_interval: {lower, upper} bounds
            - horizon_used: Actual model horizon used
        
        Example:
            result = api.predict('Rice', 'Colombo', 7)
            print(f"Price will change by {result['price_change_percent']}%")
        """
        if crop not in self.crops:
            return {'error': f'Unknown crop. Available: {self.crops}'}
        
        data = self._load_data()
        result = self._predictor.predict_price(data, crop, days_ahead, market)
        
        return result
    
    def get_recommendation(self, crop: str, market: str = None, days_ahead: int = 7,
                          quantity_kg: float = 1000) -> Dict:
        """
        Get buy/sell recommendation with profit analysis.
        
        Args:
            crop: Crop name
            market: Market name (optional)
            days_ahead: Holding period in days
            quantity_kg: Amount of crop in kg
        
        Returns:
            Dict containing:
            - decision: 'STRONG HOLD', 'HOLD', 'NEUTRAL', 'SELL', 'STRONG SELL'
            - reasoning: Explanation text
            - profit_analysis: Detailed profit breakdown
            - confidence: Confidence score (0-1)
        
        Example:
            rec = api.get_recommendation('Beetroot', 'Colombo', 14, 500)
            if rec['decision'] in ['HOLD', 'STRONG HOLD']:
                print(f"Wait to sell: {rec['reasoning']}")
        """
        # First get prediction
        prediction = self.predict(crop, market, days_ahead)
        
        if 'error' in prediction:
            return prediction
        
        # Get recommendation
        result = self._predictor.get_recommendation(
            crop=crop,
            current_price=prediction['current_price'],
            predicted_price=prediction['predicted_price'],
            days_ahead=days_ahead,
            quantity_kg=quantity_kg
        )
        
        # Add prediction info
        result['prediction'] = prediction
        
        return result
    
    def get_price_history(self, crop: str, market: str = None, days: int = 30) -> Dict:
        """
        Get historical price data for charting.
        
        Args:
            crop: Crop name
            market: Market name (optional)
            days: Number of days of history
        
        Returns:
            Dict with dates and prices arrays
        """
        data = self._load_data()
        
        crop_data = data[data['item'] == crop].copy()
        if market:
            crop_data = crop_data[crop_data['market'] == market]
        
        if crop_data.empty:
            return {'error': f'No data for {crop}'}
        
        # Get last N days
        crop_data = crop_data.sort_values('Date')
        if days:
            cutoff = crop_data['Date'].max() - pd.Timedelta(days=days)
            crop_data = crop_data[crop_data['Date'] >= cutoff]
        
        # Daily average price
        daily = crop_data.groupby('Date')['price'].mean().reset_index()
        
        return {
            'crop': crop,
            'market': market,
            'dates': daily['Date'].dt.strftime('%Y-%m-%d').tolist(),
            'prices': daily['price'].round(2).tolist(),
            'count': len(daily)
        }
    
    def get_all_predictions(self) -> Dict:
        """
        Get predictions for all crops at default horizon.
        
        Returns:
            Dict mapping crop name to prediction result
        """
        results = {}
        for crop in self.crops:
            results[crop] = self.predict(crop, days_ahead=7)
        return results
    
    def get_crop_info(self, crop: str) -> Dict:
        """
        Get static information about a crop.
        
        Returns:
            Dict with perishability, markets, etc.
        """
        return {
            'crop': crop,
            'perishability_days': PERISHABILITY.get(crop, 30),
            'markets': CROP_MARKETS.get(crop, []),
            'forecast_horizons': FORECAST_HORIZONS
        }


# =============================================================================
# QUICK FUNCTIONS
# =============================================================================

_api_instance = None

def get_api() -> YieldSyncAPI:
    """Get singleton API instance."""
    global _api_instance
    if _api_instance is None:
        _api_instance = YieldSyncAPI()
    return _api_instance


def quick_predict(crop: str, market: str = None, days: int = 7) -> Dict:
    """Quick prediction without explicit initialization."""
    return get_api().predict(crop, market, days)


def quick_recommendation(crop: str, market: str = None, days: int = 7, 
                        quantity_kg: float = 1000) -> Dict:
    """Quick recommendation without explicit initialization."""
    return get_api().get_recommendation(crop, market, days, quantity_kg)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("YieldSync API")
    print("=" * 40)
    
    api = YieldSyncAPI()
    
    print(f"\nSupported crops: {api.crops}")
    print(f"Forecast horizons: {api.horizons} days")
    
    print("\nExample usage:")
    print("  from api import YieldSyncAPI, quick_predict")
    print("  ")
    print("  # Using class")
    print("  api = YieldSyncAPI()")
    print("  result = api.predict('Rice', 'Colombo', 7)")
    print("  ")
    print("  # Using quick function")
    print("  result = quick_predict('Rice', 'Colombo', 7)")
