"""
YieldSync Model Trainer
=======================
Retraining module for price and demand forecasting models.

This module handles:
- Multi-horizon model training (7, 14, 30 days)
- Per-market model training (location-specific)
- LSTM, RandomForest, and LightGBM models
- Automatic feature engineering

Usage:
    from trainer import retrain_models
    
    results = retrain_models(
        price_data_path='data/full_history_features_real_weather.csv',
        demand_data_path='data/full_history_demand_data.csv',
        save_dir='models/saved_models'
    )

Author: YieldSync Research Team
License: MIT
"""

import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from typing import Dict, Tuple, Optional, Callable, List
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# ML LIBRARIES
# =============================================================================
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("Warning: LightGBM not installed. Red Onion models will not train.")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_LSTM = True
    tf.get_logger().setLevel('ERROR')
except ImportError:
    HAS_LSTM = False
    print("Warning: TensorFlow not installed. Rice LSTM models will not train.")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Import from config or use defaults
try:
    from .config import CROP_MARKETS, FORECAST_HORIZONS
except ImportError:
    CROP_MARKETS = {
        'Rice': ['Colombo', 'Anuradhapura', 'Dambulla', 'Kandy'],
        'Beetroot': ['Colombo', 'Dambulla', 'Bandarawela'],
        'Radish': ['Colombo', 'Dambulla', 'Kandy'],
        'Red Onion': ['Colombo', 'Dambulla', 'Jaffna']
    }
    FORECAST_HORIZONS = [7, 14, 30]

# Training flag - enable per-market model training
PER_MARKET_MODELS = True

# Per-crop model configurations
DEMAND_CONFIG = {
    'Rice': {
        'model_type': 'LSTM',
        'lag_days': 60,
        'univariate': True,
        'epochs': 100,
        'batch_size': 32
    },
    'Beetroot': {
        'model_type': 'RandomForest',
        'lag_days': 7,
        'univariate': False,
        'n_estimators': 200,
        'max_depth': 15
    },
    'Radish': {
        'model_type': 'RandomForest',
        'lag_days': 90,
        'univariate': False,
        'n_estimators': 200,
        'max_depth': 15
    },
    'Red Onion': {
        'model_type': 'LightGBM',
        'lag_days': 45,
        'univariate': False,
        'num_leaves': 31,
        'learning_rate': 0.05
    }
}

PRICE_CONFIG = {
    'Rice': {
        'model_type': 'LSTM',
        'lag_days': 60,
        'univariate': True,
        'epochs': 50,
        'batch_size': 32
    },
    'Beetroot': {
        'model_type': 'RandomForest',
        'lag_days': 7,
        'univariate': False,
        'n_estimators': 100
    },
    'Radish': {
        'model_type': 'RandomForest',
        'lag_days': 90,
        'univariate': False,
        'n_estimators': 100
    },
    'Red Onion': {
        'model_type': 'LightGBM',
        'lag_days': 45,
        'univariate': False,
        'n_estimators': 100
    }
}

WEATHER_FEATURES = ['temp', 'rainfall', 'humidity', 'wind_speed', 'sunshine_hours']


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features for model training."""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['quarter'] = df['Date'].dt.quarter
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Sri Lankan agricultural seasons
    df['season_encoded'] = df['month'].isin([5, 6, 7, 8, 9]).astype(int)  # Yala
    
    # Harvest periods by crop
    harvest_map = {
        'Rice': [3, 4, 8, 9],
        'Beetroot': [7, 8, 9],
        'Radish': [6, 7, 8],
        'Red Onion': [6, 7, 8]
    }
    df['harvest_period'] = 0
    for crop, months in harvest_map.items():
        mask = (df['item'] == crop) & (df['month'].isin(months))
        df.loc[mask, 'harvest_period'] = 1
    
    return df


def create_demand_lag_features(df: pd.DataFrame, crop: str, lag_days: int, 
                               horizon: int = 1) -> pd.DataFrame:
    """Create lag features for demand prediction."""
    crop_df = df[df['item'] == crop].copy().sort_values('Date')
    
    # Create quantity lags
    for i in range(1, lag_days + 1):
        crop_df[f'qty_lag_{i}'] = crop_df['quantity_tonnes'].shift(i)
    
    # Create price lags
    for i in range(1, lag_days + 1):
        crop_df[f'price_lag_{i}'] = crop_df['price'].shift(i)
    
    # Target: future quantity
    crop_df['target'] = crop_df['quantity_tonnes'].shift(-horizon)
    
    return crop_df.dropna()


def create_price_features(df: pd.DataFrame, crop: str, lag_days: int, 
                         horizon: int = 1, include_weather: bool = False) -> Tuple:
    """Create features for price prediction."""
    crop_df = df[df['item'] == crop].copy().sort_values('Date').set_index('Date')
    
    # Resample to daily
    resampled = {'price': crop_df['price'].resample('D').mean().ffill(limit=3)}
    
    if include_weather:
        for feat in WEATHER_FEATURES:
            if feat in crop_df.columns:
                if feat == 'rainfall':
                    resampled[feat] = crop_df[feat].resample('D').sum().fillna(0)
                else:
                    resampled[feat] = crop_df[feat].resample('D').mean().ffill()
    
    series_df = pd.DataFrame(resampled).dropna()
    
    if len(series_df) < lag_days + horizon:
        return np.array([]), np.array([]), series_df
    
    # Build feature vectors
    feature_cols = [c for c in series_df.columns if c != 'price']
    X, y = [], []
    
    for i in range(lag_days, len(series_df) - horizon):
        row = list(series_df['price'].values[i-lag_days:i])
        for col in feature_cols:
            row.extend(series_df[col].values[i-lag_days:i])
        X.append(row)
        y.append(series_df['price'].values[i + horizon])
    
    return np.array(X), np.array(y), series_df


# =============================================================================
# MODEL TRAINER CLASS
# =============================================================================

class ModelTrainer:
    """Handles all model training operations."""
    
    def __init__(self, save_dir: str):
        """
        Initialize trainer.
        
        Args:
            save_dir: Base directory for saving models (models/saved_models)
        """
        self.save_dir = save_dir
        self.demand_dir = os.path.join(save_dir, 'demand forcasting')
        self.price_dir = os.path.join(save_dir, 'price forcasting')
        
        os.makedirs(self.demand_dir, exist_ok=True)
        os.makedirs(self.price_dir, exist_ok=True)
    
    def train_demand_model(self, df: pd.DataFrame, crop: str, horizon: int,
                          progress_callback: Callable = None) -> Dict:
        """Train demand model for a crop and horizon."""
        config = DEMAND_CONFIG[crop]
        lag_days = config['lag_days']
        
        if progress_callback:
            progress_callback(f"Training {crop} demand model ({horizon} days)...")
        
        # Prepare data
        df_features = add_temporal_features(df)
        crop_df = create_demand_lag_features(df_features, crop, lag_days, horizon)
        
        if len(crop_df) < lag_days + horizon + 100:
            return {'error': f'Insufficient data for {crop}'}
        
        # Train/test split
        split = int(len(crop_df) * 0.8)
        train_df, test_df = crop_df.iloc[:split], crop_df.iloc[split:]
        
        # Select features
        if config['univariate']:
            feature_cols = [c for c in train_df.columns if c.startswith('qty_lag_')]
        else:
            exclude = ['Date', 'market', 'item', 'quantity_tonnes', 'target', 'holiday_name']
            feature_cols = [c for c in train_df.columns if c not in exclude and 'Unnamed' not in c]
        
        X_train = train_df[feature_cols].values
        y_train = train_df['target'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['target'].values
        
        # Scale
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train based on model type
        if config['model_type'] == 'LSTM' and HAS_LSTM:
            X_train_lstm = X_train_scaled.reshape((len(X_train_scaled), lag_days, 1))
            X_test_lstm = X_test_scaled.reshape((len(X_test_scaled), lag_days, 1))
            
            model = Sequential([
                LSTM(64, activation='relu', input_shape=(lag_days, 1)),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            
            model.fit(X_train_lstm, y_train, epochs=config['epochs'],
                     batch_size=config['batch_size'], validation_split=0.2,
                     callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                     verbose=0)
            
            y_pred = model.predict(X_test_lstm, verbose=0).flatten()
            model_path = os.path.join(self.demand_dir, f'demand_{crop}_{horizon}day_lstm.h5')
            model.save(model_path)
            
        elif config['model_type'] == 'RandomForest':
            model = RandomForestRegressor(n_estimators=config['n_estimators'],
                                         max_depth=config['max_depth'],
                                         random_state=42, n_jobs=-1)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            model_path = os.path.join(self.demand_dir, f'demand_{crop}_{horizon}day_rf.pkl')
            joblib.dump(model, model_path)
            
        elif config['model_type'] == 'LightGBM' and HAS_LGBM:
            model = LGBMRegressor(num_leaves=config['num_leaves'],
                                 learning_rate=config['learning_rate'],
                                 n_estimators=200, random_state=42, verbose=-1)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            model_path = os.path.join(self.demand_dir, f'demand_{crop}_{horizon}day_lgb.pkl')
            joblib.dump(model, model_path)
        else:
            return {'error': f'Model type {config["model_type"]} not available'}
        
        # Metrics
        return {
            'crop': crop,
            'horizon': horizon,
            'model_type': config['model_type'],
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'model_path': model_path
        }
    
    def train_price_model(self, df: pd.DataFrame, crop: str, horizon: int,
                         market: str = None, progress_callback: Callable = None) -> Dict:
        """Train price model for a crop, horizon, and optionally market."""
        config = PRICE_CONFIG[crop]
        lag_days = config['lag_days']
        include_weather = not config['univariate']
        
        # Get market data
        crop_df = df[df['item'] == crop]
        if len(crop_df) == 0:
            return {'error': f'No data for {crop}'}
        
        if market:
            market_df = crop_df[crop_df['market'] == market].copy()
            target_market = market
        else:
            target_market = crop_df['market'].value_counts().idxmax()
            market_df = crop_df[crop_df['market'] == target_market].copy()
        
        if len(market_df) == 0:
            return {'error': f'No data for {crop} in {target_market}'}
        
        if progress_callback:
            label = f" ({market})" if market else ""
            progress_callback(f"Training {crop}{label} price model ({horizon} days)...")
        
        # Create features
        X, y, _ = create_price_features(market_df, crop, lag_days, horizon, include_weather)
        
        if len(X) < 100:
            return {'error': f'Insufficient data: {len(X)} samples'}
        
        # Split
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Filename components
        crop_slug = crop.lower().replace(' ', '_')
        market_slug = target_market.lower().replace(' ', '_') if market else ''
        suffix = f'_{market_slug}' if market else ''
        
        # Train
        if config['model_type'] == 'LSTM' and HAS_LSTM:
            scaler_y = MinMaxScaler()
            y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
            X_train_scaled = scaler_y.transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
            X_test_scaled = scaler_y.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
            
            X_train_lstm = X_train_scaled.reshape((len(X_train_scaled), lag_days, 1))
            X_test_lstm = X_test_scaled.reshape((len(X_test_scaled), lag_days, 1))
            
            model = Sequential([
                LSTM(50, activation='relu', input_shape=(lag_days, 1), return_sequences=True),
                Dropout(0.2),
                LSTM(30, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            
            model.fit(X_train_lstm, y_train_scaled, epochs=config['epochs'],
                     batch_size=config['batch_size'],
                     callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
                     verbose=0)
            
            y_pred_scaled = model.predict(X_test_lstm, verbose=0).flatten()
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            
            model_path = os.path.join(self.price_dir, f'{crop_slug}{suffix}_{horizon}day_lstm.h5')
            model.save(model_path)
            
            scalers_path = os.path.join(self.price_dir, f'{crop_slug}{suffix}_{horizon}day_lstm_scalers.joblib')
            joblib.dump({'y': scaler_y}, scalers_path)
            
        elif config['model_type'] == 'RandomForest':
            model = RandomForestRegressor(n_estimators=config['n_estimators'],
                                         random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            model_path = os.path.join(self.price_dir, f'{crop_slug}{suffix}_{horizon}day_rf.joblib')
            joblib.dump(model, model_path)
            
        elif config['model_type'] == 'LightGBM' and HAS_LGBM:
            model = LGBMRegressor(n_estimators=config['n_estimators'],
                                 random_state=42, verbose=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            model_path = os.path.join(self.price_dir, f'{crop_slug}{suffix}_{horizon}day_lgbm.joblib')
            joblib.dump(model, model_path)
        else:
            return {'error': f'Model type {config["model_type"]} not available'}
        
        # Save config
        config_path = os.path.join(self.price_dir, f'{crop_slug}{suffix}_{horizon}day_config.joblib')
        joblib.dump({
            'model': config['model_type'],
            'horizon': horizon,
            'lag': lag_days,
            'market': target_market
        }, config_path)
        
        return {
            'crop': crop,
            'horizon': horizon,
            'market': target_market,
            'model_type': config['model_type'],
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'model_path': model_path
        }
    
    def train_all_models(self, price_df: pd.DataFrame, demand_df: pd.DataFrame,
                        progress_callback: Callable = None) -> Dict:
        """Train all models for all crops and horizons."""
        results = {
            'demand_models': {},
            'price_models': {},
            'timestamp': datetime.now().isoformat(),
            'success': True
        }
        
        crops = list(DEMAND_CONFIG.keys())
        
        # Skip demand models (removed from app)
        if progress_callback:
            progress_callback("=" * 50)
            progress_callback("DEMAND MODELS SKIPPED (Price-only mode)")
            progress_callback("=" * 50)
        
        # Price models
        if progress_callback:
            progress_callback("")
            progress_callback("=" * 50)
            progress_callback("TRAINING PRICE MODELS")
            progress_callback("=" * 50)
        
        if PER_MARKET_MODELS:
            for crop in crops:
                results['price_models'][crop] = {}
                for market in CROP_MARKETS.get(crop, []):
                    results['price_models'][crop][market] = {}
                    for horizon in FORECAST_HORIZONS:
                        try:
                            result = self.train_price_model(price_df, crop, horizon, market, progress_callback)
                            results['price_models'][crop][market][f'{horizon}day'] = result
                        except Exception as e:
                            results['price_models'][crop][market][f'{horizon}day'] = {'error': str(e)}
                            results['success'] = False
        else:
            for crop in crops:
                results['price_models'][crop] = {}
                for horizon in FORECAST_HORIZONS:
                    try:
                        result = self.train_price_model(price_df, crop, horizon, None, progress_callback)
                        results['price_models'][crop][f'{horizon}day'] = result
                    except Exception as e:
                        results['price_models'][crop][f'{horizon}day'] = {'error': str(e)}
                        results['success'] = False
        
        if progress_callback:
            progress_callback("")
            progress_callback("=" * 50)
            progress_callback("TRAINING COMPLETE!")
            progress_callback("=" * 50)
        
        return results


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def retrain_models(price_data_path: str, 
                  save_dir: str, progress_callback: Callable = None,
                  demand_data_path: str = None) -> Dict:
    """
    Main function to retrain all models.
    
    Args:
        price_data_path: Path to price CSV (full_history_features_real_weather.csv)
        save_dir: Directory to save models (models/saved_models)
        progress_callback: Optional function for progress updates
        demand_data_path: Optional path to demand CSV (deprecated, not used)
    
    Returns:
        Dict with training results and metrics
    
    Example:
        results = retrain_models(
            'data/full_history_features_real_weather.csv',
            'models/saved_models'
        )
    """
    if progress_callback:
        progress_callback("Loading data...")
    
    price_df = pd.read_csv(price_data_path)
    price_df['Date'] = pd.to_datetime(price_df['Date'])
    
    # Demand models no longer used - create empty dataframe
    demand_df = pd.DataFrame()
    
    if progress_callback:
        progress_callback(f"Loaded {len(price_df)} price records")
    
    trainer = ModelTrainer(save_dir)
    return trainer.train_all_models(price_df, demand_df, progress_callback)


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    print("YieldSync Model Trainer")
    print("=" * 40)
    
    # Auto-detect paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    price_path = os.path.join(base_dir, 'data', 'full_history_features_real_weather.csv')
    save_dir = os.path.join(base_dir, 'models', 'saved_models')
    
    # Check if paths exist
    if not os.path.exists(price_path):
        parent = os.path.dirname(base_dir)
        price_path = os.path.join(parent, 'data', 'full_history_features_real_weather.csv')
        save_dir = os.path.join(parent, 'models', 'saved_models')
    
    print(f"Price data: {price_path}")
    print(f"Save directory: {save_dir}")
    print()
    
    results = retrain_models(price_path, save_dir, print)
    
    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    print(f"Success: {results['success']}")
    print(f"Timestamp: {results['timestamp']}")
