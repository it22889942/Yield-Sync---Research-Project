"""
Model Trainer Module - Retraining for YieldSync
Matches training logic from notebooks exactly:
- notebooks/demand forecasting/demand_forecasting.ipynb
- notebooks/price forecasting/2_model_comparison.ipynb
"""

import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from typing import Dict, Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_LSTM = True
    tf.get_logger().setLevel('ERROR')
except ImportError:
    HAS_LSTM = False


# =============================================================================
# CONFIGURATION (Matching Notebooks)
# =============================================================================

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
        'univariate': True,  # Price only, no weather
        'epochs': 50,
        'batch_size': 32
    },
    'Beetroot': {
        'model_type': 'RandomForest',
        'lag_days': 7,
        'univariate': False,  # Price + weather
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
# FEATURE ENGINEERING (Matching Notebooks)
# =============================================================================

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal features matching demand_forecasting.ipynb"""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['quarter'] = df['Date'].dt.quarter
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Sri Lankan seasons: Yala (May-Sep), Maha (Oct-Apr)
    df['season_encoded'] = df['month'].isin([5, 6, 7, 8, 9]).astype(int)
    
    # Harvest period
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


def create_demand_lag_features(df: pd.DataFrame, crop: str, lag_days: int) -> pd.DataFrame:
    """Create lag features for demand forecasting matching notebook"""
    crop_df = df[df['item'] == crop].copy().sort_values('Date')
    
    # Create quantity lags
    for i in range(1, lag_days + 1):
        crop_df[f'qty_lag_{i}'] = crop_df['quantity_tonnes'].shift(i)
    
    # Create price lags (for multivariate models)
    for i in range(1, lag_days + 1):
        crop_df[f'price_lag_{i}'] = crop_df['price'].shift(i)
    
    # Drop NaN rows
    crop_df = crop_df.dropna()
    
    return crop_df


def create_price_features(df: pd.DataFrame, crop: str, lag_days: int, 
                          include_weather: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create features for price forecasting matching 2_model_comparison.ipynb
    
    Returns:
        X: Feature array
        y: Target array
    """
    crop_df = df[df['item'] == crop].copy().sort_values('Date')
    
    # Resample to daily (fill gaps)
    crop_df = crop_df.set_index('Date')
    
    # Resample price
    resampled = {'price': crop_df['price'].resample('D').mean().ffill(limit=3)}
    
    # Add weather features if needed
    if include_weather:
        for feat in WEATHER_FEATURES:
            if feat in crop_df.columns:
                if feat == 'rainfall':
                    resampled[feat] = crop_df[feat].resample('D').sum().fillna(0)
                else:
                    resampled[feat] = crop_df[feat].resample('D').mean().ffill()
    
    series_df = pd.DataFrame(resampled).dropna()
    
    # Create lag features
    feature_cols = [c for c in series_df.columns if c != 'price']
    X, y = [], []
    
    for i in range(lag_days, len(series_df)):
        row_feats = []
        # Price lags
        row_feats.extend(series_df['price'].values[i-lag_days:i])
        # Weather lags
        for col in feature_cols:
            row_feats.extend(series_df[col].values[i-lag_days:i])
        X.append(row_feats)
        y.append(series_df['price'].values[i])
    
    return np.array(X), np.array(y), series_df


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

class ModelTrainer:
    """Handles model training for YieldSync"""
    
    def __init__(self, save_dir: str):
        """
        Args:
            save_dir: Directory to save trained models (models/saved_models/)
        """
        self.save_dir = save_dir
        self.demand_dir = os.path.join(save_dir, 'demand forcasting')
        self.price_dir = os.path.join(save_dir, 'price forcasting')
        
        # Create directories if needed
        os.makedirs(self.demand_dir, exist_ok=True)
        os.makedirs(self.price_dir, exist_ok=True)
        
        self.training_results = {}
    
    def train_demand_model(self, df: pd.DataFrame, crop: str, 
                           progress_callback: Optional[Callable] = None) -> Dict:
        """
        Train demand model for a specific crop.
        
        Args:
            df: DataFrame with demand data
            crop: Crop name
            progress_callback: Optional callback for progress updates
        
        Returns:
            Dict with training results
        """
        config = DEMAND_CONFIG[crop]
        lag_days = config['lag_days']
        
        if progress_callback:
            progress_callback(f"Training demand model for {crop}...")
        
        # Add temporal features
        df_features = add_temporal_features(df)
        
        # Create lag features
        crop_df = create_demand_lag_features(df_features, crop, lag_days)
        
        if len(crop_df) < lag_days + 100:
            return {'error': f'Insufficient data for {crop}: need {lag_days + 100}, have {len(crop_df)}'}
        
        # Split train/test (80/20)
        split_idx = int(len(crop_df) * 0.8)
        train_df = crop_df.iloc[:split_idx]
        test_df = crop_df.iloc[split_idx:]
        
        # Select features
        if config['univariate']:
            feature_cols = [c for c in train_df.columns if c.startswith('qty_lag_')]
        else:
            exclude_cols = ['Date', 'market', 'item', 'quantity_tonnes', 'holiday_name']
            feature_cols = [c for c in train_df.columns if c not in exclude_cols and not c.startswith('Unnamed')]
        
        X_train = train_df[feature_cols].values
        y_train = train_df['quantity_tonnes'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['quantity_tonnes'].values
        
        # Scale features
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        if config['model_type'] == 'LSTM' and HAS_LSTM:
            X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
            X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
            
            model = Sequential([
                LSTM(64, activation='relu', input_shape=(lag_days, 1)),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            model.fit(X_train_lstm, y_train, 
                     epochs=config['epochs'], 
                     batch_size=config['batch_size'],
                     validation_split=0.2, 
                     callbacks=[early_stop], 
                     verbose=0)
            
            y_pred = model.predict(X_test_lstm, verbose=0).flatten()
            
            # Save model
            model_path = os.path.join(self.demand_dir, f'demand_{crop}_lstm.h5')
            model.save(model_path)
            
        elif config['model_type'] == 'RandomForest':
            model = RandomForestRegressor(
                n_estimators=config['n_estimators'],
                max_depth=config['max_depth'],
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Save model
            model_path = os.path.join(self.demand_dir, f'demand_{crop}_rf.pkl')
            joblib.dump(model, model_path)
            
        elif config['model_type'] == 'LightGBM' and HAS_LGBM:
            model = LGBMRegressor(
                num_leaves=config['num_leaves'],
                learning_rate=config['learning_rate'],
                n_estimators=200,
                random_state=42,
                verbose=-1
            )
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Save model
            model_path = os.path.join(self.demand_dir, f'demand_{crop}_lgb.pkl')
            joblib.dump(model, model_path)
        else:
            return {'error': f'Model type {config["model_type"]} not available'}
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        result = {
            'crop': crop,
            'model_type': config['model_type'],
            'lag_days': lag_days,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'model_path': model_path
        }
        
        if progress_callback:
            progress_callback(f"✓ {crop} demand model trained (MAE: {mae:.2f})")
        
        return result
    
    def train_price_model(self, df: pd.DataFrame, crop: str,
                          progress_callback: Optional[Callable] = None) -> Dict:
        """
        Train price model for a specific crop.
        
        Args:
            df: DataFrame with price data
            crop: Crop name
            progress_callback: Optional callback for progress updates
        
        Returns:
            Dict with training results
        """
        config = PRICE_CONFIG[crop]
        lag_days = config['lag_days']
        include_weather = not config['univariate']
        
        if progress_callback:
            progress_callback(f"Training price model for {crop}...")
        
        # Get top market for this crop
        crop_df = df[df['item'] == crop]
        if len(crop_df) == 0:
            return {'error': f'No data for {crop}'}
        top_market = crop_df['market'].value_counts().idxmax()
        market_df = crop_df[crop_df['market'] == top_market].copy()
        
        # Create features
        X, y, series_df = create_price_features(market_df, crop, lag_days, include_weather)
        
        if len(X) < 100:
            return {'error': f'Insufficient data for {crop}: have {len(X)} samples'}
        
        # Split train/test (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        if config['model_type'] == 'LSTM' and HAS_LSTM:
            # Scale data
            scaler_y = MinMaxScaler()
            y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
            
            # For LSTM: scale X using same scaler (price only)
            X_train_scaled = scaler_y.transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
            X_test_scaled = scaler_y.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
            
            # Reshape for LSTM
            X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
            X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
            
            model = Sequential([
                LSTM(50, activation='relu', input_shape=(lag_days, 1), return_sequences=True),
                Dropout(0.2),
                LSTM(30, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
            
            model.fit(X_train_lstm, y_train_scaled,
                     epochs=config['epochs'],
                     batch_size=config['batch_size'],
                     callbacks=[early_stop],
                     verbose=0)
            
            y_pred_scaled = model.predict(X_test_lstm, verbose=0).flatten()
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            
            # Save model and scalers
            crop_slug = crop.lower().replace(' ', '_')
            model_path = os.path.join(self.price_dir, f'{crop_slug}_lstm.h5')
            model.save(model_path)
            
            scalers_path = os.path.join(self.price_dir, f'{crop_slug}_lstm_scalers.joblib')
            joblib.dump({'y': scaler_y}, scalers_path)
            
            config_path = os.path.join(self.price_dir, f'{crop_slug}_config.joblib')
            joblib.dump({
                'model': 'LSTM',
                'lag': lag_days,
                'market': top_market,
                'features': []
            }, config_path)
            
        elif config['model_type'] == 'RandomForest':
            model = RandomForestRegressor(
                n_estimators=config['n_estimators'],
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Save model
            crop_slug = crop.lower().replace(' ', '_')
            model_path = os.path.join(self.price_dir, f'{crop_slug}_rf.joblib')
            joblib.dump(model, model_path)
            
            config_path = os.path.join(self.price_dir, f'{crop_slug}_config.joblib')
            joblib.dump({
                'model': 'Random Forest',
                'lag': lag_days,
                'market': top_market,
                'features': WEATHER_FEATURES if include_weather else []
            }, config_path)
            
        elif config['model_type'] == 'LightGBM' and HAS_LGBM:
            model = LGBMRegressor(
                n_estimators=config['n_estimators'],
                random_state=42,
                verbose=-1
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Save model
            crop_slug = crop.lower().replace(' ', '_')
            model_path = os.path.join(self.price_dir, f'{crop_slug}_lgbm.joblib')
            joblib.dump(model, model_path)
            
            config_path = os.path.join(self.price_dir, f'{crop_slug}_config.joblib')
            joblib.dump({
                'model': 'LightGBM',
                'lag': lag_days,
                'market': top_market,
                'features': WEATHER_FEATURES if include_weather else []
            }, config_path)
        else:
            return {'error': f'Model type {config["model_type"]} not available'}
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        result = {
            'crop': crop,
            'model_type': config['model_type'],
            'lag_days': lag_days,
            'market': top_market,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'model_path': model_path
        }
        
        if progress_callback:
            progress_callback(f"✓ {crop} price model trained (MAE: {mae:.2f})")
        
        return result
    
    def train_all_models(self, price_df: pd.DataFrame, demand_df: pd.DataFrame,
                         progress_callback: Optional[Callable] = None) -> Dict:
        """
        Train all 8 models (4 demand + 4 price).
        
        Args:
            price_df: DataFrame with price data
            demand_df: DataFrame with demand data
            progress_callback: Optional callback for progress updates
        
        Returns:
            Dict with all training results
        """
        results = {
            'demand_models': {},
            'price_models': {},
            'timestamp': datetime.now().isoformat(),
            'success': True
        }
        
        crops = ['Rice', 'Beetroot', 'Radish', 'Red Onion']
        
        # Train demand models
        if progress_callback:
            progress_callback("="*50)
            progress_callback("TRAINING DEMAND MODELS")
            progress_callback("="*50)
        
        for crop in crops:
            try:
                result = self.train_demand_model(demand_df, crop, progress_callback)
                results['demand_models'][crop] = result
                if 'error' in result:
                    results['success'] = False
            except Exception as e:
                results['demand_models'][crop] = {'error': str(e)}
                results['success'] = False
        
        # Train price models
        if progress_callback:
            progress_callback("")
            progress_callback("="*50)
            progress_callback("TRAINING PRICE MODELS")
            progress_callback("="*50)
        
        for crop in crops:
            try:
                result = self.train_price_model(price_df, crop, progress_callback)
                results['price_models'][crop] = result
                if 'error' in result:
                    results['success'] = False
            except Exception as e:
                results['price_models'][crop] = {'error': str(e)}
                results['success'] = False
        
        if progress_callback:
            progress_callback("")
            progress_callback("="*50)
            progress_callback("TRAINING COMPLETE!")
            progress_callback("="*50)
        
        return results


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def retrain_models(price_data_path: str, demand_data_path: str, 
                   save_dir: str, progress_callback: Optional[Callable] = None) -> Dict:
    """
    Main function to retrain all models.
    
    Args:
        price_data_path: Path to price CSV file
        demand_data_path: Path to demand CSV file
        save_dir: Directory to save models
        progress_callback: Optional callback for progress updates
    
    Returns:
        Dict with training results
    """
    # Load data
    if progress_callback:
        progress_callback("Loading data...")
    
    price_df = pd.read_csv(price_data_path)
    price_df['Date'] = pd.to_datetime(price_df['Date'])
    
    demand_df = pd.read_csv(demand_data_path)
    demand_df['Date'] = pd.to_datetime(demand_df['Date'])
    
    if progress_callback:
        progress_callback(f"Loaded {len(price_df)} price records, {len(demand_df)} demand records")
    
    # Train models
    trainer = ModelTrainer(save_dir)
    results = trainer.train_all_models(price_df, demand_df, progress_callback)
    
    return results


if __name__ == '__main__':
    # Test training
    import sys
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    
    price_path = os.path.join(project_root, 'data', 'full_history_features_real_weather.csv')
    demand_path = os.path.join(project_root, 'data', 'full_history_demand_data.csv')
    save_dir = os.path.join(project_root, 'models', 'saved_models')
    
    def print_progress(msg):
        print(msg)
    
    results = retrain_models(price_path, demand_path, save_dir, print_progress)
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Success: {results['success']}")
    print(f"Timestamp: {results['timestamp']}")
    
    print("\nDemand Models:")
    for crop, res in results['demand_models'].items():
        if 'error' in res:
            print(f"  {crop}: ERROR - {res['error']}")
        else:
            print(f"  {crop}: MAE={res['mae']:.2f}, R²={res['r2']:.3f}")
    
    print("\nPrice Models:")
    for crop, res in results['price_models'].items():
        if 'error' in res:
            print(f"  {crop}: ERROR - {res['error']}")
        else:
            print(f"  {crop}: MAE={res['mae']:.2f}, R²={res['r2']:.3f}")
