"""
Generate Actual vs Predicted plots using ROLLING day-by-day prediction.
Each prediction uses previous ACTUAL values as input (not predictions).
This matches the notebook methodology.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import CROP_MARKETS
from app.trainer import PRICE_CONFIG

# Data Path
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'full_history_features_real_weather.csv')

def create_rolling_prediction_plot():
    """Generate day-by-day rolling predictions using actual previous values"""
    
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Output directory
    out_dir = os.path.join('docs', 'images', 'evaluation')
    os.makedirs(out_dir, exist_ok=True)
    
    # Plot settings
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # Configs to plot
    plot_configs = [
        ('Beetroot', 'Nuwara Eliya'),
        ('Rice', 'Colombo'),
        ('Radish', 'Kandy'),
        ('Red Onion', 'Dambulla'),
    ]
    
    for crop, market in plot_configs:
        print(f"Processing {crop} - {market}...")
        
        config = PRICE_CONFIG[crop]
        lag_days = config['lag_days']
        
        # Filter data
        mask = (df['item'] == crop) & (df['market'] == market)
        crop_df = df[mask].copy().sort_values('Date')
        
        if len(crop_df) < lag_days + 50:
            print(f"  Skipping - insufficient data")
            continue
        
        # Resample to daily
        crop_df = crop_df.set_index('Date')
        daily_prices = crop_df['price'].resample('D').mean().ffill()
        daily_prices = daily_prices.dropna()
        
        if len(daily_prices) < lag_days + 100:
            print(f"  Skipping - insufficient daily data")
            continue
        
        # Train/Test split (last 20% for test)
        split_idx = int(len(daily_prices) * 0.8)
        train_prices = daily_prices.iloc[:split_idx]
        test_prices = daily_prices.iloc[split_idx:]
        
        # Create training data
        X_train, y_train = [], []
        for i in range(lag_days, len(train_prices)):
            X_train.append(train_prices.values[i-lag_days:i])
            y_train.append(train_prices.values[i])
        X_train, y_train = np.array(X_train), np.array(y_train)
        
        # Train model
        if config['model_type'] == 'RandomForest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
        elif config['model_type'] == 'LightGBM':
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
            model.fit(X_train, y_train)
        else:
            # LSTM - skip for now (requires different handling)
            print(f"  Skipping LSTM model")
            continue
        
        # Day-by-day rolling prediction on TEST set
        # Each prediction uses the ACTUAL previous values
        predictions = []
        actuals = []
        dates = []
        
        # Start with last lag_days from training + test period
        full_series = daily_prices.values
        test_start_idx = split_idx
        
        for i in range(test_start_idx, len(full_series)):
            # Use ACTUAL previous values as input
            input_lags = full_series[i-lag_days:i]
            
            # Predict next day
            pred = model.predict(input_lags.reshape(1, -1))[0]
            predictions.append(pred)
            actuals.append(full_series[i])
            dates.append(daily_prices.index[i])
        
        # Calculate metrics
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(dates, actuals, label='Actual Price', color='#2ca02c', linewidth=2, alpha=0.8)
        ax.plot(dates, predictions, label='Predicted Price', color='#1f77b4', linewidth=2, alpha=0.8, linestyle='--')
        
        ax.set_title(f'{crop} - {market} | Day-by-Day Prediction\nMAE: {mae:.1f} LKR | R²: {r2:.3f}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price (LKR/kg)', fontsize=12)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45, ha='right')
        
        # Save
        crop_slug = crop.lower().replace(' ', '_')
        market_slug = market.lower().replace(' ', '_')
        fname = f'rolling_{crop_slug}_{market_slug}.png'
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=100)
        plt.close()
        
        print(f"  Saved: {fname} (MAE={mae:.1f}, R²={r2:.3f})")
    
    print("\nDone!")

if __name__ == "__main__":
    create_rolling_prediction_plot()
