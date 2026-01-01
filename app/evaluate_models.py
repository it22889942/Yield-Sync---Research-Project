"""
Evaluation script to generate Accuracy Tables and Actual vs Predicted Plots.
This script evaluates the trained models against the historical data to generate performance metrics.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model

# Add parent dir to path to allow imports from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import CROP_MARKETS
from app.trainer import PRICE_CONFIG, create_price_features

# Define Data Path matches app logic
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'full_history_features_real_weather.csv')

def evaluate():
    print("Starting Model Evaluation...")
    
    # Load Main Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    
    results = []
    
    # Directories
    models_dir = os.path.join('models', 'saved_models', 'price forcasting')
    images_dir = os.path.join('docs', 'images', 'evaluation')
    os.makedirs(images_dir, exist_ok=True)
    
    # Plot settings
    sns.set_style("whitegrid")
    
    # Selected Representative Plots -> (Crop, Market, Horizon)
    plots_to_generate = [
        ('Rice', 'Colombo', 7),
        ('Rice', 'Colombo', 30),
        ('Red Onion', 'Dambulla', 7),
        ('Beetroot', 'Nuwara Eliya', 7),
        ('Radish', 'Kandy', 7)
    ]
    
    total_combinations = sum(len(markets) for markets in CROP_MARKETS.values())
    print(f"Found {total_combinations} crop-market combinations to evaluate.")
    
    for crop, markets in CROP_MARKETS.items():
        if crop not in PRICE_CONFIG: continue
        config = PRICE_CONFIG[crop]
        
        for market in markets:
            # Filter Data for this market
            market_df = df[(df['item'] == crop) & (df['market'] == market)].copy()
            
            # Skip if insufficient data
            if len(market_df) < 100: 
                continue
                
            market_slug = market.lower().replace(' ', '_')
            crop_slug = crop.lower().replace(' ', '_')
            
            # Evaluate single-day prediction (horizon=1)
            for horizon in [1]:  # Changed from [7, 14, 30, 60, 84]
                try:
                    # 1. Feature Engineering
                    # Re-use trainer logic to get X, y
                    X, y, series = create_price_features(
                        market_df, crop, config['lag_days'], horizon, not config['univariate']
                    )
                    
                    if len(X) == 0: continue
                    
                    # 2. Train/Test Split (Use last 20% validation set as 'Test' for evaluation)
                    # Note: We are evaluating on data the model likely saw during training if we trained on full dataset?
                    # The trainer splits 80/20. We should look at the Test portion (last 20%) to be fair.
                    split_idx = int(len(X) * 0.8)
                    
                    X_test = X[split_idx:]
                    y_test = y[split_idx:]
                    
                    # Calculate Dates for the Test Set
                    # y[k] corresponds to target at index k
                    # In create_price_features loop: y.append(series_df['price'].values[i + horizon])
                    # i starts at lag_days.
                    # So y indices map to series indices: range(lag_days + horizon, len(series_df))
                    target_indices = range(config['lag_days'] + horizon, len(series))
                    test_indices = target_indices[split_idx:]
                    target_dates = series.index[test_indices]
                    
                    if len(X_test) == 0: continue
                    
                    # 3. Load Model
                    model_file = ''
                    if config['model_type'] == 'LSTM':
                        model_file = f'{crop_slug}_{market_slug}_{horizon}day_lstm.h5'
                    elif config['model_type'] == 'RandomForest':
                        model_file = f'{crop_slug}_{market_slug}_{horizon}day_rf.joblib'
                    elif config['model_type'] == 'LightGBM':
                         model_file = f'{crop_slug}_{market_slug}_{horizon}day_lgbm.joblib'
                    
                    fpath = os.path.join(models_dir, model_file)
                    if not os.path.exists(fpath): 
                        # Try generic model if per-market missing? 
                        # No, we only want to evaluate what exists.
                        continue
                        
                    # 4. Predict
                    y_pred = []
                    
                    if config['model_type'] == 'LSTM':
                        model = load_model(fpath, compile=False)
                        
                        # Load Scaler
                        scalers_file = f'{crop_slug}_{market_slug}_{horizon}day_lstm_scalers.joblib'
                        scalers_path = os.path.join(models_dir, scalers_file)
                        
                        if os.path.exists(scalers_path):
                            scalers = joblib.load(scalers_path)
                            scaler_y = scalers['y']
                            
                            # Scale Test Data (Using scaler_y as proxy for input scaling - matching Predictor logic)
                            # Flatten, scale, reshape
                            orig_shape = X_test.shape
                            X_flat = X_test.flatten().reshape(-1, 1)
                            X_scaled_flat = scaler_y.transform(X_flat)
                            X_scaled = X_scaled_flat.reshape(orig_shape[0], orig_shape[1], 1)
                            
                            # Predict
                            pred_scaled = model.predict(X_scaled, verbose=0)
                            y_pred = scaler_y.inverse_transform(pred_scaled).flatten()
                        else:
                            print(f"Warning: Scaler not found for {fpath}")
                            continue
                            
                    else:
                        model = joblib.load(fpath)
                        y_pred = model.predict(X_test)
                    
                    # 5. Metrics
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    r2 = r2_score(y_test, y_pred)
                    
                    results.append({
                        'Crop': crop,
                        'Market': market,
                        'Horizon': horizon,
                        'Model': config['model_type'],
                        'MAE': mae,
                        'RMSE': rmse,
                        'R2': r2,
                        'Test Samples': len(y_test)
                    })
                    
                    # 6. Plot (if selected)
                    if (crop, market, horizon) in plots_to_generate:
                        plt.figure(figsize=(12, 6))
                        plt.plot(target_dates, y_test, label='Actual Price', color='#2ca02c', alpha=0.8, linewidth=2)
                        plt.plot(target_dates, y_pred, label='Predicted Price', color='#1f77b4', alpha=0.8, linestyle='--', linewidth=2)
                        
                        plt.title(f'{crop} ({market}) - {horizon} Day Price Forecast\nModel: {config["model_type"]} | MAE: LKR {mae:.2f} | R²: {r2:.2f}', fontsize=14)
                        plt.xlabel('Date', fontsize=12)
                        plt.ylabel('Price (LKR/kg)', fontsize=12)
                        plt.legend(fontsize=12)
                        plt.grid(True, alpha=0.3)
                        
                        # Format x-axis dates
                        plt.gcf().autofmt_xdate()
                        
                        save_name = f'eval_{crop_slug}_{market_slug}_{horizon}day.png'
                        plt.savefig(os.path.join(images_dir, save_name), dpi=100, bbox_inches='tight')
                        plt.close()
                        print(f"Generated plot: {save_name}")
                        
                except Exception as e:
                    print(f"Error evaluating {crop} {market} {horizon}day: {str(e)}")
                    # import traceback
                    # traceback.print_exc()
                    continue

    # Save Summaries
    if not results:
        print("No results generated.")
        return

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(images_dir, 'model_metrics.csv'), index=False)
    
    # Create Markdown Summary
    summary_path = os.path.join(images_dir, 'evaluation_summary.md')
    with open(summary_path, 'w') as f:
        f.write("# Model Evaluation Summary\n\n")
        f.write(f"Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        f.write("## 1. Overall Performance by Crop\n\n")
        crop_summary = res_df.groupby('Crop')[['MAE', 'RMSE', 'R2']].mean().round(2)
        f.write(crop_summary.to_markdown())
        f.write("\n\n")
        
        f.write("## 2. Performance by Horizon\n\n")
        hor_summary = res_df.groupby('Horizon')[['MAE', 'RMSE', 'R2']].mean().round(2)
        f.write(hor_summary.to_markdown())
        f.write("\n\n")
        
        f.write("## 3. Top Performing Models (Best R2)\n\n")
        top_models = res_df.sort_values('R2', ascending=False).head(10)[['Crop', 'Market', 'Horizon', 'Model', 'MAE', 'R2']]
        f.write(top_models.to_markdown(index=False))
        f.write("\n\n")
        
    print(f"Evaluation Complete. Results saved to {images_dir}")

if __name__ == "__main__":
    evaluate()
