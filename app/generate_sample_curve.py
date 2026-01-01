import os
import sys
import pandas as pd
# Add parent dir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.trainer import ModelTrainer

# Define Data Path
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'full_history_features_real_weather.csv')

def run_sample_training():
    print("Running sample training to generate learning curve...")
    if not os.path.exists(DATA_PATH):
        print("Data not found")
        return

    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Initialize trainer
    # We use a temp dir for models to avoid overwriting production models if we want, 
    # but overwriting is fine as we are training the same thing.
    # Actually, let's use the real dir to ensure consistency.
    save_dir = os.path.join('models', 'saved_models')
    trainer = ModelTrainer(save_dir)
    
    # Train Rice Colombo 7-day (LSTM)
    # This invokes plot_loss_curve which saves to docs/images/training_curves
    print("Training Rice (Colombo) 7-day model...")
    trainer.train_price_model(df, 'Rice', 7, market='Colombo')
    
    print("Done. Check docs/images/training_curves/")

if __name__ == "__main__":
    run_sample_training()
