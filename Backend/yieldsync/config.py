"""
YieldSync Configuration
=======================
All configuration settings for the price forecasting system.
"""

# =============================================================================
# SUPPORTED CROPS AND MARKETS
# =============================================================================

TARGET_CROPS = ['Rice', 'Beetroot', 'Radish', 'Red Onion']

# Crop-Market Mapping: Only these markets are valid for each crop
CROP_MARKETS = {
    'Rice': [
        'Colombo', 'Anuradhapura', 'Moneragala', 'Dambulla',
        'Ampara', 'Kandy', 'Kurunegala', 'Polonnaruwa'
    ],
    'Beetroot': [
        'Colombo', 'Thambuththegama', 'Bandarawela',
        'Dambulla', 'Kandy', 'Nuwara Eliya'
    ],
    'Radish': [
        'Colombo', 'Moneragala', 'Dambulla', 'Kandy', 'Meegoda'
    ],
    'Red Onion': [
        'Colombo', 'Puttalam', 'Mullaittivu', 'Vavuniya', 'Batticaloa',
        'Dambulla', 'Embilipitiya', 'Jaffna', 'Kandy', 'Mannar',
        'Meegoda', 'Moneragala', 'Nuwara Eliya', 'Thambuththegama', 'Trincomalee'
    ]
}

# Get all unique markets
ALL_MARKETS = sorted(set(m for markets in CROP_MARKETS.values() for m in markets))

# =============================================================================
# FORECAST HORIZONS
# =============================================================================

# Label -> Days mapping for UI
HORIZONS = {
    '1 Week': 7,
    '2 Weeks': 14,
    '1 Month': 30
}

# Forecast horizons in days (for models)
FORECAST_HORIZONS = [7, 14, 30]

# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

# Price model configurations per crop
PRICE_MODEL_CONFIG = {
    'Rice': {
        'model_type': 'LSTM',
        'lag_days': 60,
        'univariate': True,  # Price only, no weather
    },
    'Beetroot': {
        'model_type': 'RandomForest',
        'lag_days': 7,
        'univariate': False,  # Price + weather
    },
    'Radish': {
        'model_type': 'RandomForest',
        'lag_days': 90,
        'univariate': False,
    },
    'Red Onion': {
        'model_type': 'LightGBM',
        'lag_days': 45,
        'univariate': False,
    }
}

# Demand model configurations per crop
DEMAND_MODEL_CONFIG = {
    'Rice': {
        'model_type': 'LSTM',
        'lag_days': 60,
        'univariate': True,
    },
    'Beetroot': {
        'model_type': 'RandomForest',
        'lag_days': 7,
        'univariate': False,
    },
    'Radish': {
        'model_type': 'RandomForest',
        'lag_days': 90,
        'univariate': False,
    },
    'Red Onion': {
        'model_type': 'LightGBM',
        'lag_days': 45,
        'univariate': False,
    }
}

# Weather features used for multivariate models
WEATHER_FEATURES = ['temp', 'rainfall', 'humidity', 'wind_speed', 'sunshine_hours']

# =============================================================================
# CROP PROPERTIES
# =============================================================================

# Perishability (Days until spoiled without storage)
PERISHABILITY = {
    'Rice': 180,
    'Beetroot': 7,
    'Radish': 5,
    'Red Onion': 30
}

# Estimated Model RMSE (for confidence intervals)
MODEL_RMSE = {
    'Rice': 15.5,
    'Beetroot': 22.3,
    'Radish': 12.8,
    'Red Onion': 45.2
}

# =============================================================================
# RECOMMENDATION THRESHOLDS
# =============================================================================

RECOMMENDATION_THRESHOLDS = {
    'strong_hold': 0.10,    # >= +10% profit change
    'hold': 0.02,           # +2% to +10%
    'neutral_upper': 0.02,  # -2% to +2%
    'neutral_lower': -0.02,
    'sell': -0.10,          # -2% to -10%
    'strong_sell': -0.10    # <= -10%
}

# =============================================================================
# DEFAULT VALUES
# =============================================================================

# Default Weather Values (Sri Lanka Averages)
DEFAULT_WEATHER = {
    'temp': 27.5,
    'rainfall': 5.0,
    'humidity': 75.0,
    'wind_speed': 13.5,
    'sunshine_hours': 10.5
}

# =============================================================================
# SINHALA TRANSLATIONS
# =============================================================================

CROP_NAMES_SI = {
    'Rice': 'සහල්',
    'Beetroot': 'බීට්රූට්',
    'Radish': 'රාබු',
    'Red Onion': 'රතු ලූනු'
}

RECOMMENDATION_NAMES_SI = {
    'STRONG HOLD': 'ශක්තිමත් රඳවා ගන්න',
    'HOLD': 'රඳවා ගන්න',
    'NEUTRAL': 'මධ්‍යස්ථ',
    'SELL': 'විකුණන්න',
    'STRONG SELL': 'වහාම විකුණන්න'
}

# UI Translations
TRANSLATIONS = {
    'en': {
        'title': 'YieldSync',
        'subtitle': 'Smart Farming Decision Support System',
        'crop': 'Select Crop',
        'market': 'Select Market',
        'quantity': 'Quantity to Sell (kg)',
        'days_harvest': 'Days Since Harvest',
        'get_recommendation': 'Get Recommendation',
        'decision': 'Recommendation',
        'confidence': 'Confidence',
        'expected_profit': 'Expected Profit',
        'price_forecast': 'Price Forecast',
        'demand_forecast': 'Demand Forecast',
        'sell_now': 'SELL NOW',
        'hold': 'HOLD FOR',
        'wait': 'WAIT',
        'days': 'days'
    },
    'si': {
        'title': 'YieldSync',
        'subtitle': 'බුද්ධිමත් කෘෂිකාර්මික තීරණ සහායක පද්ධතිය',
        'crop': 'බෝගය තෝරන්න',
        'market': 'වෙළඳපොළ තෝරන්න',
        'quantity': 'විකිණීමට ඇති ප්‍රමාණය (kg)',
        'days_harvest': 'අස්වැන්න නෙලා දින ගණන',
        'get_recommendation': 'නිර්දේශය ලබා ගන්න',
        'decision': 'නිර්දේශය',
        'confidence': 'විශ්වාසය',
        'expected_profit': 'බලාපොරොත්තු වන ලාභය',
        'price_forecast': 'මිල අනාවැකිය',
        'demand_forecast': 'ඉල්ලුම අනාවැකිය',
        'sell_now': 'දැන් විකුණන්න',
        'hold': 'රඳවා තබා ගන්න',
        'wait': 'රඳවා තබා ගන්න',
        'days': 'දින'
    }
}

# =============================================================================
# FILE PATHS (Relative to deployment package)
# =============================================================================

# Model directories
MODELS_DIR = 'models/saved_models'
PRICE_MODELS_DIR = f'{MODELS_DIR}/price forcasting'
DEMAND_MODELS_DIR = f'{MODELS_DIR}/demand forcasting'

# Data directory
DATA_DIR = 'data'
PRICE_DATA_FILE = f'{DATA_DIR}/full_history_features_real_weather.csv'
DEMAND_DATA_FILE = f'{DATA_DIR}/full_history_demand_data.csv'
