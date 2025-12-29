"""
Configuration settings for Yield Sync App
"""

# Supported Crops
TARGET_CROPS = ['Rice', 'Beetroot', 'Radish', 'Red Onion']

# Forecast Horizons (Label -> Days)
HORIZONS = {
    '1 Week': 7,
    '2 Weeks': 14,
    '1 Month': 30,
    '2 Months': 60
}

# Perishability (Days until spoiled without storage)
PERISHABILITY = {
    'Rice': 180,
    'Beetroot': 7,
    'Radish': 5,
    'Red Onion': 30
}

# Sinhala Translations for Crop Names
CROP_NAMES_SI = {
    'Rice': 'සහල්',
    'Beetroot': 'බීට්රූට්',
    'Radish': 'රාබු',
    'Red Onion': 'රතු ලූනු'
}

# Default Weather Values (Sri Lanka Averages)
DEFAULT_WEATHER = {
    'temperature_avg_C': 27.5,
    'rainfall_mm': 5.0,
    'humidity_percent': 75.0
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
        'wait': 'රඳවා තබා ගන්න', # Using same word for Wait/Hold in simple context or specific word
        'days': 'දින'
    }
}
