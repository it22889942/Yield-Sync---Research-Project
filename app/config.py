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
    '2 Months': 60,
    '3 Months': 84
}

# Perishability (Days until spoiled without storage)
PERISHABILITY = {
    'Rice': 180,
    'Beetroot': 7,
    'Radish': 5,
    'Red Onion': 30
}

# Crop-Market Mapping: Only these markets are valid for each crop
CROP_MARKETS = {
    'Rice': [
        'Colombo',
        'Anuradhapura',
        'Moneragala',
        'Dambulla',
        'Ampara',
        'Kandy',
        'Kurunegala',
        'Polonnaruwa'
    ],
    'Beetroot': [
        'Colombo',
        'Thambuththegama',
        'Bandarawela',
        'Dambulla',
        'Kandy',
        'Nuwara Eliya'  # Standardized spelling
    ],
    'Radish': [
        'Colombo',
        'Moneragala',
        'Dambulla',
        'Kandy',
        'Meegoda'
    ],
    'Red Onion': [
        'Colombo',
        'Puttalam',
        'Mullaittivu',  # Standardized from Mulathiv
        'Vavuniya',
        'Batticaloa',
        'Dambulla',
        'Embilipitiya',
        'Jaffna',
        'Kandy',
        'Mannar',
        'Meegoda',
        'Moneragala',
        'Nuwara Eliya',  # Standardized spelling
        'Thambuththegama',
        'Trincomalee'
    ]
}

# Get all unique markets across all crops
ALL_MARKETS = sorted(set(m for markets in CROP_MARKETS.values() for m in markets))


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
    'humidity_percent': 75.0,
    'wind_speed': 13.5,
    'sunshine_hours': 10.5
}

# Sri Lankan Agricultural Seasons
SEASONS = {
    'Maha': {'months': [10, 11, 12, 1, 2, 3], 'name_si': 'මහ', 'description': 'Main cultivation season (Oct-Mar)'},
    'Yala': {'months': [4, 5, 6, 7, 8, 9], 'name_si': 'යල', 'description': 'Secondary season (Apr-Sep)'}
}

# Major Sri Lankan Festivals (approximate dates - vary by lunar calendar)
FESTIVALS = {
    'Sinhala New Year': {'month': 4, 'day': 14, 'impact': 'high', 'name_si': 'අලුත් අවුරුද්ද'},
    'Vesak': {'month': 5, 'day': 15, 'impact': 'high', 'name_si': 'වෙසක්'},
    'Poson': {'month': 6, 'day': 15, 'impact': 'medium', 'name_si': 'පොසොන්'},
    'Esala': {'month': 7, 'day': 15, 'impact': 'medium', 'name_si': 'ඇසළ'},
    'Christmas': {'month': 12, 'day': 25, 'impact': 'high', 'name_si': 'නත්තල'},
    'Thai Pongal': {'month': 1, 'day': 14, 'impact': 'medium', 'name_si': 'තෛ පොංගල්'}
}

# Harvest Periods by Crop
HARVEST_PERIODS = {
    'Rice': [3, 4, 8, 9],  # March-April (Maha), Aug-Sep (Yala)
    'Beetroot': [7, 8, 9],
    'Radish': [6, 7, 8],
    'Red Onion': [6, 7, 8]
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
