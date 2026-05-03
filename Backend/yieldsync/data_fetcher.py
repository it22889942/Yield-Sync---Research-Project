# """
# YieldSync Weekly Data Fetcher
# =============================
# Automated data fetching for weekly model updates.
# Combines HARTI price bulletins + Open-Meteo weather data.

# Features:
# - Fetches HARTI weekly price bulletins (PDF parsing)
# - Fetches Open-Meteo weather data (free API)
# - Fallback to last week's prices if bulletin not available
# - Auto-updates full historical dataset

# Usage:
#     from data_fetcher import update_this_week, update_since_date
    
#     # Update current week
#     update_this_week()
    
#     # Catch up from a specific date
#     update_since_date('2026-01-01')

# Dependencies:
#     pip install requests pdfplumber pandas
# """

# import os
# import requests
# from datetime import datetime, timedelta
# from io import BytesIO

# import pandas as pd
# import numpy as np

# # =============================================================================
# # PDF LIBRARY
# # =============================================================================
# try:
#     import pdfplumber
#     HAS_PDF = True
# except ImportError:
#     HAS_PDF = False
#     print("Warning: pdfplumber not installed. Run: pip install pdfplumber")


# # =============================================================================
# # CONFIGURATION
# # =============================================================================

# # HARTI PDF Source
# HARTI_BASE_URL = "https://www.harti.gov.lk/images/download/market_information"
# HARTI_PAGES = [11, 12, 14]  # Rice, Red Onion, Vegetables

# # Open-Meteo API (free, no API key needed)
# OPEN_METEO_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
# OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# # Market coordinates for weather data
# MARKET_COORDINATES = {
#     'Colombo': (6.9271, 79.8612),
#     'Dambulla': (7.8675, 80.6519),
#     'Kandy': (7.2906, 80.6337),
#     'Meegoda': (6.8044, 80.0639),
#     'Moneragala': (6.8722, 81.3489),
#     'Kurunegala': (7.4867, 80.3647),
#     'Anuradhapura': (8.3114, 80.4037),
#     'Polonnaruwa': (7.9403, 81.0188),
#     'Ampara': (7.2983, 81.6747),
#     'Bandarawela': (6.8304, 80.9910),
#     'Nuwara Eliya': (6.9497, 80.7891),
#     'Thambuththegama': (8.1500, 80.3000),
#     'Jaffna': (9.6615, 80.0255),
#     'Vavuniya': (8.7514, 80.4971),
#     'Batticaloa': (7.7310, 81.6747),
#     'Puttalam': (8.0362, 79.8283),
#     'Trincomalee': (8.5874, 81.2152),
#     'Embilipitiya': (6.3333, 80.8500),
#     'Mullaittivu': (9.2675, 80.8128),
#     'Mannar': (8.9810, 79.9044)
# }

# # Weather variables to fetch
# WEATHER_VARS = [
#     'temperature_2m_mean',
#     'precipitation_sum',
#     'relative_humidity_2m_max',
#     'wind_speed_10m_max',
#     'shortwave_radiation_sum'
# ]


# # =============================================================================
# # HARTI PDF PARSING
# # =============================================================================

# def download_harti_pdf(year: int, week: int) -> bytes:
#     """Download HARTI weekly bulletin PDF."""
#     url = f"{HARTI_BASE_URL}/{year}/weekly/weekly_{week:02d}_{year}_Eng.pdf"
#     print(f"📥 Downloading: {url.split('/')[-1]}")
    
#     try:
#         response = requests.get(url, timeout=30, headers={
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
#         })
#         response.raise_for_status()
#         return response.content
#     except Exception as e:
#         print(f"   ❌ Download failed: {e}")
#         return None


# def parse_price_range(price_str: str) -> float:
#     """Parse price string like '450-650' to average."""
#     if not price_str or price_str in ('−', '-', ''):
#         return None
#     try:
#         price_str = str(price_str).replace('−', '-').replace('–', '-')
#         if '-' in price_str:
#             parts = price_str.split('-')
#             return (float(parts[0].strip()) + float(parts[1].strip())) / 2
#         return float(price_str.strip())
#     except:
#         return None


# def extract_prices_from_pdf(pdf_bytes: bytes, year: int, week: int) -> pd.DataFrame:
#     """Extract prices from HARTI PDF."""
#     if not HAS_PDF:
#         return pd.DataFrame()
    
#     week_start = datetime.strptime(f'{year}-W{week:02d}-1', '%Y-W%W-%w')
#     results = []
    
#     with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
#         for page_num in HARTI_PAGES:
#             if page_num > len(pdf.pages):
#                 continue
            
#             tables = pdf.pages[page_num - 1].extract_tables()
#             for table in tables:
#                 if len(table) < 2:
#                     continue
                
#                 header = table[0] if table[0] else []
                
#                 # Find crop columns
#                 crop_cols = {}
#                 for i, col in enumerate(header):
#                     if not col:
#                         continue
#                     col_lower = str(col).lower()
#                     if 'samba' in col_lower or 'nadu' in col_lower:
#                         crop_cols[i] = 'Rice'
#                     elif 'red onion' in col_lower:
#                         crop_cols[i] = 'Red Onion'
#                     elif 'beetroot' in col_lower:
#                         crop_cols[i] = 'Beetroot'
#                     elif 'radish' in col_lower or 'raddish' in col_lower:
#                         crop_cols[i] = 'Radish'
                
#                 # Parse data rows
#                 for row in table[1:]:
#                     if not row or not row[0]:
#                         continue
                    
#                     locations = [loc.strip() for loc in str(row[0]).split('\n') if loc.strip()]
                    
#                     for col_idx, crop in crop_cols.items():
#                         if col_idx >= len(row):
#                             continue
                        
#                         prices = [parse_price_range(p.strip()) 
#                                  for p in str(row[col_idx]).split('\n')]
                        
#                         for loc, price in zip(locations, prices):
#                             if price is not None:
#                                 # Red Onion is per 100kg, convert
#                                 if crop == 'Red Onion':
#                                     price = price / 100
                                
#                                 results.append({
#                                     'Date': week_start,
#                                     'item': crop,
#                                     'market': loc.strip(),
#                                     'price': price,
#                                     'year': year,
#                                     'week': week
#                                 })
    
#     return pd.DataFrame(results)


# # =============================================================================
# # WEATHER DATA
# # =============================================================================

# def fetch_weather(year: int, week: int, markets: list = None) -> pd.DataFrame:
#     """Fetch weather data for a week."""
#     week_start = datetime.strptime(f'{year}-W{week:02d}-1', '%Y-W%W-%w')
#     week_end = week_start + timedelta(days=6)
    
#     # Use historical API if data is old
#     days_ago = (datetime.now() - week_end).days
#     api_url = OPEN_METEO_HISTORICAL_URL if days_ago > 5 else OPEN_METEO_FORECAST_URL
    
#     if markets is None:
#         markets = list(MARKET_COORDINATES.keys())
    
#     results = []
    
#     for market in markets:
#         if market not in MARKET_COORDINATES:
#             continue
        
#         lat, lon = MARKET_COORDINATES[market]
        
#         try:
#             response = requests.get(api_url, params={
#                 'latitude': lat,
#                 'longitude': lon,
#                 'start_date': week_start.strftime('%Y-%m-%d'),
#                 'end_date': week_end.strftime('%Y-%m-%d'),
#                 'daily': ','.join(WEATHER_VARS),
#                 'timezone': 'Asia/Colombo'
#             }, timeout=30)
            
#             data = response.json()
            
#             if 'daily' in data:
#                 daily = data['daily']
#                 # Aggregate to weekly
#                 results.append({
#                     'market': market,
#                     'temp': np.mean(daily.get('temperature_2m_mean', [27])),
#                     'rainfall': np.sum(daily.get('precipitation_sum', [0])),
#                     'humidity': np.max(daily.get('relative_humidity_2m_max', [75])),
#                     'wind_speed': np.max(daily.get('wind_speed_10m_max', [10])) / 3.6,
#                     'sunshine_hours': np.sum(daily.get('shortwave_radiation_sum', [20])) / 3.6 / 7
#                 })
#         except Exception as e:
#             print(f"   ⚠️ Weather error for {market}: {e}")
    
#     return pd.DataFrame(results)


# # =============================================================================
# # COMBINED FETCH
# # =============================================================================

# def fetch_week(year: int, week: int, last_week_prices: pd.DataFrame = None) -> pd.DataFrame:
#     """
#     Fetch combined price + weather data for a week.
#     Uses last week's prices as fallback if PDF not available.
    
#     Returns:
#         DataFrame with columns matching historical format
#     """
#     week_start = datetime.strptime(f'{year}-W{week:02d}-1', '%Y-W%W-%w')
    
#     print(f"\n{'='*50}")
#     print(f"📅 Fetching Week {week}, {year} ({week_start.date()})")
#     print('='*50)
    
#     # 1. Get prices
#     pdf_bytes = download_harti_pdf(year, week)
    
#     if pdf_bytes:
#         prices_df = extract_prices_from_pdf(pdf_bytes, year, week)
#         print(f"   ✅ Extracted {len(prices_df)} price records")
#     else:
#         prices_df = pd.DataFrame()
    
#     # Fallback to last week
#     if prices_df.empty and last_week_prices is not None:
#         print("   ⚠️ Using last week's prices as fallback")
#         prices_df = last_week_prices.copy()
#         prices_df['Date'] = week_start
#         prices_df['year'] = year
#         prices_df['week'] = week
    
#     if prices_df.empty:
#         print("   ❌ No price data available")
#         return pd.DataFrame()
    
#     # 2. Get weather
#     markets = prices_df['market'].unique().tolist()
#     weather_df = fetch_weather(year, week, markets)
#     print(f"   ✅ Weather data for {len(weather_df)} markets")
    
#     # 3. Combine
#     if not weather_df.empty:
#         combined = prices_df.merge(weather_df, on='market', how='left')
#     else:
#         combined = prices_df
#         combined['temp'] = 27.5
#         combined['rainfall'] = 5.0
#         combined['humidity'] = 75.0
#         combined['wind_speed'] = 10.0
#         combined['sunshine_hours'] = 5.0
    
#     # Add required columns
#     combined['is_holiday'] = 0.0
#     combined['holiday_name'] = np.nan
#     combined['volume_MT'] = np.nan
#     combined['is_public_holiday'] = np.nan
#     combined['demand_multiplier'] = np.nan
#     combined['quantity_tonnes'] = np.nan
    
#     # Aggregate rice varieties
#     combined = combined.groupby(['Date', 'market', 'item']).agg({
#         'price': 'mean',
#         'temp': 'first',
#         'rainfall': 'first',
#         'humidity': 'first',
#         'wind_speed': 'first',
#         'sunshine_hours': 'first',
#         'is_holiday': 'first',
#         'holiday_name': 'first',
#         'volume_MT': 'first',
#         'is_public_holiday': 'first',
#         'demand_multiplier': 'first',
#         'quantity_tonnes': 'first'
#     }).reset_index()
    
#     print(f"   ✅ Final: {len(combined)} records")
#     return combined


# def update_dataset(new_data: pd.DataFrame, data_path: str):
#     """Append new data to existing dataset, avoiding duplicates."""
#     if new_data.empty:
#         return
    
#     new_data['Date'] = pd.to_datetime(new_data['Date'])
    
#     if os.path.exists(data_path):
#         existing = pd.read_csv(data_path, parse_dates=['Date'])
        
#         # Remove existing data for new dates
#         new_dates = new_data['Date'].dt.date.unique()
#         existing = existing[~existing['Date'].dt.date.isin(new_dates)]
        
#         # Append
#         combined = pd.concat([existing, new_data], ignore_index=True)
#         combined = combined.sort_values(['Date', 'item', 'market']).reset_index(drop=True)
#         combined.to_csv(data_path, index=False)
        
#         print(f"\n✅ Updated {data_path}")
#         print(f"   Total records: {len(combined)}")
#         print(f"   Date range: {combined['Date'].min().date()} to {combined['Date'].max().date()}")
#     else:
#         new_data.to_csv(data_path, index=False)
#         print(f"\n✅ Created {data_path} with {len(new_data)} records")


# # =============================================================================
# # CONVENIENCE FUNCTIONS
# # =============================================================================

# def update_this_week(data_path: str = None):
#     """Fetch and update data for the current week."""
#     if data_path is None:
#         data_path = os.path.join(os.path.dirname(__file__), 'data', 
#                                  'full_history_features_real_weather.csv')
    
#     today = datetime.now()
#     year = today.year
#     week = today.isocalendar()[1]
    
#     data = fetch_week(year, week)
#     if not data.empty:
#         update_dataset(data, data_path)


# def update_since_date(start_date: str, data_path: str = None):
#     """
#     Fetch and update all weeks since a given date.
    
#     Args:
#         start_date: Start date in 'YYYY-MM-DD' format
#         data_path: Path to dataset file
#     """
#     if data_path is None:
#         data_path = os.path.join(os.path.dirname(__file__), 'data',
#                                  'full_history_features_real_weather.csv')
    
#     start = datetime.strptime(start_date, '%Y-%m-%d')
#     end = datetime.now()
    
#     all_data = []
#     last_prices = None
    
#     current = start
#     while current <= end:
#         year = current.year
#         week = current.isocalendar()[1]
        
#         data = fetch_week(year, week, last_prices)
        
#         if not data.empty:
#             all_data.append(data)
#             last_prices = data[['market', 'item', 'price']].copy()
        
#         current += timedelta(weeks=1)
    
#     if all_data:
#         combined = pd.concat(all_data, ignore_index=True)
#         update_dataset(combined, data_path)
#         print(f"\n🎉 Updated {len(all_data)} weeks of data")


# # =============================================================================
# # MAIN
# # =============================================================================

# if __name__ == '__main__':
#     print("YieldSync Data Fetcher")
#     print("=" * 40)
#     print("\nUsage:")
#     print("  from data_fetcher import update_this_week, update_since_date")
#     print("  update_this_week()  # Fetch current week")
#     print("  update_since_date('2026-01-01')  # Catch up")
"""
YieldSync Weekly Data Fetcher
=============================
Automated data fetching for weekly model updates.
Combines HARTI price bulletins + Open-Meteo weather data.

Features:
- Fetches HARTI weekly price bulletins (PDF parsing)
- Fetches Open-Meteo weather data (free API)
- Fallback to last week's prices if bulletin not available
- Auto-updates full historical dataset

Usage:
    from data_fetcher import update_this_week, update_since_date
    
    # Update current week
    update_this_week()
    
    # Catch up from a specific date
    update_since_date('2026-01-01')

Dependencies:
    pip install requests pdfplumber pandas
"""

import os
import requests
from datetime import datetime, timedelta
from io import BytesIO

import pandas as pd
import numpy as np

# =============================================================================
# PDF LIBRARY
# =============================================================================
try:
    import pdfplumber
    HAS_PDF = True
except ImportError:
    HAS_PDF = False
    print("Warning: pdfplumber not installed. Run: pip install pdfplumber")


# =============================================================================
# CONFIGURATION
# =============================================================================

# HARTI PDF Source
HARTI_BASE_URL = "https://www.harti.gov.lk/images/download/market_information"
HARTI_PAGES = [11, 12, 14]  # Rice, Red Onion, Vegetables

# Open-Meteo API (free, no API key needed)
OPEN_METEO_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Market coordinates for weather data
MARKET_COORDINATES = {
    'Colombo': (6.9271, 79.8612),
    'Dambulla': (7.8675, 80.6519),
    'Kandy': (7.2906, 80.6337),
    'Meegoda': (6.8044, 80.0639),
    'Moneragala': (6.8722, 81.3489),
    'Kurunegala': (7.4867, 80.3647),
    'Anuradhapura': (8.3114, 80.4037),
    'Polonnaruwa': (7.9403, 81.0188),
    'Ampara': (7.2983, 81.6747),
    'Bandarawela': (6.8304, 80.9910),
    'Nuwara Eliya': (6.9497, 80.7891),
    'Thambuththegama': (8.1500, 80.3000),
    'Jaffna': (9.6615, 80.0255),
    'Vavuniya': (8.7514, 80.4971),
    'Batticaloa': (7.7310, 81.6747),
    'Puttalam': (8.0362, 79.8283),
    'Trincomalee': (8.5874, 81.2152),
    'Embilipitiya': (6.3333, 80.8500),
    'Mullaittivu': (9.2675, 80.8128),
    'Mannar': (8.9810, 79.9044)
}

# Weather variables to fetch
WEATHER_VARS = [
    'temperature_2m_mean',
    'precipitation_sum',
    'relative_humidity_2m_max',
    'wind_speed_10m_max',
    'shortwave_radiation_sum'
]


# =============================================================================
# HARTI PDF PARSING
# =============================================================================

def download_harti_pdf(year: int, week: int) -> bytes:
    """Download HARTI weekly bulletin PDF."""
    url = f"{HARTI_BASE_URL}/{year}/weekly/weekly_{week:02d}_{year}_Eng.pdf"
    print(f"📥 Downloading: {url.split('/')[-1]}")
    
    try:
        response = requests.get(url, timeout=30, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        })
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        return None


def parse_price_range(price_str: str) -> float:
    """Parse price string like '450-650' to average."""
    if not price_str or price_str in ('−', '-', ''):
        return None
    try:
        price_str = str(price_str).replace('−', '-').replace('–', '-')
        if '-' in price_str:
            parts = price_str.split('-')
            return (float(parts[0].strip()) + float(parts[1].strip())) / 2
        return float(price_str.strip())
    except:
        return None


def extract_prices_from_pdf(pdf_bytes: bytes, year: int, week: int) -> pd.DataFrame:
    """Extract prices from HARTI PDF."""
    if not HAS_PDF:
        return pd.DataFrame()
    
    week_start = datetime.strptime(f'{year}-W{week:02d}-1', '%Y-W%W-%w')
    results = []
    
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page_num in HARTI_PAGES:
            if page_num > len(pdf.pages):
                continue
            
            tables = pdf.pages[page_num - 1].extract_tables()
            for table in tables:
                if len(table) < 2:
                    continue
                
                header = table[0] if table[0] else []
                
                # Find crop columns
                crop_cols = {}
                for i, col in enumerate(header):
                    if not col:
                        continue
                    col_lower = str(col).lower()
                    if 'samba' in col_lower or 'nadu' in col_lower:
                        crop_cols[i] = 'Rice'
                    elif 'red onion' in col_lower:
                        crop_cols[i] = 'Red Onion'
                    elif 'beetroot' in col_lower:
                        crop_cols[i] = 'Beetroot'
                    elif 'radish' in col_lower or 'raddish' in col_lower:
                        crop_cols[i] = 'Radish'
                
                # Parse data rows
                for row in table[1:]:
                    if not row or not row[0]:
                        continue
                    
                    locations = [loc.strip() for loc in str(row[0]).split('\n') if loc.strip()]
                    
                    for col_idx, crop in crop_cols.items():
                        if col_idx >= len(row):
                            continue
                        
                        prices = [parse_price_range(p.strip()) 
                                 for p in str(row[col_idx]).split('\n')]
                        
                        for loc, price in zip(locations, prices):
                            if price is not None:
                                # Red Onion is per 100kg, convert
                                if crop == 'Red Onion':
                                    price = price / 100
                                
                                results.append({
                                    'Date': week_start,
                                    'item': crop,
                                    'market': loc.strip(),
                                    'price': price,
                                    'year': year,
                                    'week': week
                                })
    
    return pd.DataFrame(results)


# =============================================================================
# WEATHER DATA
# =============================================================================

def fetch_weather(year: int, week: int, markets: list = None) -> pd.DataFrame:
    """Fetch weather data for a week."""
    week_start = datetime.strptime(f'{year}-W{week:02d}-1', '%Y-W%W-%w')
    week_end = week_start + timedelta(days=6)
    
    # Use historical API if data is old
    days_ago = (datetime.now() - week_end).days
    api_url = OPEN_METEO_HISTORICAL_URL if days_ago > 5 else OPEN_METEO_FORECAST_URL
    
    if markets is None:
        markets = list(MARKET_COORDINATES.keys())
    
    results = []
    
    for market in markets:
        if market not in MARKET_COORDINATES:
            continue
        
        lat, lon = MARKET_COORDINATES[market]
        
        try:
            response = requests.get(api_url, params={
                'latitude': lat,
                'longitude': lon,
                'start_date': week_start.strftime('%Y-%m-%d'),
                'end_date': week_end.strftime('%Y-%m-%d'),
                'daily': ','.join(WEATHER_VARS),
                'timezone': 'Asia/Colombo'
            }, timeout=30)
            
            data = response.json()
            
            if 'daily' in data:
                daily = data['daily']
                # Aggregate to weekly
                results.append({
                    'market': market,
                    'temp': np.mean(daily.get('temperature_2m_mean', [27])),
                    'rainfall': np.sum(daily.get('precipitation_sum', [0])),
                    'humidity': np.max(daily.get('relative_humidity_2m_max', [75])),
                    'wind_speed': np.max(daily.get('wind_speed_10m_max', [10])) / 3.6,
                    'sunshine_hours': np.sum(daily.get('shortwave_radiation_sum', [20])) / 3.6 / 7
                })
        except Exception as e:
            print(f"   ⚠️ Weather error for {market}: {e}")
    
    return pd.DataFrame(results)


# =============================================================================
# COMBINED FETCH
# =============================================================================

def fetch_week(year: int, week: int, last_week_prices: pd.DataFrame = None) -> pd.DataFrame:
    """
    Fetch combined price + weather data for a week.
    Uses last week's prices as fallback if PDF not available.
    
    Returns:
        DataFrame with columns matching historical format
    """
    week_start = datetime.strptime(f'{year}-W{week:02d}-1', '%Y-W%W-%w')
    
    print(f"\n{'='*50}")
    print(f"📅 Fetching Week {week}, {year} ({week_start.date()})")
    print('='*50)
    
    # 1. Get prices
    pdf_bytes = download_harti_pdf(year, week)
    
    if pdf_bytes:
        prices_df = extract_prices_from_pdf(pdf_bytes, year, week)
        print(f"   ✅ Extracted {len(prices_df)} price records")
    else:
        prices_df = pd.DataFrame()
    
    # Fallback to last week
    if prices_df.empty and last_week_prices is not None:
        print("   ⚠️ Using last week's prices as fallback")
        prices_df = last_week_prices.copy()
        prices_df['Date'] = week_start
        prices_df['year'] = year
        prices_df['week'] = week
    
    if prices_df.empty:
        print("   ❌ No price data available")
        return pd.DataFrame()
    
    # 2. Get weather
    markets = prices_df['market'].unique().tolist()
    weather_df = fetch_weather(year, week, markets)
    print(f"   ✅ Weather data for {len(weather_df)} markets")
    
    # 3. Combine
    if not weather_df.empty:
        combined = prices_df.merge(weather_df, on='market', how='left')
    else:
        combined = prices_df
        combined['temp'] = 27.5
        combined['rainfall'] = 5.0
        combined['humidity'] = 75.0
        combined['wind_speed'] = 10.0
        combined['sunshine_hours'] = 5.0
    
    # Add required columns
    combined['is_holiday'] = 0.0
    combined['holiday_name'] = np.nan
    combined['volume_MT'] = np.nan
    combined['is_public_holiday'] = np.nan
    combined['demand_multiplier'] = np.nan
    combined['quantity_tonnes'] = np.nan
    
    # Aggregate rice varieties
    combined = combined.groupby(['Date', 'market', 'item']).agg({
        'price': 'mean',
        'temp': 'first',
        'rainfall': 'first',
        'humidity': 'first',
        'wind_speed': 'first',
        'sunshine_hours': 'first',
        'is_holiday': 'first',
        'holiday_name': 'first',
        'volume_MT': 'first',
        'is_public_holiday': 'first',
        'demand_multiplier': 'first',
        'quantity_tonnes': 'first'
    }).reset_index()
    
    print(f"   ✅ Final: {len(combined)} records")
    return combined


def update_dataset(new_data: pd.DataFrame, data_path: str):
    """Append new data to existing dataset, avoiding duplicates."""
    if new_data.empty:
        return
    
    new_data['Date'] = pd.to_datetime(new_data['Date'])
    
    if os.path.exists(data_path):
        existing = pd.read_csv(data_path, parse_dates=['Date'])
        
        # Remove existing data for new dates
        new_dates = new_data['Date'].dt.date.unique()
        existing = existing[~existing['Date'].dt.date.isin(new_dates)]
        
        # Append
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined = combined.sort_values(['Date', 'item', 'market']).reset_index(drop=True)
        combined.to_csv(data_path, index=False)
        
        print(f"\n✅ Updated {data_path}")
        print(f"   Total records: {len(combined)}")
        print(f"   Date range: {combined['Date'].min().date()} to {combined['Date'].max().date()}")
    else:
        new_data.to_csv(data_path, index=False)
        print(f"\n✅ Created {data_path} with {len(new_data)} records")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def update_this_week(data_path: str = None):
    """Fetch and update data for the current week."""
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), 'data', 
                                 'full_history_features_real_weather.csv')
    
    today = datetime.now()
    year = today.year
    week = today.isocalendar()[1]
    
    data = fetch_week(year, week)
    if not data.empty:
        update_dataset(data, data_path)


def update_since_date(start_date: str, data_path: str = None):
    """
    Fetch and update all weeks since a given date.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        data_path: Path to dataset file
    """
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), 'data',
                                 'full_history_features_real_weather.csv')
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.now()
    
    all_data = []
    last_prices = None
    
    current = start
    while current <= end:
        year = current.year
        week = current.isocalendar()[1]
        
        data = fetch_week(year, week, last_prices)
        
        if not data.empty:
            all_data.append(data)
            last_prices = data[['market', 'item', 'price']].copy()
        
        current += timedelta(weeks=1)
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        update_dataset(combined, data_path)
        print(f"\n🎉 Updated {len(all_data)} weeks of data")


def update_complete_weeks_only(start_date: str = None, data_path: str = None):
    """
    Fetch and update ONLY complete weeks (excludes current incomplete week).
    
    This is the SMART updater - it only fetches weeks that have ended,
    avoiding incomplete data from the current week.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format (optional, auto-detects)
        data_path: Path to dataset file
    
    Returns:
        dict with status info
    """
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), 'data',
                                 'full_history_features_real_weather.csv')
    
    today = datetime.now()
    
    # Calculate the end of last complete week (last Sunday)
    days_since_monday = today.weekday()  # Monday=0, Sunday=6
    if days_since_monday == 6:  # Today is Sunday
        last_sunday = today
    else:
        # Go back to last Sunday
        last_sunday = today - timedelta(days=days_since_monday + 1)
    
    # Start date logic
    if start_date is None:
        # Auto-detect: check what's the last date in existing data
        if os.path.exists(data_path):
            try:
                existing = pd.read_csv(data_path, parse_dates=['Date'])
                last_date = existing['Date'].max()
                # Start from the week after last date
                start = last_date + timedelta(days=7)
                print(f"📊 Last data: {last_date.date()}, starting from: {start.date()}")
            except Exception:
                # Default to 4 weeks ago if can't read
                start = today - timedelta(weeks=4)
        else:
            # No existing file, start from 4 weeks ago
            start = today - timedelta(weeks=4)
    else:
        start = datetime.strptime(start_date, '%Y-%m-%d')
    
    # Only fetch weeks up to last Sunday (last complete week)
    if start > last_sunday:
        return {
            "status": "no_update_needed",
            "message": "No complete weeks to fetch",
            "last_complete_week": str(last_sunday.date()),
            "today": str(today.date())
        }
    
    print(f"\n{'='*60}")
    print(f"📅 SMART UPDATE: Fetching COMPLETE weeks only")
    print(f"   From: {start.date()}")
    print(f"   To: {last_sunday.date()} (last complete week)")
    print(f"   Today: {today.date()} (week {today.isocalendar()[1]} is INCOMPLETE)")
    print('='*60)
    
    all_data = []
    last_prices = None
    weeks_fetched = 0

    # Seed fallback prices from existing dataset so a missing weekly PDF
    # can still be filled using the latest known market/item prices.
    if os.path.exists(data_path):
        try:
            existing = pd.read_csv(data_path, parse_dates=['Date'])
            if not existing.empty and {"Date", "market", "item", "price"}.issubset(existing.columns):
                latest_date = existing["Date"].max()
                snap = existing[existing["Date"] == latest_date][["market", "item", "price"]].copy()
                if not snap.empty:
                    last_prices = snap
                    print(f"📌 Seeded fallback prices from existing data: {latest_date.date()}")
        except Exception as e:
            print(f"⚠️ Could not seed fallback prices from existing data: {e}")
    
    current = start
    while current <= last_sunday:
        year = current.year
        week = current.isocalendar()[1]
        
        data = fetch_week(year, week, last_prices)
        
        if not data.empty:
            all_data.append(data)
            last_prices = data[['market', 'item', 'price']].copy()
            weeks_fetched += 1
        
        current += timedelta(weeks=1)
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        update_dataset(combined, data_path)
        
        result = {
            "status": "success",
            "weeks_fetched": weeks_fetched,
            "records_added": len(combined),
            "date_range": {
                "from": str(combined['Date'].min().date()),
                "to": str(combined['Date'].max().date())
            },
            "last_complete_week": str(last_sunday.date()),
            "today": str(today.date()),
            "message": f"Successfully updated {weeks_fetched} complete weeks"
        }
        
        print(f"\n🎉 {result['message']}")
        print(f"   Added {result['records_added']} records")
        print(f"   Range: {result['date_range']['from']} to {result['date_range']['to']}")
        
        return result
    else:
        return {
            "status": "no_data",
            "message": "No data fetched (all weeks may have failed)",
            "last_complete_week": str(last_sunday.date()),
            "today": str(today.date())
        }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("YieldSync Data Fetcher")
    print("=" * 40)
    print("\n⭐ RECOMMENDED Usage (Smart Update):")
    print("  from data_fetcher import update_complete_weeks_only")
    print("  update_complete_weeks_only()  # Only fetches COMPLETE weeks")
    print("\nOther Usage:")
    print("  from data_fetcher import update_this_week, update_since_date")
    print("  update_this_week()  # Fetch current week (may be incomplete)")
    print("  update_since_date('2026-01-01')  # Catch up from date")
    print("\nVia API:")
    print("  POST /api/yieldsync/data/update-smart  # ⭐ Recommended")
