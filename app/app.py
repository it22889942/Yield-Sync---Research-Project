"""
🌾 YieldSync - Smart Farming Decisions
Streamlit Web Application with Daily Data Entry

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys

# Ensure app directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predictor import YieldSyncPredictor
from config import (
    TARGET_CROPS, HORIZONS, PERISHABILITY,
    CROP_NAMES_SI, TRANSLATIONS, DEFAULT_WEATHER
)

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="YieldSync - Smart Farming",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #2E7D32; text-align: center; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.2rem; color: #666; text-align: center; margin-bottom: 2rem; }
    .decision-sell { background-color: #ffebee; border-left: 4px solid #f44336; padding: 1rem; border-radius: 0.5rem; }
    .decision-hold { background-color: #e8f5e9; border-left: 4px solid #4caf50; padding: 1rem; border-radius: 0.5rem; }
    .decision-wait { background-color: #fff3e0; border-left: 4px solid #ff9800; padding: 1rem; border-radius: 0.5rem; }
    .next-date-box { background-color: #e3f2fd; border: 2px solid #1976D2; padding: 1rem; border-radius: 0.5rem; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA PATH
# ============================================================================
# Use robust pathing
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
# Main historical data file (Price + Weather)
DATA_PATH = os.path.join(DATA_DIR, 'full_history_features_real_weather.csv')
# Demand data file (Volume)
DEMAND_DATA_PATH = os.path.join(DATA_DIR, 'full_history_demand_data.csv')

# ============================================================================
# LOAD DATA
# ============================================================================
@st.cache_data
def load_data():
    """Load market data from CSV, merging Price/Weather with Volume."""
    if os.path.exists(DATA_PATH):
        try:
            # 1. Load Main Data (Price + Weather)
            df = pd.read_csv(DATA_PATH)
            df['Date'] = pd.to_datetime(df['Date'])
            
            # 2. Load Volume Data if available
            if os.path.exists(DEMAND_DATA_PATH):
                try:
                    df_vol = pd.read_csv(DEMAND_DATA_PATH)
                    # Keep only keys and quantity
                    if 'quantity_tonnes' in df_vol.columns:
                         vol_cols = ['Date', 'market', 'item', 'quantity_tonnes']
                         # Filter to exist columns
                         vol_cols = [c for c in vol_cols if c in df_vol.columns]
                         df_vol = df_vol[vol_cols]
                         df_vol['Date'] = pd.to_datetime(df_vol['Date'])
                         
                         # Merge
                         df = pd.merge(df, df_vol, on=['Date', 'market', 'item'], how='left')
                except Exception as e:
                    st.warning(f"Could not load extra volume data: {e}")

            # Map column names if needed
            if 'quantity_tonnes' in df.columns:
                df = df.rename(columns={'quantity_tonnes': 'volume_MT'})
            
            # Fallback if volume still missing (e.g. merge failed or file missing)
            if 'volume_MT' not in df.columns:
                 # Create dummy volume for app stability if absolutely necessary, 
                 # or let it fail but with better message? 
                 # Better to fill with default to prevent crash
                 df['volume_MT'] = 10.0 

            return df
        except Exception as e:
            st.error(f"Error reading data: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def save_data(df):
    """Save data back to CSV."""
    # Backup first? Optional but good practice.
    # df.to_csv(DATA_PATH + ".bak", index=False)
    df.to_csv(DATA_PATH, index=False)
    # Clear cache to reload data
    load_data.clear()

def get_next_entry_date(df):
    """Get the next date that should be entered (day after last entry)."""
    if df.empty:
        return datetime(2025, 1, 1).date()
    last_date = df['Date'].max()
    next_date = last_date + timedelta(days=1)
    return next_date.date()

def get_last_entry_date(df):
    """Get the last date in the dataset."""
    if df.empty:
        return None
    return df['Date'].max().date()

# ============================================================================
# LOAD PREDICTOR
# ============================================================================
@st.cache_resource
def load_predictor():
    # Models are in models/saved_models/ directory (matching notebooks)
    # YieldSyncPredictor will auto-detect the path
    return YieldSyncPredictor()

try:
    predictor = load_predictor()
except Exception as e:
    st.error(f"Failed to load predictor: {e}")
    st.stop()

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Language
    language = st.radio(
        "Language / භාෂාව",
        options=['en', 'si'],
        format_func=lambda x: "English" if x == 'en' else "සිංහල"
    )
    lang = TRANSLATIONS[language]
    
    st.markdown("---")
    
    # Mode selection
    mode = st.radio(
        "Mode",
        options=['📊 Get Prediction', '📝 Add Daily Data', '📈 View Data', '🔄 Retrain Models', '⚙️ Settings'],
        index=0
    )
    
    st.info("Yield Sync v2.0")

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Header
st.markdown(f"<div class='main-header'>{lang['title']}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='sub-header'>{lang['subtitle']}</div>", unsafe_allow_html=True)

# Load data
df = load_data()

# ============================================================================
# MODE: ADD DAILY DATA
# ============================================================================
if mode == '📝 Add Daily Data':
    st.subheader("📝 Add New Daily Market Data")
    
    # Get the next date to add
    next_date = get_next_entry_date(df)
    last_date = get_last_entry_date(df)
    
    # Show status
    if last_date:
        current_sys_date = datetime.now().date()
        days_lag = (current_sys_date - last_date).days
        
        status_color = "#4caf50" if days_lag <= 0 else "#f44336"
        status_text = "✅ Up to Date" if days_lag <= 0 else f"⚠️ Missing {days_lag} Days of Data"
        
        st.markdown(f"""
        <div class='next-date-box'>
            <h3>📅 Data Status</h3>
            <div style="display: flex; justify-content: space-around; align-items: center; margin-bottom: 10px;">
                <div>
                    <p style="margin: 0; color: #666; font-size: 0.9rem;">Real-World Date</p>
                    <p style="margin: 0; font-weight: bold; font-size: 1.1rem;">{current_sys_date.strftime('%Y-%m-%d')}</p>
                </div>
                <div style="font-size: 2rem;">👉</div>
                <div>
                    <p style="margin: 0; color: #666; font-size: 0.9rem;">Last Data Entry</p>
                    <p style="margin: 0; font-weight: bold; font-size: 1.1rem;">{last_date.strftime('%Y-%m-%d')}</p>
                </div>
            </div>
            <div style="background-color: {status_color}; color: white; padding: 5px; border-radius: 5px; font-weight: bold; margin-top: 5px;">
                {status_text}
            </div>
            <p style="margin-top: 10px;"><strong>Next date to add:</strong> <span style='color: #1976D2; font-size: 1.5rem; font-weight: bold;'>{next_date.strftime('%Y-%m-%d')}</span></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("📅 No data yet. Starting from 2025-01-01")
    
    st.markdown("---")
    
    # Tabs for Manual vs Bulk
    tab1, tab2 = st.tabs(["✍️ Manual Entry", "📂 Bulk Upload (Automated)"])

    # --- TAB 1: MANUAL ENTRY ---
    with tab1:
        # Fixed date (next sequential date)
        entry_date = next_date
        st.markdown(f"### 📆 Adding data for: **{entry_date.strftime('%Y-%m-%d')}**")
        
        # Market selector
        # We should ideally show markets present in data, but default to major list
        known_markets = sorted(df['market'].dropna().unique().tolist()) if not df.empty else ['Colombo', 'Dambulla', 'Kandy']
        market = st.selectbox("Market", options=known_markets)
        
        # Weather data (REQUIRED for predictions)
        st.markdown("### 🌤️ Weather Data (Required)")
        weather_cols = st.columns(5)
        with weather_cols[0]:
            temperature = st.number_input("Temperature (°C)", value=float(DEFAULT_WEATHER['temperature_avg_C']), step=0.5)
        with weather_cols[1]:
            rainfall = st.number_input("Rainfall (mm)", value=float(DEFAULT_WEATHER['rainfall_mm']), step=0.5)
        with weather_cols[2]:
            humidity = st.number_input("Humidity (%)", value=float(DEFAULT_WEATHER['humidity_percent']), step=1.0)
        with weather_cols[3]:
            wind_speed = st.number_input("Wind Speed (km/h)", value=float(DEFAULT_WEATHER['wind_speed']), step=0.5)
        with weather_cols[4]:
            sunshine_hours = st.number_input("Sunshine (hours)", value=float(DEFAULT_WEATHER['sunshine_hours']), step=0.5)
        
        # Crop data entry
        st.subheader("Enter Price and Volume for Each Crop")
        
        # Get last known prices for reference
        last_prices = {}
        if not df.empty:
            for crop in TARGET_CROPS:
                # Get last price for this market specifically if possible
                crop_df = df[(df['item'] == crop) & (df['market'] == market)].sort_values('Date')
                if not crop_df.empty:
                    last_prices[crop] = crop_df.iloc[-1]['price']
                else:
                    # Fallback to any market
                    crop_df_any = df[df['item'] == crop].sort_values('Date')
                    if not crop_df_any.empty:
                        last_prices[crop] = crop_df_any.iloc[-1]['price']
        
        crop_data = {}
        for crop in TARGET_CROPS:
            with st.container():
                st.markdown(f"### 🌾 {crop} / {CROP_NAMES_SI.get(crop, crop)}")
                
                default_price = last_prices.get(crop, 100.0)
                cols = st.columns(3)
                with cols[0]:
                    price = st.number_input(
                        f"Price (LKR/kg)",
                        min_value=0.0, max_value=5000.0, value=float(default_price), step=1.0,
                        key=f"price_{crop}"
                    )
                with cols[1]:
                    volume = st.number_input(
                        f"Volume (MT)",
                        min_value=0.0, max_value=2000.0, value=10.0, step=0.1,
                        key=f"volume_{crop}"
                    )
                with cols[2]:
                    if crop in last_prices:
                        change = ((price - last_prices[crop]) / last_prices[crop]) * 100
                        st.metric("vs Last Entry", f"{change:+.1f}%")
                    else:
                        st.write("")
                
                crop_data[crop] = {'price': price, 'volume': volume}
        
        st.markdown("---")
        
        # Save button
        if st.button(f"💾 Save Data for {entry_date.strftime('%Y-%m-%d')}", type="primary", use_container_width=True):
            new_rows = []
            for crop, data in crop_data.items():
                new_row = {
                    'Date': pd.Timestamp(entry_date),
                    'market': market,
                    'item': crop,
                    'price': data['price'],
                    'volume_MT': data['volume'], # Map volume to volume_MT
                    'temp': temperature,
                    'rainfall': rainfall,
                    'humidity': humidity,
                    'wind_speed': wind_speed,
                    'sunshine_hours': sunshine_hours,
                    'is_public_holiday': 0, # Default
                    'demand_multiplier': 1.0, # Default
                    # Add quantity_tonnes if your model needs it explicitly
                    'quantity_tonnes': data['volume'] 
                }
                new_rows.append(new_row)
            
            new_df = pd.DataFrame(new_rows)
            if not df.empty:
                df_updated = pd.concat([df, new_df], ignore_index=True)
            else:
                df_updated = new_df
            
            # Remove duplicates if any for same date/market/item
            df_updated = df_updated.drop_duplicates(subset=['Date', 'market', 'item'], keep='last')
            df_updated = df_updated.sort_values(['Date', 'market', 'item']).reset_index(drop=True)
            
            save_data(df_updated)
            
            st.success(f"✅ Data saved for {entry_date.strftime('%Y-%m-%d')} at {market}!")
            st.rerun()

    # --- TAB 2: BULK UPLOAD ---
    with tab2:
        st.info("📂 Upload a CSV or Excel file to automatically add multiple days/records.")
        
        uploaded_file = st.file_uploader("Upload File", type=['csv', 'xlsx'])
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    upload_df = pd.read_csv(uploaded_file)
                else:
                    upload_df = pd.read_excel(uploaded_file)
                
                # Validation and mapping
                st.write("First 5 rows of uploaded data:", upload_df.head())
                
                # Check critical cols - loosen requirements to find similar names
                # ... [Logic similar to provided snippet] ...
                
                # For brevity, implementing essential check
                required_cols = ['Date', 'item', 'price']
                missing = [c for c in required_cols if c not in upload_df.columns and c.lower() not in [x.lower() for x in upload_df.columns]]
                
                if missing:
                     st.error(f"❌ Missing columns: {missing}. Please ensure file has Date, Item, Price.")
                else:
                    # Normalize cols
                    upload_df.columns = [c.lower() for c in upload_df.columns]
                    # Rename back to standard
                    rename_map = {'date': 'Date', 'item': 'item', 'price': 'price', 'volume': 'volume_MT', 'qty': 'volume_MT'}
                    upload_df = upload_df.rename(columns=rename_map)
                    
                    upload_df['Date'] = pd.to_datetime(upload_df['Date'])
                    
                    if st.button("🚀 Process & Save Batch"):
                        # Fill defaults
                        defaults = {
                            'market': 'Colombo',
                            'volume_MT': 10.0,
                            'quantity_tonnes': 10.0,
                            'temp': DEFAULT_WEATHER['temperature_avg_C'],
                            'rainfall': DEFAULT_WEATHER['rainfall_mm'],
                            'humidity': DEFAULT_WEATHER['humidity_percent'],
                            'wind_speed': DEFAULT_WEATHER['wind_speed'],
                            'sunshine_hours': DEFAULT_WEATHER['sunshine_hours'],
                            'is_public_holiday': 0, 'demand_multiplier': 1.0
                        }
                        for col, val in defaults.items():
                            if col not in upload_df.columns:
                                upload_df[col] = val
                        
                        # Ensure quantity_tonnes matches volume if missing
                        if 'quantity_tonnes' not in upload_df.columns and 'volume_MT' in upload_df.columns:
                             upload_df['quantity_tonnes'] = upload_df['volume_MT']

                        if not df.empty:
                            df_updated = pd.concat([df, upload_df], ignore_index=True)
                        else:
                            df_updated = upload_df
                        
                        df_updated = df_updated.drop_duplicates(subset=['Date', 'market', 'item'], keep='last')
                        df_updated = df_updated.sort_values(['Date', 'market', 'item']).reset_index(drop=True)
                        save_data(df_updated)
                        st.success(f"✅ Successfully added {len(upload_df)} records!")
                        
            except Exception as e:
                st.error(f"Error reading file: {e}")

# ============================================================================
# MODE: VIEW DATA
# ============================================================================
elif mode == '📈 View Data':
    st.subheader("📈 View Historical Data")
    
    if df.empty:
        st.warning("No data available. Add some daily data first!")
    else:
        # Tabs for different views
        view_tab1, view_tab2 = st.tabs(["📊 Historical Trends", "🔄 Market Comparison"])
        
        # TAB 1: Historical Trends
        with view_tab1:
            # Stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Days of Data", len(df['Date'].unique()))
            with col3:
                st.metric("First Date", df['Date'].min().strftime('%Y-%m-%d'))
            with col4:
                st.metric("Last Date", df['Date'].max().strftime('%Y-%m-%d'))
            
            st.markdown("---")
            
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                selected_crop = st.selectbox("Crop", options=['All'] + TARGET_CROPS, key="view_crop")
            with col2:
                markets = ['All'] + sorted(list(df['market'].unique()))
                selected_market = st.selectbox("Market", options=markets, key="view_market")
            
            # Filter data
            filtered_df = df.copy()
            if selected_crop != 'All':
                filtered_df = filtered_df[filtered_df['item'] == selected_crop]
            if selected_market != 'All':
                filtered_df = filtered_df[filtered_df['market'] == selected_market]
            
            # Chart
            if not filtered_df.empty:
                # Aggregate to avoid plotting issues
                chart_df = filtered_df.groupby(['Date', 'item']).agg({'price': 'mean', 'volume_MT': 'mean'}).reset_index()
                
                fig = go.Figure()
                for crop in chart_df['item'].unique():
                    crop_df = chart_df[chart_df['item'] == crop]
                    fig.add_trace(go.Scatter(
                        x=crop_df['Date'],
                        y=crop_df['price'],
                        mode='lines+markers',
                        name=crop
                    ))
                
                fig.update_layout(
                    title=f"Price Trends ({selected_market})",
                    xaxis_title="Date",
                    yaxis_title="Price (LKR/kg)",
                    template='plotly_white',
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Recent entries table
            st.markdown("### Recent Entries (Last 50)")
            display_cols = ['Date', 'market', 'item', 'price', 'volume_MT']
            st.dataframe(
                filtered_df[display_cols].sort_values('Date', ascending=False).head(50),
                use_container_width=True
            )
        
        # TAB 2: Market Comparison
        with view_tab2:
            st.markdown("### 🔄 Compare Prices Across Markets")
            st.info("Select a crop to see current prices across different markets")
            
            compare_crop = st.selectbox("Select Crop to Compare", options=TARGET_CROPS, key="compare_crop")
            
            if compare_crop:
                crop_df = df[df['item'] == compare_crop].copy()
                
                if not crop_df.empty:
                    # Get latest prices by market
                    latest_date = crop_df['Date'].max()
                    recent_df = crop_df[crop_df['Date'] >= (latest_date - pd.Timedelta(days=7))]
                    
                    market_summary = recent_df.groupby('market').agg({
                        'price': ['mean', 'min', 'max'],
                        'volume_MT': 'sum'
                    }).reset_index()
                    
                    market_summary.columns = ['Market', 'Avg Price', 'Min Price', 'Max Price', 'Total Volume']
                    market_summary = market_summary.sort_values('Avg Price', ascending=False)
                    
                    # Display as metrics
                    st.markdown(f"#### Last 7 Days Average for {compare_crop}")
                    
                    # Top 3 markets by price
                    top_markets = market_summary.head(3)
                    cols = st.columns(3)
                    
                    for i, (_, row) in enumerate(top_markets.iterrows()):
                        with cols[i]:
                            st.metric(
                                f"🏆 {row['Market']}",
                                f"{row['Avg Price']:.2f} LKR/kg",
                                f"±{(row['Max Price'] - row['Min Price'])/2:.1f}"
                            )
                    
                    st.markdown("---")
                    
                    # Full comparison table
                    st.dataframe(
                        market_summary.style.background_gradient(subset=['Avg Price'], cmap='RdYlGn'),
                        use_container_width=True
                    )
                    
                    # Price comparison chart
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=market_summary['Market'],
                        y=market_summary['Avg Price'],
                        text=market_summary['Avg Price'].round(2),
                        textposition='auto',
                        marker_color='lightblue'
                    ))
                    
                    fig.update_layout(
                        title=f"{compare_crop} - Average Price by Market (Last 7 Days)",
                        xaxis_title="Market",
                        yaxis_title="Price (LKR/kg)",
                        template='plotly_white',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No data available for {compare_crop}")

# ============================================================================
# MODE: RETRAIN MODELS
# ============================================================================
elif mode == '🔄 Retrain Models':
    st.subheader("🔄 Monthly Model Retraining")
    
    st.info("""
    **When to retrain:**
    - At the end of each month
    - When you have accumulated significant new data
    - When prediction accuracy seems to decrease
    
    **What happens:**
    - LSTM demand models will be retrained with all accumulated data
    - LightGBM price models will be updated
    - This may take 5-10 minutes
    """)
    
    # Show data stats
    if not df.empty:
        st.markdown("### 📊 Current Data Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Date Range", f"{(df['Date'].max() - df['Date'].min()).days} days")
        with col3:
            st.metric("First Date", df['Date'].min().strftime('%Y-%m-%d'))
        with col4:
            st.metric("Last Date", df['Date'].max().strftime('%Y-%m-%d'))
        
        # Check if month-end
        last_date = df['Date'].max()
        is_month_end = (last_date + timedelta(days=1)).month != last_date.month
        
        if is_month_end:
            st.success(f"✅ Month-end detected ({last_date.strftime('%Y-%m-%d')}). Recommended to retrain!")
        else:
            days_to_month_end = (pd.Timestamp(last_date.year, last_date.month, 1) + pd.offsets.MonthEnd(1) - last_date).days
            st.info(f"📅 {days_to_month_end} days until month end. You can wait or retrain now.")
        
        st.markdown("---")
        
        # Retrain button
        if st.button("🚀 Start Retraining", type="primary", use_container_width=True):
            try:
                from trainer import retrain_models
                
                st.info("🔄 Starting model retraining... This may take 10-20 minutes.")
                
                # Create progress indicators
                status_text = st.empty()
                results_container = st.container()
                
                # Track progress for visual feedback
                progress_counter = [0]  # Use list to allow modification in nested function
                
                # Define callback for progress updates (trainer passes just message string)
                def progress_callback(message):
                    progress_counter[0] += 1
                    status_text.text(f"[{progress_counter[0]}] {message}")
                
                # Run retraining
                status_text.text("🔄 Loading training data...")
                
                results = retrain_models(
                    price_data_path=DATA_PATH,
                    demand_data_path=DEMAND_DATA_PATH,
                    save_dir=os.path.join(PROJECT_ROOT, 'models', 'saved_models'),
                    progress_callback=progress_callback
                )
                
                status_text.text("✅ Retraining complete!")
                
                # Display results
                with results_container:
                    st.success("🎉 Model retraining completed successfully!")
                    
                    # Show demand model results
                    st.markdown("### 📈 Demand Model Results")
                    demand_cols = st.columns(4)
                    for i, (crop, metrics) in enumerate(results.get('demand_models', {}).items()):
                        with demand_cols[i % 4]:
                            if 'error' in metrics:
                                st.error(f"**{crop}**: {metrics['error']}")
                            else:
                                st.metric(
                                    label=crop,
                                    value=f"MAE: {metrics.get('mae', 'N/A'):.2f}" if isinstance(metrics.get('mae'), (int, float)) else "Trained"
                                )
                    
                    # Show price model results
                    st.markdown("### 💰 Price Model Results")
                    price_cols = st.columns(4)
                    for i, (crop, metrics) in enumerate(results.get('price_models', {}).items()):
                        with price_cols[i % 4]:
                            if 'error' in metrics:
                                st.error(f"**{crop}**: {metrics['error']}")
                            else:
                                st.metric(
                                    label=crop,
                                    value=f"R²: {metrics.get('r2', 'N/A'):.3f}" if isinstance(metrics.get('r2'), (int, float)) else "Trained"
                                )
                    
                    st.info("💡 Restart the app to use the new models, or they will be loaded on next prediction.")
                    
                    # Offer to reload predictor
                    if st.button("🔄 Reload Models Now"):
                        st.session_state.pop('predictor', None)
                        st.rerun()
                        
            except ImportError as e:
                st.error(f"❌ Training module not found: {e}")
                st.info("Make sure trainer.py is in the app folder.")
            except Exception as e:
                st.error(f"❌ Retraining failed: {str(e)}")
                st.exception(e)
    else:
        st.warning("No data available. Add daily data first before retraining.")

# ============================================================================
# MODE: SETTINGS
# ============================================================================
elif mode == '⚙️ Settings':
    st.subheader("⚙️ User Preferences")
    
    # Initialize session state for settings if not exists
    if 'user_settings' not in st.session_state:
        st.session_state.user_settings = {
            'min_acceptable_price': {},
            'risk_level': 'Medium',
            'alert_quiet_hours': {'start': 22, 'end': 7},
            'sms_alerts_enabled': False,
            'sms_number': ''
        }
    
    settings = st.session_state.user_settings
    
    # Minimum Price Settings
    st.markdown("### 💰 Minimum Acceptable Prices")
    st.info("Set minimum prices you're willing to accept for each crop. System won't recommend selling below these.")
    
    price_cols = st.columns(4)
    for i, crop in enumerate(TARGET_CROPS):
        with price_cols[i]:
            current_min = settings['min_acceptable_price'].get(crop, 0.0)
            settings['min_acceptable_price'][crop] = st.number_input(
                f"{crop} (LKR/kg)",
                min_value=0.0,
                value=float(current_min),
                step=10.0,
                key=f"min_price_{crop}"
            )
    
    st.markdown("---")
    
    # Risk Level
    st.markdown("### 📊 Risk Tolerance")
    settings['risk_level'] = st.select_slider(
        "How much price volatility can you tolerate?",
        options=['Conservative', 'Medium', 'Aggressive'],
        value=settings['risk_level']
    )
    
    if settings['risk_level'] == 'Conservative':
        st.info("🛡️ **Conservative:** Prefer selling early to avoid risk, even if potential gains exist.")
    elif settings['risk_level'] == 'Aggressive':
        st.info("📈 **Aggressive:** Willing to hold longer for higher potential profits, accepting more risk.")
    else:
        st.info("⚖️ **Medium:** Balanced approach between safety and profit maximization.")
    
    st.markdown("---")
    
    # Alert Settings
    st.markdown("### 🔔 Alert Preferences")
    
    alert_cols = st.columns(2)
    
    with alert_cols[0]:
        st.markdown("**Quiet Hours** (No alerts during these times)")
        quiet_start = st.slider("Start Hour", 0, 23, settings['alert_quiet_hours']['start'])
        quiet_end = st.slider("End Hour", 0, 23, settings['alert_quiet_hours']['end'])
        settings['alert_quiet_hours'] = {'start': quiet_start, 'end': quiet_end}
    
    with alert_cols[1]:
        st.markdown("**SMS Alerts** (Coming Soon)")
        settings['sms_alerts_enabled'] = st.checkbox(
            "Enable SMS alerts",
            value=settings['sms_alerts_enabled'],
            disabled=True,
            help="SMS feature will be available in future update"
        )
        settings['sms_number'] = st.text_input(
            "Phone Number",
            value=settings['sms_number'],
            placeholder="+94771234567",
            disabled=True
        )
    
    st.markdown("---")
    
    # Save button
    if st.button("💾 Save Settings", type="primary", use_container_width=True):
        st.session_state.user_settings = settings
        st.success("✅ Settings saved successfully!")
        st.balloons()

# ============================================================================
# MODE: GET PREDICTION
# ============================================================================
else:
    # Check data availability
    if df.empty:
        st.error("❌ No data available. Please add daily data first using '📝 Add Daily Data' mode.")
        st.stop()
    
    last_data_date = get_last_entry_date(df)
    
    # Sidebar inputs for prediction
    with st.sidebar:
        st.markdown("---")
        st.subheader("Prediction Settings")
        
        if last_data_date:
            st.info(f"📅 Data available up to: **{last_data_date}**")
        
        # Crop selection
        crop = st.selectbox(
            lang['crop'],
            options=TARGET_CROPS,
            format_func=lambda x: f"{x} / {CROP_NAMES_SI.get(x, x)}" if language == 'si' else x
        )
        
        # Market
        markets = sorted(list(df['market'].dropna().unique()))
        market = st.selectbox(lang['market'], options=markets)
        
        # Quantity
        quantity_kg = st.number_input(lang['quantity'], min_value=1, max_value=10000, value=100, step=10)
        
        # Days since harvest
        days_since_harvest = st.number_input(lang['days_harvest'], min_value=0, max_value=365, value=0, step=1)
        
        get_rec = st.button(f"🎯 {lang['get_recommendation']}", type="primary", use_container_width=True)
    
    # Show current data status
    current_sys_date = datetime.now().date()
    days_lag = (current_sys_date - last_data_date).days if last_data_date else 0
    
    status_icon = "✅" if days_lag <= 0 else "⚠️"
    status_msg = "Data is Up-to-Date" if days_lag <= 0 else f"Data Lagging by {days_lag} Days"
    status_bg = "#e8f5e9" if days_lag <= 0 else "#ffebee"
    status_border = "#4caf50" if days_lag <= 0 else "#f44336"

    st.markdown(f"""
    <div style="background-color: {status_bg}; border: 2px solid {status_border}; padding: 1rem; border-radius: 0.5rem; text-align: center; margin-bottom: 1rem;">
        <h3 style="margin: 0; color: #333;">{status_icon} {status_msg}</h3>
        <p style="margin: 0.5rem 0 0 0; color: #555;">
            Today: <strong>{current_sys_date}</strong> | Data Available Up To: <strong>{last_data_date.strftime('%Y-%m-%d')}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # Logic to show results if button clicked OR valid result in session
    if get_rec:
        # Get real historical data
        crop_df = df[(df['item'] == crop) & (df['market'] == market)].sort_values('Date')
        
        if crop_df.empty:
            st.warning(f"No specific data for {crop} in {market}. Using average of all markets.")
            crop_df = df[df['item'] == crop].groupby('Date').agg({'price':'mean', 'volume_MT':'mean'}).reset_index().sort_values('Date')
        
        # Need at least some history
        if crop_df.empty:
            st.error("No historical data found for this crop.")
        else:
             # Take last 30 days
             recent = crop_df.tail(60)
             price_history = recent['price'].tolist()
             volume_history = recent['volume_MT'].tolist()
             
             with st.spinner('🔄 Analyzing with real historical data...'):
                result = predictor.get_recommendation(
                    crop=crop,
                    price_history=price_history,
                    volume_history=volume_history,
                    current_date=datetime.combine(last_data_date, datetime.min.time()),
                    quantity_kg=quantity_kg,
                    days_since_harvest=days_since_harvest
                )
                st.session_state['last_result'] = result
                st.session_state['last_crop'] = crop
                
    
    if 'last_result' in st.session_state:
        result = st.session_state['last_result']
        
        # Display Results
        current_price = result.get('current_price', 0)
        
        # Decision display
        decision = result['decision']
        if 'SELL' in decision:
            decision_class = 'decision-sell'
            decision_emoji = '🔴'
        elif 'HOLD' in decision:
            decision_class = 'decision-hold'
            decision_emoji = '🟢'
        else:
            decision_class = 'decision-wait'
            decision_emoji = '🟡'
        
        # Top metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            decision_text = result['decision']
            if language == 'si':
                if 'SELL' in decision_text:
                    decision_text = lang['sell_now']
                elif 'HOLD' in decision_text:
                    decision_text = f"{lang['hold']} {result['best_hold_days']} {lang['days']}"
                else:
                    decision_text = lang['wait']
            st.metric(lang['decision'], f"{decision_emoji} {decision_text}")
        
        with col2:
            st.metric(lang['confidence'], f"{result.get('confidence',0):.0f}%")
        
        with col3:
            # Trend Signal
            trend = result.get('trend_signal', 'Steady →')
            st.metric("Price Trend", trend)
        
        with col4:
            profit = result.get('expected_profit_per_kg', 0)
            st.metric(f"{lang['expected_profit']}/kg", f"{profit:.2f} LKR", f"{'+' if profit > 0 else ''}{profit:.2f}")
        
        with col5:
            st.metric("Total Profit", f"{result.get('expected_profit_total',0):.0f} LKR")
        
        # Seasonal & Festival Context
        season_info = result.get('season', {})
        festivals = result.get('upcoming_festivals', [])
        
        if season_info or festivals:
            st.markdown("---")
            context_cols = st.columns([1, 2])
            
            with context_cols[0]:
                if season_info:
                    season_name = season_info.get('name', 'N/A')
                    season_desc = season_info.get('description', '')
                    st.info(f"🌾 **Season:** {season_name} ({season_desc})")
            
            with context_cols[1]:
                if festivals:
                    festival_text = " | ".join([
                        f"🎉 **{f['name']}** in {f['days_until']} days" 
                        for f in festivals[:2]  # Show max 2 festivals
                    ])
                    st.warning(festival_text)
        
        st.markdown("---")
        
        # Charts
        col_price, col_demand = st.columns(2)
        
        with col_price:
            st.subheader(f"📈 {lang['price_forecast']}")
            predictions = result.get('predictions', {})
            if predictions:
                horizons = list(predictions.keys())
                prices = [predictions[h] for h in horizons]
                days = [HORIZONS[h] for h in horizons]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[0], y=[current_price], mode='markers', name='Now', marker=dict(size=15, color='red', symbol='star')))
                fig.add_trace(go.Scatter(x=[0] + days, y=[current_price] + prices, mode='lines+markers', name='Forecast', line=dict(color='#2E7D32', width=3)))
                fig.update_layout(xaxis_title="Days Ahead", yaxis_title="Price (LKR/kg)", template='plotly_white', height=350, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        with col_demand:
            st.subheader(f"📦 {lang['demand_forecast']}")
            demand_predictions = result.get('demand_predictions', {})
            if demand_predictions:
                horizons = list(demand_predictions.keys())
                volumes = [demand_predictions[h] for h in horizons]
                days = [HORIZONS[h] for h in horizons]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(x=[f"{d}d" for d in days], y=volumes, marker_color='#1976D2', text=[f"{v:.1f}" for v in volumes], textposition='outside'))
                fig.update_layout(xaxis_title="Horizon", yaxis_title="Volume (MT)", template='plotly_white', height=350)
                st.plotly_chart(fig, use_container_width=True)
        
        # Reasoning
        st.markdown(f"""
        <div class='{decision_class}'>
            <h3>{decision_emoji} {result['decision']}</h3>
            <p>{result['reasoning']}</p>
            <p><strong>Best timing:</strong> {result.get('best_time','-')} ({result.get('best_hold_days',0)} days)</p>
            <p><strong>Expected price:</strong> {result.get('best_price',0):.2f} LKR/kg</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🌾 <strong>YieldSync</strong> - Empowering Sri Lankan Farmers with AI</p>
    <p>Data-driven predictions | Version 2.0</p>
</div>
""", unsafe_allow_html=True)
