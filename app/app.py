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
    CROP_NAMES_SI, TRANSLATIONS, DEFAULT_WEATHER, CROP_MARKETS
)

# Data fetcher for automated updates
try:
    from data_fetcher import fetch_week, update_this_week, update_since_date, fetch_weather
    HAS_DATA_FETCHER = True
except ImportError:
    HAS_DATA_FETCHER = False

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
    
    /* Fix info/warning/success text visibility */
    div[data-testid="stAlert"] > div {
        color: #1f1f1f !important;
    }
    div[data-testid="stAlert"] p {
        color: #1f1f1f !important;
    }
    div[data-testid="stAlert"] strong {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA PATH
# ============================================================================
# Use robust pathing - deployment package is self-contained
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = BASE_DIR  # deployment_package is the root
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')

# Main historical data file (Price + Weather)
DATA_PATH = os.path.join(DATA_DIR, 'full_history_features_real_weather.csv')

# ============================================================================
# LOAD DATA
# ============================================================================
@st.cache_data
def load_data():
    """Load market data from CSV (Price + Weather)."""
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            df['Date'] = pd.to_datetime(df['Date'])
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
        options=['📊 Get Prediction', '📈 View Data', '🔄 Retrain Models', '⚙️ Settings'],
        index=0
    )
    
    st.markdown("---")
    
    # Quick Data Update Button
    st.markdown("### 📥 Update Data")
    if HAS_DATA_FETCHER:
        if st.button("🔄 Fetch Latest Week", use_container_width=True):
            with st.spinner("Checking for new data..."):
                try:
                    today = datetime.now()
                    
                    # Load current data to check last date
                    current_df = load_data()
                    
                    if current_df.empty:
                        st.warning("⚠️ No existing data. Please load initial dataset first.")
                    else:
                        last_data_date = current_df['Date'].max()
                        
                        # Calculate the last completed week
                        # HARTI publishes data after the week ends (usually Monday)
                        # So we should only fetch weeks where the end date has passed
                        days_since_last = (today - last_data_date).days
                        
                        if days_since_last < 7:
                            st.info(f"✅ Data is up to date! Last entry: {last_data_date.strftime('%Y-%m-%d')}")
                            st.info(f"📅 Next update available after: {(last_data_date + timedelta(days=7)).strftime('%Y-%m-%d')}")
                        else:
                            # There's at least one week of missing data - fetch it
                            # Calculate which week to fetch (the week after last data)
                            fetch_date = last_data_date + timedelta(days=7)
                            year = fetch_date.year
                            week = fetch_date.isocalendar()[1]
                            
                            # Get last prices for fallback
                            last_week = current_df[current_df['Date'] >= (last_data_date - timedelta(days=7))]
                            last_prices = None
                            if not last_week.empty:
                                last_prices = last_week[['market', 'item', 'price']].drop_duplicates()
                            
                            st.info(f"📥 Fetching Week {week} of {year}...")
                            data = fetch_week(year, week, last_prices)
                            
                            if not data.empty:
                                df_updated = pd.concat([current_df, data], ignore_index=True)
                                df_updated = df_updated.drop_duplicates(subset=['Date', 'market', 'item'], keep='last')
                                df_updated = df_updated.sort_values(['Date', 'market', 'item']).reset_index(drop=True)
                                save_data(df_updated)
                                st.success(f"✅ Week {week} data added! ({len(data)} records)")
                                st.rerun()
                            else:
                                st.warning("⚠️ No data available for that week yet")
                except Exception as e:
                    st.error(f"❌ {e}")
    else:
        st.warning("Install pdfplumber")
    
    st.markdown("---")
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
# MODE: VIEW DATA
# ============================================================================
if mode == '📈 View Data':
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
                # Show crop-specific markets when a crop is selected
                if selected_crop != 'All':
                    valid_markets = ['All'] + sorted(CROP_MARKETS.get(selected_crop, []))
                else:
                    valid_markets = ['All'] + sorted(list(df['market'].unique()))
                selected_market = st.selectbox("Market", options=valid_markets, key="view_market")

            
            # Filter data
            filtered_df = df.copy()
            if selected_crop != 'All':
                filtered_df = filtered_df[filtered_df['item'] == selected_crop]
            if selected_market != 'All':
                filtered_df = filtered_df[filtered_df['market'] == selected_market]
            
            # Chart
            if not filtered_df.empty:
                # Aggregate to avoid plotting issues - only aggregate price
                chart_df = filtered_df.groupby(['Date', 'item'])['price'].mean().reset_index()
                
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
            display_cols = ['Date', 'market', 'item', 'price']
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
                        'price': ['mean', 'min', 'max']
                    }).reset_index()
                    
                    market_summary.columns = ['Market', 'Avg Price', 'Min Price', 'Max Price']
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
    - Price models will be retrained with all accumulated data
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
                    save_dir=MODELS_DIR,
                    progress_callback=progress_callback
                )
                
                status_text.text("✅ Retraining complete!")
                
                # Display results
                with results_container:
                    st.success("🎉 Model retraining completed successfully!")
                    
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
        
        # Date Selection
        if last_data_date:
            st.info(f"📅 Data available up to: **{last_data_date}**")
            
            # Determine the max selectable date (today or last data date, whichever is earlier)
            today = datetime.now().date()
            # Use last available data date (may be ahead if using last week's avg prices)
            max_date = last_data_date
            
            # Default to the last data date
            default_date = last_data_date
            
            selected_date = st.date_input(
                "📅 Select Date for Prediction",
                value=default_date,
                min_value=df['Date'].min().date() if not df.empty else today,
                max_value=max_date,
                help="Select the date for which you want predictions. Uses last week's avg prices until new HARTI data."
            )
        else:
            selected_date = datetime.now().date()
            st.warning("No data available. Please add data first.")
        
        # Crop selection
        crop = st.selectbox(
            lang['crop'],
            options=TARGET_CROPS,
            format_func=lambda x: f"{x} / {CROP_NAMES_SI.get(x, x)}" if language == 'si' else x
        )
        
        # Market - Show only markets valid for selected crop
        crop_markets = sorted(CROP_MARKETS.get(crop, []))
        market = st.selectbox(lang['market'], options=crop_markets)
        
        # Quantity
        quantity_kg = st.number_input(lang['quantity'], min_value=1, max_value=10000, value=100, step=10)
        
        # Not yet harvested checkbox
        not_yet_harvested = st.checkbox(
            "Not Yet Harvested",
            value=False,
            help="Check this if the crop has not been harvested yet (for planning purposes)"
        )
        
        # Days since harvest input
        days_since_harvest = st.number_input(
            lang['days_harvest'], 
            min_value=0,
            max_value=365, 
            value=0, 
            step=1,
            help="Enter 0 for fresh harvest, or number of days since the crop was harvested.",
            disabled=not_yet_harvested
        )
        
        # Set to -1 internally if not yet harvested
        if not_yet_harvested:
            days_since_harvest = -1
        
        get_rec = st.button(f"🎯 {lang['get_recommendation']}", type="primary", use_container_width=True)
    
    # Show selected date info
    if 'selected_date' in locals():
        st.markdown(f"""
        <div style="background-color: #e3f2fd; border: 2px solid #1976D2; padding: 1rem; border-radius: 0.5rem; text-align: center; margin-bottom: 1rem;">
            <h3 style="margin: 0; color: #333;">📅 Prediction Date</h3>
            <p style="margin: 0.5rem 0 0 0; color: #555;">
                Making predictions for: <strong style="color: #1976D2; font-size: 1.2rem;">{selected_date.strftime('%Y-%m-%d')}</strong>
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
            crop_df = df[df['item'] == crop].groupby('Date')['price'].mean().reset_index().sort_values('Date')
        
        # Validate selected date has data
        if selected_date > last_data_date:
            st.error(f"❌ No data available for {selected_date.strftime('%Y-%m-%d')}. Latest data is from {last_data_date.strftime('%Y-%m-%d')}.")
        else:
            # Get real historical data up to selected date
            crop_df = df[(df['item'] == crop) & (df['market'] == market) & (df['Date'] <= pd.Timestamp(selected_date))].sort_values('Date')
            
            if crop_df.empty:
                st.warning(f"No specific data for {crop} in {market}. Using average of all markets.")
                crop_df = df[(df['item'] == crop) & (df['Date'] <= pd.Timestamp(selected_date))].groupby('Date')['price'].mean().reset_index().sort_values('Date')
            
            # Need at least some history
            if crop_df.empty:
                st.error("No historical data found for this crop up to the selected date.")
            else:
                 # Get current price
                 current_price = float(crop_df.iloc[-1]['price'])
                 
                 with st.spinner('🔄 Analyzing with real historical data...'):
                    # Predict prices for all horizons (7, 14, 30 days)
                    predictions = {}
                    primary_horizon = days_since_harvest if days_since_harvest > 0 else 7
                    
                    for horizon in HORIZONS.keys():  # '7day', '14day', '30day'
                        horizon_days = HORIZONS[horizon]
                        price_result_h = predictor.predict_price(
                            data=df[df['Date'] <= pd.Timestamp(selected_date)],
                            crop=crop,
                            market=market,
                            days_ahead=horizon_days
                        )
                        if 'error' not in price_result_h:
                            predictions[horizon] = price_result_h['predicted_price']
                    
                    # Get primary prediction for recommendation
                    price_result = predictor.predict_price(
                        data=df[df['Date'] <= pd.Timestamp(selected_date)],
                        crop=crop,
                        market=market,
                        days_ahead=primary_horizon
                    )
                    
                    if 'error' in price_result:
                        st.error(f"Prediction error: {price_result['error']}")
                        result = {
                            'decision': 'UNKNOWN',
                            'predicted_price': current_price,
                            'current_price': current_price,
                            'price_change_percent': 0,
                            'reasoning': price_result['error'],
                            'confidence': 0,
                            'predictions': predictions,
                            'expected_profit_per_kg': 0,
                            'expected_profit_total': 0,
                            'best_price': current_price,
                            'best_time': 'Now',
                            'best_hold_days': 0,
                            'shelf_life_remaining': PERISHABILITY.get(crop, 30),
                            'perishability': 'Medium'
                        }
                    else:
                        predicted_price = price_result['predicted_price']
                        
                        # Get recommendation based on prices
                        rec_result = predictor.get_recommendation(
                            crop=crop,
                            current_price=current_price,
                            predicted_price=predicted_price,
                            days_ahead=price_result.get('days_ahead', 7),
                            quantity_kg=quantity_kg
                        )
                        
                        # Calculate additional display values
                        shelf_life = PERISHABILITY.get(crop, 30)
                        days_harvest = days_since_harvest if days_since_harvest >= 0 else 0
                        shelf_remaining = max(0, shelf_life - days_harvest)
                        
                        # Determine perishability category
                        if shelf_life <= 7:
                            perish_cat = 'High'
                        elif shelf_life <= 30:
                            perish_cat = 'Medium'
                        else:
                            perish_cat = 'Low'
                        
                        # Find best horizon from predictions
                        best_horizon = '7day'
                        best_price_val = predicted_price
                        if predictions:
                            best_horizon = max(predictions.keys(), key=lambda h: predictions[h])
                            best_price_val = predictions[best_horizon]
                        
                        # Calculate profit per kg
                        profit_per_kg = predicted_price - current_price
                        profit_total = profit_per_kg * quantity_kg
                        
                        # Determine best time text
                        if 'HOLD' in rec_result.get('decision', ''):
                            best_time_text = f"Hold for {HORIZONS.get(best_horizon, 7)} days"
                            best_hold = HORIZONS.get(best_horizon, 7)
                        else:
                            best_time_text = "Sell Now"
                            best_hold = 0
                        
                        # Combine results with predictions for chart
                        result = {
                            **price_result,
                            **rec_result,
                            'predictions': predictions,
                            'confidence': 75,
                            'expected_profit_per_kg': round(profit_per_kg, 2),
                            'expected_profit_total': round(profit_total, 2),
                            'best_price': round(best_price_val, 2),
                            'best_time': best_time_text,
                            'best_hold_days': best_hold,
                            'shelf_life_remaining': shelf_remaining,
                            'perishability': perish_cat
                        }
                    
                    st.session_state['last_result'] = result
                    st.session_state['last_crop'] = crop
                    st.session_state['last_selected_date'] = selected_date
                    
                    # Display results only if we have a valid result
                    # URGENT WARNINGS - Show first if critical
                    urgency_warning = result.get('urgency_warning', '')
                    if urgency_warning:
                        if '🔴 URGENT' in urgency_warning or 'CRITICAL' in urgency_warning:
                            st.error(urgency_warning)
                        elif '⚠️ WARNING' in urgency_warning:
                            st.warning(urgency_warning)
                    
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
                                decision_text = f"{lang['hold']} {result.get('best_hold_days', 0)} {lang['days']}"
                            else:
                                decision_text = lang['wait']
                        st.metric(lang['decision'], f"{decision_emoji} {decision_text}")
                    
                    with col2:
                        st.metric(lang['confidence'], f"{result.get('confidence',0):.0f}%")
                    
                    with col3:
                        # Shelf Life Remaining
                        shelf_life = result.get('shelf_life_remaining', 'N/A')
                        perishability = result.get('perishability', 'Medium')
                        if isinstance(shelf_life, int):
                            shelf_emoji = '🟢' if shelf_life > 7 else ('🟡' if shelf_life > 3 else '🔴')
                            st.metric(f"{shelf_emoji} Shelf Life", f"{shelf_life} days", f"({perishability})")
                        else:
                            st.metric("Shelf Life", "N/A")
                    
                    with col4:
                        profit = result.get('expected_profit_per_kg', 0)
                        if profit >= 0:
                            st.metric("📈 Price Change/kg", f"+{profit:.2f} LKR", f"Gain if hold", delta_color="normal")
                        else:
                            st.metric("📉 Price Change/kg", f"{profit:.2f} LKR", f"Loss if hold", delta_color="inverse")
                    
                    with col5:
                        total = result.get('expected_profit_total', 0)
                        if total >= 0:
                            st.metric("Total if Hold", f"+{total:.0f} LKR")
                        else:
                            st.metric("Total if Hold", f"{total:.0f} LKR")
                    
                    # Profit Comparison Table for all horizons
                    st.markdown("---")
                    st.markdown("### 📊 Profit Comparison by Horizon")
                    
                    predictions = result.get('predictions', {})
                    if predictions:
                        comparison_data = []
                        for horizon_key, pred_price in predictions.items():
                            horizon_days = HORIZONS.get(horizon_key, 7)
                            change = pred_price - current_price
                            change_pct = (change / current_price * 100) if current_price > 0 else 0
                            total_change = change * quantity_kg
                            
                            # Determine recommendation for this horizon
                            if change_pct >= 2:
                                rec = "🟢 HOLD"
                            elif change_pct <= -2:
                                rec = "🔴 SELL NOW"
                            else:
                                rec = "🟡 NEUTRAL"
                            
                            comparison_data.append({
                                'Horizon': f"{horizon_days} days",
                                'Predicted Price': f"{pred_price:.2f} LKR/kg",
                                'Change/kg': f"{change:+.2f} LKR",
                                'Change %': f"{change_pct:+.1f}%",
                                f'Total ({quantity_kg}kg)': f"{total_change:+,.0f} LKR",
                                'Action': rec
                            })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                        
                        st.caption(f"📍 Current Price: **{current_price:.2f} LKR/kg** | Quantity: **{quantity_kg} kg**")
                    
                    # Harvest & Seasonal Context
                    st.markdown("---")
                    context_row1 = st.columns([1, 1])
                    
                    with context_row1[0]:
                        # Harvest season context
                        harvest_ctx = result.get('harvest_context', '')
                        if harvest_ctx:
                            if '⚠️' in harvest_ctx:
                                st.warning(harvest_ctx)
                            else:
                                st.success(harvest_ctx)
                        
                        # Season info
                        season_info = result.get('season', {})
                        if season_info:
                            season_name = season_info.get('name', 'N/A')
                            season_desc = season_info.get('description', '')
                            st.info(f"🌾 **Growing Season:** {season_name} - {season_desc}")
                    
                    with context_row1[1]:
                        # Crop age tracking
                        days_harvest = result.get('days_since_harvest', 0)
                        if days_harvest == -1:
                            # Not yet harvested
                            st.info(f"🌱 **Planning Mode:** Not yet harvested")
                        elif days_harvest > 0:
                            age_emoji = '🆕' if days_harvest <= 3 else ('⏰' if days_harvest <= 7 else '⚠️')
                            st.info(f"{age_emoji} **Crop Age:** {days_harvest} days since harvest")
                        else:
                            # Just harvested today
                            st.success(f"🆕 **Fresh Harvest:** Just harvested today!")
                        
                        # Trend Signal
                        trend = result.get('trend_signal', 'Steady →')
                        st.info(f"📈 **Price Trend:** {trend}")
                    
                    # Festival Context
                    festivals = result.get('upcoming_festivals', [])
                    if festivals:
                        st.markdown("---")
                        st.markdown("### 🎉 Upcoming Festivals")
                        festival_text = " | ".join([
                            f"**{f['name']}** in {f['days_until']} days (Impact: {f['impact']})"
                            for f in festivals[:2]  # Show max 2 festivals
                        ])
                        st.warning(festival_text)
                    
                    st.markdown("---")
                    
                    # Charts
                    # Price Forecast Chart (Full width)
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
                    else:
                        st.warning("No price predictions available for this configuration.")
                    
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
