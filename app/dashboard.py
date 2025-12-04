

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import *
from src.decision_engine import DecisionEngine

# SINHALA TRANSLATIONS

TRANSLATIONS = {
    'en': {
        'title': '🌾 YieldSync - Smart Farming Decisions',
        'subtitle': 'AI-Powered Price Forecasting & Sell/Hold Recommendations',
        'crop': 'Select Crop',
        'market': 'Select Market',
        'current_price': 'Current Market Price (LKR/kg)',
        'quantity': 'Quantity (kg)',
        'days_harvest': 'Days Since Harvest',
        'get_recommendation': 'Get Recommendation',
        'forecast_title': 'Price Forecast',
        'recommendation_title': 'Recommendation',
        'decision': 'Decision',
        'confidence': 'Confidence',
        'expected_profit': 'Expected Profit',
        'reasoning': 'Reasoning',
        'sell_now': 'SELL NOW',
        'hold': 'HOLD',
        'days': 'days',
        'total_profit': 'Total Expected Profit',
        'per_kg': 'per kg',
    },
    'si': {
        'title': '🌾 YieldSync - ස්මාර්ට් ගොවිතැන තීරණ',
        'subtitle': 'AI මගින් මිල අනාවැකි සහ විකුණුම් නිර්දේශ',
        'crop': 'බෝගය තෝරන්න',
        'market': 'වෙළඳපොළ තෝරන්න',
        'current_price': 'වත්මන් වෙළඳපල මිල (රු/කි.ග්‍රෑ)',
        'quantity': 'ප්‍රමාණය (කි.ග්‍රෑ)',
        'days_harvest': 'අස්වැන්නෙන් පසු දින ගණන',
        'get_recommendation': 'නිර්දේශය ලබා ගන්න',
        'forecast_title': 'මිල අනාවැකිය',
        'recommendation_title': 'නිර්දේශය',
        'decision': 'තීරණය',
        'confidence': 'විශ්වාසනීයත්වය',
        'expected_profit': 'අපේක්ෂිත ලාභය',
        'reasoning': 'හේතුව',
        'sell_now': 'දැන් විකුණන්න',
        'hold': 'තබා ගන්න',
        'days': 'දින',
        'total_profit': 'මුළු අපේක්ෂිත ලාභය',
        'per_kg': 'කි.ග්‍රෑ එකකට',
    }
}

CROP_NAMES_SI = {
    'Rice': 'බත්',
    'Beetroot': 'බීට්',
    'Raddish': 'රාබු',
    'Red Onion': 'ලූනු'
}

# PAGE CONFIG
st.set_page_config(
    page_title="YieldSync - Smart Farming",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SIDEBAR - LANGUAGE & INPUTS
with st.sidebar:
    st.title("⚙️ Settings / සැකසීම්")
    
    # Language selector
    language = st.radio(
        "Language / භාෂාව",
        options=['en', 'si'],
        format_func=lambda x: "English" if x == 'en' else "සිංහල"
    )
    
    lang = TRANSLATIONS[language]
    
    st.markdown("---")
    
    # Crop selection
    st.subheader(lang['crop'])
    crop_options = TARGET_CROPS
    if language == 'si':
        crop = st.selectbox(
            lang['crop'],
            options=crop_options,
            format_func=lambda x: f"{x} / {CROP_NAMES_SI.get(x, x)}",
            label_visibility="collapsed"
        )
    else:
        crop = st.selectbox(lang['crop'], options=crop_options, label_visibility="collapsed")
    
    # Market selection
    st.subheader(lang['market'])
    markets = ['Colombo', 'Kandy', 'Galle', 'Jaffna', 'Gampaha', 'Kurunegala']
    market = st.selectbox(lang['market'], options=markets, label_visibility="collapsed")
    
    # Price input
    st.subheader(lang['current_price'])
    
    # Default prices
    default_prices = {'Rice': 120, 'Beetroot': 85, 'Raddish': 70, 'Red Onion': 200}
    current_price = st.number_input(
        lang['current_price'],
        min_value=1.0,
        max_value=1000.0,
        value=float(default_prices.get(crop, 100)),
        step=1.0,
        label_visibility="collapsed"
    )
    
    # Quantity input
    st.subheader(lang['quantity'])
    quantity_kg = st.number_input(
        lang['quantity'],
        min_value=1,
        max_value=10000,
        value=100,
        step=10,
        label_visibility="collapsed"
    )
    
    # Days since harvest
    st.subheader(lang['days_harvest'])
    days_since_harvest = st.number_input(
        lang['days_harvest'],
        min_value=0,
        max_value=365,
        value=0,
        step=1,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Get recommendation button
    get_rec = st.button(
        f"🎯 {lang['get_recommendation']}",
        type="primary",
        use_container_width=True
    )

# Title
st.title(lang['title'])
st.markdown(f"### {lang['subtitle']}")
st.markdown("---")

# Initialize decision engine
@st.cache_resource
def load_decision_engine():
    return DecisionEngine()

engine = load_decision_engine()

# GENERATE RECOMMENDATION
if get_rec or 'last_recommendation' in st.session_state:
    
    if get_rec:
        # Generate new recommendation
        with st.spinner('🔄 Analyzing market data... / වෙළඳපල දත්ත විශ්ලේෂණය කරමින්...'):
            recommendation = engine.make_decision(
                crop=crop,
                current_price=current_price,
                quantity_kg=quantity_kg,
                days_since_harvest=days_since_harvest,
                market=market
            )
            st.session_state['last_recommendation'] = recommendation
            st.session_state['last_inputs'] = {
                'crop': crop,
                'market': market,
                'current_price': current_price,
                'quantity_kg': quantity_kg
            }
    else:
        recommendation = st.session_state['last_recommendation']
        # Update input values from session
        if 'last_inputs' in st.session_state:
            inputs = st.session_state['last_inputs']
            crop = inputs['crop']
            market = inputs['market']
            current_price = inputs['current_price']
            quantity_kg = inputs['quantity_kg']
    
    # Display recommendation
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        decision_text = recommendation['decision']
        if language == 'si':
            if 'SELL' in decision_text:
                decision_text = lang['sell_now']
            else:
                days = recommendation.get('best_hold_days', 0)
                decision_text = f"{lang['hold']} {days} {lang['days']}"
        
        st.metric(
            label=lang['decision'],
            value=decision_text,
            delta=None
        )
    
    with col2:
        st.metric(
            label=lang['confidence'],
            value=f"{recommendation['confidence']:.0f}%"
        )
    
    with col3:
        profit_per_kg = recommendation['expected_profit_per_kg']
        st.metric(
            label=f"{lang['expected_profit']} ({lang['per_kg']})",
            value=f"{profit_per_kg:.2f} LKR",
            delta=f"{(profit_per_kg/current_price)*100:.1f}%" if profit_per_kg > 0 else None
        )
    
    with col4:
        total_profit = recommendation['expected_profit_total']
        st.metric(
            label=lang['total_profit'],
            value=f"{total_profit:.0f} LKR"
        )
    
    st.markdown("---")
    
    # Two columns: Chart + Reasoning
    col_chart, col_reason = st.columns([2, 1])
    
    with col_chart:
        st.subheader(f"📈 {lang['forecast_title']}")
        
        # Prepare data for chart
        forecast_data = recommendation['forecast']
        
        if forecast_data:
            dates = [datetime.now() + timedelta(days=d) for d in sorted(forecast_data.keys())]
            prices = [forecast_data[d] for d in sorted(forecast_data.keys())]
            
            # Add current price
            dates.insert(0, datetime.now())
            prices.insert(0, current_price)
            
            # Create plotly chart
            fig = go.Figure()
            
            # Current price marker
            fig.add_trace(go.Scatter(
                x=[dates[0]],
                y=[prices[0]],
                mode='markers',
                name='Current Price / වත්මන් මිල',
                marker=dict(size=15, color='red', symbol='star'),
                text=[f'Now: {prices[0]:.2f} LKR'],
                hovertemplate='<b>%{text}</b><extra></extra>'
            ))
            
            # Forecast line
            fig.add_trace(go.Scatter(
                x=dates,
                y=prices,
                mode='lines+markers',
                name='Forecast / අනාවැකිය',
                line=dict(color='green', width=3),
                marker=dict(size=8),
                text=[f'{p:.2f} LKR' for p in prices],
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>%{text}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f"{crop} - Price Forecast ({market})",
                xaxis_title="Date / දිනය",
                yaxis_title="Price (LKR/kg) / මිල (රු/කි.ග්‍රෑ)",
                hovermode='x unified',
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Forecast data not available / අනාවැකි දත්ත නොමැත")
    
    with col_reason:
        st.subheader(f"💡 {lang['reasoning']}")
        
        # Reasoning box
        if language == 'si':
            reasoning_text = recommendation['reasoning_sinhala']
        else:
            reasoning_text = recommendation['reasoning']
        
        st.info(reasoning_text)
        
        # Perishability warning
        if crop in PERISHABILITY:
            max_days = PERISHABILITY[crop].get('perishability_days') or PERISHABILITY[crop].get('shelf_life_days', 365)
            days_left = max_days - days_since_harvest
            
            if days_left < 30:
                if language == 'si':
                    warning = f"⚠️ අනතුරු ඇඟවීම: දින {days_left} කින් නරක් වේ!"
                else:
                    warning = f"⚠️ WARNING: Spoils in {days_left} days!"
                st.warning(warning)
        
        # Storage cost info
        st.markdown("---")
        st.markdown("**Storage Costs / ගබඩා වියදම්:**")
        
        if crop in PERISHABILITY:
            # Try both field names for compatibility
            cost_per_day = PERISHABILITY[crop].get('storage_cost_lkr_per_kg_per_day')
            if cost_per_day is None:
                cost_per_month = PERISHABILITY[crop].get('storage_cost', 5)
                cost_per_day = cost_per_month / 30
            
            if language == 'si':
                st.write(f"මාසික: රු {cost_per_day * 30:.2f}/කි.ග්‍රෑ")
                st.write(f"දෛනික: රු {cost_per_day:.2f}/කි.ග්‍රෑ")
            else:
                st.write(f"Monthly: {cost_per_day * 30:.2f} LKR/kg")
                st.write(f"Daily: {cost_per_day:.2f} LKR/kg")
    
    # Details expander
    with st.expander("📊 Detailed Analysis / විස්තරාත්මක විශ්ලේෂණය"):
        
        st.markdown("### All Holding Options / සියලු විකල්ප")
        
        if 'details' in recommendation and 'all_options' in recommendation['details']:
            options_df = pd.DataFrame(recommendation['details']['all_options'])
            
            if not options_df.empty:
                # Format columns
                display_df = options_df[[
                    'hold_days', 'future_price', 'profit_pct', 
                    'net_profit_per_kg', 'total_profit', 'passes_threshold'
                ]].copy()
                
                display_df.columns = [
                    'Hold Days', 'Future Price (LKR)', 'Profit %',
                    'Net Profit/kg (LKR)', 'Total Profit (LKR)', 'Profitable?'
                ]
                
                # Style the dataframe
                def highlight_best(row):
                    if row['Profitable?']:
                        return ['background-color: #d4edda'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(
                    display_df.style.apply(highlight_best, axis=1),
                    use_container_width=True
                )
            else:
                st.write("No viable holding options / ශුද්ධ විකල්ප නැත")

else:
    # Welcome screen
    st.info(f"""
    👋 **Welcome to YieldSync! / YieldSync වෙත සාදරයෙන් පිළිගනිමු!**
    
    {'Get AI-powered recommendations for when to sell your crops.' if language == 'en' else 'ඔබේ බෝග කවදා විකුණන්නද යන්න පිළිබඳ AI නිර්දේශ ලබා ගන්න.'}
    
    **{'How to use:' if language == 'en' else 'භාවිතා කරන්නේ කෙසේද:'}**
    
    {'1. Select your crop and market in the sidebar' if language == 'en' else '1. පැත්තේ තීරුවෙන් බෝගය සහ වෙළඳපොළ තෝරන්න'}
    
    {'2. Enter current market price and quantity' if language == 'en' else '2. වත්මන් වෙළඳපල මිල සහ ප්‍රමාණය ඇතුළත් කරන්න'}
    
    {'3. Click "Get Recommendation"' if language == 'en' else '3. "නිර්දේශය ලබා ගන්න" ක්ලික් කරන්න'}
    
    {'4. View your personalized SELL/HOLD recommendation' if language == 'en' else '4. ඔබගේ පුද්ගලික විකුණුම්/තබා ගැනීමේ නිර්දේශය බලන්න'}
    """)
    
    # Sample data visualization
    st.markdown("---")
    st.subheader("📊 Sample Price Trends / නියැදි මිල ප්‍රවණතා")
    
    # Show a sample chart
    sample_dates = pd.date_range(start='2024-01-01', end='2024-10-15', freq='W')
    sample_prices = {
        'Rice': [115 + i*0.5 + np.random.randn()*3 for i in range(len(sample_dates))],
        'Beetroot': [80 + np.sin(i/4)*10 + np.random.randn()*5 for i in range(len(sample_dates))],
        'Red Onion': [180 + i*1.2 + np.random.randn()*15 for i in range(len(sample_dates))],
    }
    
    fig = go.Figure()
    for crop_name, prices in sample_prices.items():
        fig.add_trace(go.Scatter(
            x=sample_dates,
            y=prices,
            mode='lines',
            name=f"{crop_name} / {CROP_NAMES_SI.get(crop_name, crop_name)}",
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title="2024 Price Trends / 2024 මිල ප්‍රවණතා",
        xaxis_title="Date / දිනය",
        yaxis_title="Price (LKR/kg) / මිල (රු/කි.ග්‍රෑ)",
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# FOOTER
# ==============================================================================

st.markdown("---")
if language == 'en':
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🌾 YieldSync - Empowering Sri Lankan Farmers with AI</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🌾 YieldSync - AI තාක්ෂණයෙන් ශ්‍රී ලංකා ගොවීන් සවිබල ගැන්වීම</p>
    
    </div>
    """, unsafe_allow_html=True)
