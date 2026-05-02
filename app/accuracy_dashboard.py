"""
YieldSync Model Accuracy Dashboard
====================================
Interactive dashboard to visualize model performance metrics.

Run with: streamlit run accuracy_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# Page config
st.set_page_config(
    page_title="YieldSync - Model Accuracy",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .excellent { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .good { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
    .moderate { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); }
    .needs-improvement { background: linear-gradient(135deg, #fc4a1a 0%, #f7b733 100%); }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center; color: #2E7D32;'>📊 YieldSync Model Accuracy Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; margin-bottom: 2rem;'>Comprehensive Performance Analysis of Price Forecasting Models</p>", unsafe_allow_html=True)

# ============================================================================
# MODEL PERFORMANCE DATA
# ============================================================================

# Actual per-market model metrics (from training)
model_metrics = {
    'Rice': {
        'Colombo': {'MAE': 12.5, 'RMSE': 15.5, 'R2': 0.82, 'model': 'LSTM'},
        'Anuradhapura': {'MAE': 13.2, 'RMSE': 16.8, 'R2': 0.79, 'model': 'LSTM'},
        'Dambulla': {'MAE': 11.8, 'RMSE': 14.9, 'R2': 0.84, 'model': 'LSTM'},
        'Kandy': {'MAE': 12.9, 'RMSE': 16.2, 'R2': 0.80, 'model': 'LSTM'},
        'overall': {'MAE': 12.6, 'RMSE': 15.9, 'R2': 0.81}
    },
    'Beetroot': {
        'Colombo': {'MAE': 15.2, 'RMSE': 19.3, 'R2': 0.91, 'model': 'RandomForest'},
        'Dambulla': {'MAE': 16.8, 'RMSE': 21.5, 'R2': 0.89, 'model': 'RandomForest'},
        'Bandarawela': {'MAE': 14.5, 'RMSE': 18.7, 'R2': 0.93, 'model': 'RandomForest'},
        'Nuwara Eliya': {'MAE': 7.7, 'RMSE': 11.2, 'R2': 0.984, 'model': 'RandomForest'},
        'overall': {'MAE': 13.6, 'RMSE': 17.7, 'R2': 0.92}
    },
    'Radish': {
        'Colombo': {'MAE': 8.5, 'RMSE': 11.8, 'R2': 0.88, 'model': 'RandomForest'},
        'Dambulla': {'MAE': 9.2, 'RMSE': 12.9, 'R2': 0.85, 'model': 'RandomForest'},
        'Kandy': {'MAE': 8.0, 'RMSE': 11.2, 'R2': 0.90, 'model': 'RandomForest'},
        'overall': {'MAE': 8.6, 'RMSE': 12.0, 'R2': 0.88}
    },
    'Red Onion': {
        'Colombo': {'MAE': 42.3, 'RMSE': 52.8, 'R2': 0.65, 'model': 'LightGBM'},
        'Dambulla': {'MAE': 38.9, 'RMSE': 48.2, 'R2': 0.72, 'model': 'LightGBM'},
        'Jaffna': {'MAE': 45.1, 'RMSE': 55.6, 'R2': 0.61, 'model': 'LightGBM'},
        'overall': {'MAE': 42.1, 'RMSE': 52.2, 'R2': 0.66}
    }
}

# Horizon-based accuracy (from evaluation_summary.md)
horizon_metrics = [
    {'Horizon': '7 days', 'Days': 7, 'MAE': 39.7, 'Use Case': 'Short-term decisions', 'Reliability': '⭐⭐⭐⭐⭐'},
    {'Horizon': '14 days', 'Days': 14, 'MAE': 50.9, 'Use Case': 'Planning sales', 'Reliability': '⭐⭐⭐⭐'},
    {'Horizon': '30 days', 'Days': 30, 'MAE': 67.3, 'Use Case': 'General trends', 'Reliability': '⭐⭐⭐'},
    {'Horizon': '60 days', 'Days': 60, 'MAE': 78.4, 'Use Case': 'Long-term planning', 'Reliability': '⭐⭐'},
    {'Horizon': '84 days', 'Days': 84, 'MAE': 89.1, 'Use Case': 'Seasonal patterns', 'Reliability': '⭐'}
]

# ============================================================================
# METRICS OVERVIEW
# ============================================================================

st.markdown("## 🎯 Overall Model Performance")

col1, col2, col3, col4 = st.columns(4)

crops = ['Rice', 'Beetroot', 'Radish', 'Red Onion']
for i, crop in enumerate(crops):
    metrics = model_metrics[crop]['overall']
    
    # Classify performance
    r2 = metrics['R2']
    if r2 >= 0.9:
        grade, css_class = "Excellent", "excellent"
    elif r2 >= 0.8:
        grade, css_class = "Very Good", "good"
    elif r2 >= 0.65:
        grade, css_class = "Good", "moderate"
    else:
        grade, css_class = "Moderate", "needs-improvement"
    
    with [col1, col2, col3, col4][i]:
        st.markdown(f"""
        <div class="metric-card {css_class}">
            <h3>{crop}</h3>
            <h2>{grade}</h2>
            <p><strong>R²:</strong> {r2:.3f}</p>
            <p><strong>MAE:</strong> ±{metrics['MAE']:.1f} LKR</p>
            <p><strong>RMSE:</strong> {metrics['RMSE']:.1f} LKR</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# DETAILED ANALYSIS TABS
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs(["📊 By Crop & Market", "📈 By Forecast Horizon", "🎯 Model Comparison", "💡 Insights"])

# TAB 1: By Crop & Market
with tab1:
    st.markdown("### Performance by Crop and Market")
    
    selected_crop = st.selectbox("Select Crop", crops)
    
    crop_data = model_metrics[selected_crop]
    markets = [k for k in crop_data.keys() if k != 'overall']
    
    # Create comparison chart
    market_df = pd.DataFrame([
        {
            'Market': market,
            'MAE': crop_data[market]['MAE'],
            'RMSE': crop_data[market]['RMSE'],
            'R²': crop_data[market]['R2'],
            'Model': crop_data[market]['model']
        }
        for market in markets
    ])
    
    # Metrics display
    cols = st.columns(len(markets))
    for i, market in enumerate(markets):
        with cols[i]:
            m = crop_data[market]
            st.metric(
                label=market,
                value=f"R²: {m['R2']:.3f}",
                delta=f"MAE: ±{m['MAE']:.1f} LKR"
            )
    
    # Visual comparison
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='MAE (LKR)',
        x=market_df['Market'],
        y=market_df['MAE'],
        text=market_df['MAE'].round(1),
        textposition='auto',
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Scatter(
        name='R² Score',
        x=market_df['Market'],
        y=market_df['R²'] * 100,  # Scale to percentage
        mode='lines+markers',
        yaxis='y2',
        marker_color='green',
        line=dict(width=3)
    ))
    
    fig.update_layout(
        title=f"{selected_crop} - Accuracy by Market",
        xaxis_title="Market",
        yaxis_title="MAE (LKR/kg)",
        yaxis2=dict(
            title="R² Score (%)",
            overlaying='y',
            side='right',
            range=[0, 100]
        ),
        template='plotly_white',
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.markdown("#### Detailed Metrics")
    st.dataframe(
        market_df.style.background_gradient(subset=['R²'], cmap='RdYlGn'),
        use_container_width=True
    )

# TAB 2: By Forecast Horizon
with tab2:
    st.markdown("### Accuracy vs Forecast Horizon")
    
    st.info("**Key Finding**: Accuracy decreases as forecast horizon increases. Best results within 7-14 days.")
    
    horizon_df = pd.DataFrame(horizon_metrics)
    
    # Horizon accuracy chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=horizon_df['Days'],
        y=horizon_df['MAE'],
        mode='lines+markers',
        name='MAE',
        line=dict(color='red', width=3),
        marker=dict(size=12)
    ))
    
    fig.update_layout(
        title="Mean Absolute Error vs Forecast Horizon",
        xaxis_title="Forecast Horizon (Days)",
        yaxis_title="MAE (LKR/kg)",
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Horizon table
    st.markdown("#### Forecast Horizon Performance")
    st.dataframe(
        horizon_df[['Horizon', 'MAE', 'Use Case', 'Reliability']],
        use_container_width=True,
        hide_index=True
    )
    
    # Recommendations
    st.success("✅ **Recommended**: Use 7-14 day forecasts for critical decisions")
    st.warning("⚠️ **Caution**: 30+ day forecasts are best for trend analysis, not exact prices")

# TAB 3: Model Comparison
with tab3:
    st.markdown("### Model Type Performance")
    
    # Aggregate by model type
    model_performance = {
        'LSTM': {'crops': [], 'avg_r2': [], 'avg_mae': []},
        'RandomForest': {'crops': [], 'avg_r2': [], 'avg_mae': []},
        'LightGBM': {'crops': [], 'avg_r2': [], 'avg_mae': []}
    }
    
    for crop, data in model_metrics.items():
        model_type = list(data.values())[0]['model']  # Get model type
        model_performance[model_type]['crops'].append(crop)
        model_performance[model_type]['avg_r2'].append(data['overall']['R2'])
        model_performance[model_type]['avg_mae'].append(data['overall']['MAE'])
    
    # Comparison
    comparison_data = []
    for model, stats in model_performance.items():
        if stats['crops']:
            comparison_data.append({
                'Model': model,
                'Crops': ', '.join(stats['crops']),
                'Avg R²': np.mean(stats['avg_r2']),
                'Avg MAE': np.mean(stats['avg_mae']),
                'Best For': 'High volume crops' if model == 'LightGBM' else ('Fast training' if model == 'RandomForest' else 'Time series')
            })
    
    comp_df = pd.DataFrame(comparison_data)
    
    # Display
    col1, col2, col3 = st.columns(3)
    for i, row in comp_df.iterrows():
        with [col1, col2, col3][i]:
            st.markdown(f"""
            <div class="metric-card good">
                <h3>{row['Model']}</h3>
                <p><strong>Avg R²:</strong> {row['Avg R²']:.3f}</p>
                <p><strong>Avg MAE:</strong> {row['Avg MAE']:.1f} LKR</p>
                <p style="font-size: 0.9em; margin-top: 1rem;">{row['Best For']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

# TAB 4: Insights
with tab4:
    st.markdown("### 💡 Key Insights & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ✅ Strengths")
        st.success("""
        - **Beetroot** has exceptional accuracy (R² = 0.92-0.98)
        - **Radish** models are highly reliable (R² = 0.88)
        - **Short-term forecasts** (7-14 days) are very accurate
        - Per-market models improve local prediction accuracy
        - RandomForest models show excellent stability
        """)
        
        st.markdown("#### 📊 Performance Grades")
        grades_df = pd.DataFrame([
            {'Crop': 'Beetroot', 'Grade': 'A+', 'Confidence': '98%', 'Production Ready': '✅'},
            {'Crop': 'Radish', 'Grade': 'A', 'Confidence': '88%', 'Production Ready': '✅'},
            {'Crop': 'Rice', 'Grade': 'B+', 'Confidence': '81%', 'Production Ready': '✅'},
            {'Crop': 'Red Onion', 'Grade': 'B', 'Confidence': '66%', 'Production Ready': '⚠️'},
        ])
        st.dataframe(grades_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### 🎯 Recommendations")
        st.info("""
        **For Users:**
        - Trust 7-14 day forecasts for selling decisions
        - Use longer forecasts (30+ days) for trend awareness only
        - Red Onion predictions have wider confidence intervals
        - Cross-check predictions with actual market visits
        
        **For Developers:**
        - Rice model could benefit from more training data
        - Consider ensemble methods for Red Onion (high volatility)
        - Retrain monthly to capture seasonal patterns
        - Add external factors (fuel prices, festivals) to improve accuracy
        """)
        
        st.markdown("#### ⚠️ Known Limitations")
        st.warning("""
        - Red Onion has high price volatility (MAE ±42 LKR)
        - Long-term forecasts (60+ days) have limited accuracy
        - Weather impact varies by crop and season
        - Market disruptions (strikes, fuel shortages) not predicted
        """)
    
    st.markdown("---")
    
    # Confidence Intervals
    st.markdown("### 📏 Prediction Confidence Intervals (95%)")
    
    confidence_data = []
    for crop, data in model_metrics.items():
        rmse = data['overall']['RMSE']
        margin = 1.96 * rmse
        confidence_data.append({
            'Crop': crop,
            'RMSE': f"±{rmse:.1f} LKR",
            '95% Confidence': f"±{margin:.1f} LKR",
            'Example (100 LKR)': f"{100-margin:.0f}-{100+margin:.0f} LKR"
        })
    
    conf_df = pd.DataFrame(confidence_data)
    st.dataframe(conf_df, use_container_width=True, hide_index=True)
    
    st.caption("*95% Confidence Interval means predictions will fall within this range 95% of the time*")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>📊 <strong>Model Accuracy Dashboard</strong> - Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    <p>YieldSync v2.0 | Comprehensive Price Forecasting System</p>
</div>
""", unsafe_allow_html=True)
