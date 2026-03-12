import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

from data_collector import fetch_atmospheric_data
from anomaly_detector import detect_anomalies
from forecaster import forecast_temperature
from utils import calculate_aqi_category, calculate_risk_level, save_to_csv, load_historical_data

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Atmospheric Data Visualizer & Anomaly Detection",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3a);
        border: 1px solid #2d3448;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #00d4ff; }
    .metric-label { font-size: 0.85rem; color: #8892a4; margin-top: 4px; }
    .anomaly-badge {
        background: linear-gradient(135deg, #ff4444, #cc0000);
        color: white;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .normal-badge {
        background: linear-gradient(135deg, #00c851, #007e33);
        color: white;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #e0e6f0;
        border-left: 4px solid #00d4ff;
        padding-left: 12px;
        margin: 20px 0 15px 0;
    }
    .risk-low { color: #00c851; font-weight: 700; }
    .risk-moderate { color: #ffbb33; font-weight: 700; }
    .risk-high { color: #ff4444; font-weight: 700; }
    .stButton>button {
        background: linear-gradient(135deg, #00d4ff, #0088cc);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,212,255,0.4); }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #131722, #1a1f2e); }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 30px 0 10px 0;'>
    <h1 style='font-size:2.5rem; font-weight:800; background: linear-gradient(135deg, #00d4ff, #0088cc, #00ff88);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0;'>
        🌍 Atmospheric Data Visualizer
    </h1>
    <p style='color:#8892a4; font-size:1.05rem; margin-top:8px;'>
        Real-time Monitoring · Anomaly Detection · Temperature Forecasting
    </p>
</div>
<hr style='border:none; border-top:1px solid #2d3448; margin-bottom:20px;'>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 Configuration")
    st.markdown("---")
    city_name = st.text_input("🏙️ Enter City Name", value="London", placeholder="e.g. New York, Tokyo...")
    
    st.markdown("**📅 Data Range**")
    days_back = st.slider("Historical Days", min_value=1, max_value=7, value=3)
    
    st.markdown("**⚙️ Model Settings**")
    contamination = st.slider("Anomaly Sensitivity", min_value=0.01, max_value=0.2, value=0.05, step=0.01,
                               help="Higher = more anomalies detected")
    forecast_hours = st.slider("Forecast Hours", min_value=6, max_value=48, value=24)
    
    st.markdown("---")
    fetch_btn = st.button("🚀 Fetch & Analyze Data", use_container_width=True)
    
    st.markdown("---")
    st.markdown("""
    <div style='color:#8892a4; font-size:0.8rem; text-align:center;'>
        <p>📡 Data: Open-Meteo APIs</p>
        <p>🤖 ML: Isolation Forest</p>
        <p>📈 Forecast: Random Forest</p>
    </div>
    """, unsafe_allow_html=True)

# ── Main Logic ────────────────────────────────────────────────────────────────
if fetch_btn:
    with st.spinner(f"🌐 Fetching atmospheric data for **{city_name}**..."):
        df = fetch_atmospheric_data(city_name, days_back=days_back)
    
    if df is None or df.empty:
        st.error("❌ Failed to fetch data. Please check the city name and try again.")
        st.stop()

    # Save & enrich
    df = calculate_aqi_category(df)
    df = calculate_risk_level(df)
    save_to_csv(df, city_name)
    
    with st.spinner("🤖 Running anomaly detection..."):
        df = detect_anomalies(df, contamination=contamination)
    
    with st.spinner("📈 Forecasting temperatures..."):
        forecast_df = forecast_temperature(df, hours=forecast_hours)

    # ── Summary Metrics ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Summary Dashboard</div>', unsafe_allow_html=True)
    
    latest = df.iloc[-1]
    total_records = len(df)
    anomaly_count = int((df['anomaly'] == -1).sum()) if 'anomaly' in df.columns else 0
    anomaly_pct = round(anomaly_count / total_records * 100, 1)
    
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    metrics = [
        (c1, "🌡️", f"{latest['temperature_2m']:.1f}°C", "Temperature"),
        (c2, "💧", f"{latest['relativehumidity_2m']:.0f}%", "Humidity"),
        (c3, "🌬️", f"{latest['pressure_msl']:.0f} hPa", "Pressure"),
        (c4, "💨", f"{latest['windspeed_10m']:.1f} km/h", "Wind Speed"),
        (c5, "🔴", f"{total_records}", f"Records | {anomaly_count} anomalies ({anomaly_pct}%)"),
        (c6, "⚠️", f"{anomaly_pct}%", "Anomaly Rate"),
    ]
    for col, icon, val, label in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style='font-size:1.5rem'>{icon}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Air Quality Info ──────────────────────────────────────────────────────
    aqi_cat = latest.get('aqi_category', 'N/A')
    risk_lvl = latest.get('risk_level', 'N/A')
    pm25 = latest.get('pm2_5', 0)
    pm10 = latest.get('pm10', 0)
    
    risk_color = {"Low Risk": "#00c851", "Moderate Risk": "#ffbb33", "High Risk": "#ff4444"}.get(risk_lvl, "#8892a4")
    aqi_color = {"Good": "#00c851", "Moderate": "#adff2f", "Unhealthy for Sensitive Groups": "#ffbb33",
                 "Unhealthy": "#ff8800", "Very Unhealthy": "#cc0000", "Hazardous": "#7d0023"}.get(aqi_cat, "#8892a4")
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg,#1e2130,#252a3a); border:1px solid #2d3448;
                border-radius:12px; padding:20px; margin-bottom:20px;'>
        <h4 style='color:#e0e6f0; margin:0 0 15px 0;'>🌫️ Current Air Quality — {city_name}</h4>
        <div style='display:flex; gap:30px; flex-wrap:wrap;'>
            <div><span style='color:#8892a4'>AQI Category:</span>
                 <span style='color:{aqi_color}; font-weight:700; margin-left:8px;'>{aqi_cat}</span></div>
            <div><span style='color:#8892a4'>Risk Level:</span>
                 <span style='color:{risk_color}; font-weight:700; margin-left:8px;'>{risk_lvl}</span></div>
            <div><span style='color:#8892a4'>PM2.5:</span>
                 <span style='color:#00d4ff; font-weight:700; margin-left:8px;'>{pm25:.1f} µg/m³</span></div>
            <div><span style='color:#8892a4'>PM10:</span>
                 <span style='color:#00d4ff; font-weight:700; margin-left:8px;'>{pm10:.1f} µg/m³</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Visualizations ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Interactive Visualizations</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🌡️ Temperature", "🌫️ Air Quality", "📊 AQI Distribution", "⚠️ Anomalies", "🔮 Forecast"])

    with tab1:
        fig = go.Figure()
        normal = df[df['anomaly'] != -1] if 'anomaly' in df.columns else df
        anomalies = df[df['anomaly'] == -1] if 'anomaly' in df.columns else pd.DataFrame()
        
        fig.add_trace(go.Scatter(x=normal['time'], y=normal['temperature_2m'],
            mode='markers', name='Normal', marker=dict(color='#00d4ff', size=5, opacity=0.7)))
        if not anomalies.empty:
            fig.add_trace(go.Scatter(x=anomalies['time'], y=anomalies['temperature_2m'],
                mode='markers', name='Anomaly', marker=dict(color='#ff4444', size=10, symbol='x', line=dict(width=2))))
        fig.add_trace(go.Scatter(x=df['time'], y=df['temperature_2m'].rolling(6).mean(),
            mode='lines', name='6h Moving Avg', line=dict(color='#00ff88', width=2)))
        
        fig.update_layout(title=f"Temperature Anomaly Scatter — {city_name}",
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30,33,48,0.8)',
            xaxis_title="Time", yaxis_title="Temperature (°C)", height=420)
        st.plotly_chart(fig, use_container_width=True)

        # Humidity + Pressure subplot
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             subplot_titles=("Relative Humidity (%)", "Atmospheric Pressure (hPa)"))
        fig2.add_trace(go.Scatter(x=df['time'], y=df['relativehumidity_2m'],
            fill='tozeroy', fillcolor='rgba(0,136,204,0.15)', line=dict(color='#0088cc'), name='Humidity'), row=1, col=1)
        fig2.add_trace(go.Scatter(x=df['time'], y=df['pressure_msl'],
            line=dict(color='#ff8c00'), name='Pressure'), row=2, col=1)
        fig2.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                           plot_bgcolor='rgba(30,33,48,0.8)', height=380, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df['time'], y=df['pm2_5'],
            mode='lines+markers', name='PM2.5', line=dict(color='#ff6b6b', width=2),
            marker=dict(size=4)))
        fig3.add_trace(go.Scatter(x=df['time'], y=df['pm10'],
            mode='lines+markers', name='PM10', line=dict(color='#ffd93d', width=2),
            marker=dict(size=4)))
        
        # WHO guidelines
        fig3.add_hline(y=15, line_dash="dash", line_color="#ff4444", annotation_text="WHO PM2.5 Annual Limit (15 µg/m³)")
        fig3.add_hline(y=45, line_dash="dash", line_color="#ffbb33", annotation_text="WHO PM10 Annual Limit (45 µg/m³)")
        
        fig3.update_layout(title=f"PM2.5 & PM10 Trend — {city_name}", template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30,33,48,0.8)',
            xaxis_title="Time", yaxis_title="Concentration (µg/m³)", height=420)
        st.plotly_chart(fig3, use_container_width=True)

        # Wind speed
        fig4 = px.area(df, x='time', y='windspeed_10m', title=f"Wind Speed — {city_name}",
                       template='plotly_dark', color_discrete_sequence=['#00ff88'])
        fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30,33,48,0.8)',
                           xaxis_title="Time", yaxis_title="Wind Speed (km/h)", height=300)
        st.plotly_chart(fig4, use_container_width=True)

    with tab3:
        aqi_counts = df['aqi_category'].value_counts().reset_index()
        aqi_counts.columns = ['AQI Category', 'Count']
        color_map = {"Good":"#00c851","Moderate":"#adff2f","Unhealthy for Sensitive Groups":"#ffbb33",
                     "Unhealthy":"#ff8800","Very Unhealthy":"#cc0000","Hazardous":"#7d0023"}
        
        col_a, col_b = st.columns(2)
        with col_a:
            fig5 = px.bar(aqi_counts, x='AQI Category', y='Count',
                title="AQI Category Distribution", template='plotly_dark',
                color='AQI Category', color_discrete_map=color_map)
            fig5.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30,33,48,0.8)', height=380)
            st.plotly_chart(fig5, use_container_width=True)
        with col_b:
            fig6 = px.pie(aqi_counts, names='AQI Category', values='Count',
                title="AQI Category Share", template='plotly_dark',
                color='AQI Category', color_discrete_map=color_map, hole=0.4)
            fig6.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=380)
            st.plotly_chart(fig6, use_container_width=True)

        # Risk level distribution
        if 'risk_level' in df.columns:
            risk_counts = df['risk_level'].value_counts().reset_index()
            risk_counts.columns = ['Risk Level', 'Count']
            risk_color_map = {"Low Risk":"#00c851","Moderate Risk":"#ffbb33","High Risk":"#ff4444"}
            fig7 = px.bar(risk_counts, x='Risk Level', y='Count', title="Risk Level Distribution",
                          template='plotly_dark', color='Risk Level', color_discrete_map=risk_color_map)
            fig7.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30,33,48,0.8)', height=320)
            st.plotly_chart(fig7, use_container_width=True)

    with tab4:
        if 'anomaly' in df.columns:
            anomaly_df = df[df['anomaly'] == -1].copy()
            st.markdown(f"**{len(anomaly_df)} anomalies detected** out of {total_records} records ({anomaly_pct}%)")
            
            if not anomaly_df.empty:
                # Correlation heatmap of features
                feature_cols = ['temperature_2m','relativehumidity_2m','pressure_msl','windspeed_10m','pm2_5','pm10']
                avail_cols = [c for c in feature_cols if c in df.columns]
                corr = df[avail_cols].corr()
                
                fig8 = px.imshow(corr, title="Feature Correlation Heatmap",
                    template='plotly_dark', color_continuous_scale='RdBu_r', aspect='auto')
                fig8.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=380)
                st.plotly_chart(fig8, use_container_width=True)
                
                # Anomaly table
                st.markdown("**📋 Anomaly Records**")
                display_cols = ['time','temperature_2m','relativehumidity_2m','pressure_msl',
                                'windspeed_10m','pm2_5','pm10','aqi_category','risk_level']
                display_cols = [c for c in display_cols if c in anomaly_df.columns]
                st.dataframe(anomaly_df[display_cols].reset_index(drop=True), use_container_width=True)
                
                csv_anomalies = anomaly_df.to_csv(index=False)
                st.download_button("⬇️ Download Anomaly Report (CSV)", csv_anomalies,
                    file_name=f"anomaly_report_{city_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv")
            else:
                st.success("✅ No anomalies detected in the current dataset!")
        else:
            st.info("Anomaly detection data not available.")

    with tab5:
        if forecast_df is not None and not forecast_df.empty:
            fig9 = go.Figure()
            fig9.add_trace(go.Scatter(x=df['time'].tail(48), y=df['temperature_2m'].tail(48),
                mode='lines', name='Historical', line=dict(color='#00d4ff', width=2)))
            fig9.add_trace(go.Scatter(x=forecast_df['time'], y=forecast_df['temperature_forecast'],
                mode='lines', name='Forecast', line=dict(color='#ff6b6b', width=2, dash='dash'),
                fill='tonexty' if False else None))
            
            # Uncertainty band
            if 'temp_upper' in forecast_df.columns:
                fig9.add_trace(go.Scatter(x=pd.concat([forecast_df['time'], forecast_df['time'][::-1]]),
                    y=pd.concat([forecast_df['temp_upper'], forecast_df['temp_lower'][::-1]]),
                    fill='toself', fillcolor='rgba(255,107,107,0.15)',
                    line=dict(color='rgba(255,255,255,0)'), name='Uncertainty Band'))
            
            fig9.update_layout(title=f"Temperature Forecast — Next {forecast_hours}h — {city_name}",
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30,33,48,0.8)',
                xaxis_title="Time", yaxis_title="Temperature (°C)", height=420)
            st.plotly_chart(fig9, use_container_width=True)
            
            # Forecast stats
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                st.metric("Min Forecast Temp", f"{forecast_df['temperature_forecast'].min():.1f}°C")
            with fc2:
                st.metric("Max Forecast Temp", f"{forecast_df['temperature_forecast'].max():.1f}°C")
            with fc3:
                st.metric("Avg Forecast Temp", f"{forecast_df['temperature_forecast'].mean():.1f}°C")
            
            csv_forecast = forecast_df.to_csv(index=False)
            st.download_button("⬇️ Download Forecast Data (CSV)", csv_forecast,
                file_name=f"forecast_{city_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv")
        else:
            st.warning("⚠️ Not enough data to generate forecast. Try fetching more days.")

    # ── Raw Data ──────────────────────────────────────────────────────────────
    with st.expander("📄 View Raw Data"):
        st.dataframe(df, use_container_width=True)
        csv_raw = df.to_csv(index=False)
        st.download_button("⬇️ Download Full Dataset (CSV)", csv_raw,
            file_name=f"atmospheric_data_{city_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv")

else:
    # Welcome screen
    st.markdown("""
    <div style='text-align:center; padding:60px 20px;'>
        <div style='font-size:5rem; margin-bottom:20px;'>🌍</div>
        <h2 style='color:#e0e6f0; font-weight:700;'>Welcome to Atmospheric Data Visualizer</h2>
        <p style='color:#8892a4; font-size:1.1rem; max-width:600px; margin:0 auto 30px auto;'>
            Enter a city name in the sidebar and click <strong style='color:#00d4ff;'>Fetch &amp; Analyze Data</strong>
            to start monitoring real-time atmospheric conditions, detecting anomalies, and forecasting temperatures.
        </p>
        <div style='display:flex; gap:20px; justify-content:center; flex-wrap:wrap; margin-top:30px;'>
            <div style='background:linear-gradient(135deg,#1e2130,#252a3a); border:1px solid #2d3448; border-radius:12px; padding:25px; width:180px;'>
                <div style='font-size:2rem'>📡</div>
                <div style='color:#00d4ff; font-weight:600; margin-top:10px;'>Real-time Data</div>
                <div style='color:#8892a4; font-size:0.85rem; margin-top:5px;'>Open-Meteo API</div>
            </div>
            <div style='background:linear-gradient(135deg,#1e2130,#252a3a); border:1px solid #2d3448; border-radius:12px; padding:25px; width:180px;'>
                <div style='font-size:2rem'>🤖</div>
                <div style='color:#00d4ff; font-weight:600; margin-top:10px;'>ML Detection</div>
                <div style='color:#8892a4; font-size:0.85rem; margin-top:5px;'>Isolation Forest</div>
            </div>
            <div style='background:linear-gradient(135deg,#1e2130,#252a3a); border:1px solid #2d3448; border-radius:12px; padding:25px; width:180px;'>
                <div style='font-size:2rem'>📈</div>
                <div style='color:#00d4ff; font-weight:600; margin-top:10px;'>Forecasting</div>
                <div style='color:#8892a4; font-size:0.85rem; margin-top:5px;'>Random Forest</div>
            </div>
            <div style='background:linear-gradient(135deg,#1e2130,#252a3a); border:1px solid #2d3448; border-radius:12px; padding:25px; width:180px;'>
                <div style='font-size:2rem'>🌫️</div>
                <div style='color:#00d4ff; font-weight:600; margin-top:10px;'>Air Quality</div>
                <div style='color:#8892a4; font-size:0.85rem; margin-top:5px;'>AQI &amp; PM2.5/PM10</div>
            </div>
            <div style='background:linear-gradient(135deg,#1e2130,#252a3a); border:1px solid #2d3448; border-radius:12px; padding:25px; width:180px;'>
                <div style='font-size:2rem'>⚠️</div>
                <div style='color:#00d4ff; font-weight:600; margin-top:10px;'>Risk Levels</div>
                <div style='color:#8892a4; font-size:0.85rem; margin-top:5px;'>Environmental Risk</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
