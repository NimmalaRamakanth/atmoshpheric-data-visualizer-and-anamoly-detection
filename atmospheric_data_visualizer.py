import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from sklearn.ensemble import IsolationForest
from datetime import datetime

st.set_page_config(page_title="Atmospheric Data visualizer And Anomaly Detector", layout="wide")

st.title("🌍 Atmospheric Data Visualizer & Anomaly Detector")
st.markdown("Live tracking and anomaly detection for temperature, pressure, and humidity.")

st.sidebar.header("Location Settings")
city = st.sidebar.text_input("Enter City Name", "hyderabad")

def get_coords(city_name):
    try:
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=en&format=json"
        res = requests.get(url).json()
        if 'results' in res:
            return res['results'][0]['latitude'], res['results'][0]['longitude']
        return None, None
    except:
        return None, None

lat, lon = get_coords(city)

if lat and lon:
    st.sidebar.success(f"Coordinates: {lat}, {lon}")
    
    @st.cache_data(ttl=3600)  
    def fetch_weather_data(lat, lon):
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,surface_pressure&past_days=7"
        response = requests.get(url).json()
        df = pd.DataFrame(response['hourly'])
        df['time'] = pd.to_datetime(df['time'])
        return df

    data = fetch_weather_data(lat, lon)

    st.subheader("🔍 Anomaly Detection Analysis")
    contamination = st.slider("Anomaly Sensitivity (Contamination)", 0.01, 0.10, 0.05)
    
    model = IsolationForest(contamination=contamination, random_state=42)
    features = data[['temperature_2m', 'surface_pressure']]
    data['anomaly_score'] = model.fit_predict(features)
    
    data['is_anomaly'] = data['anomaly_score'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Temperature Trend")
        fig_temp = px.line(data, x='time', y='temperature_2m', title=f"Temperature in {city}")
        anomalies = data[data['is_anomaly'] == 'Anomaly']
        fig_temp.add_scatter(x=anomalies['time'], y=anomalies['temperature_2m'], 
                             mode='markers', name='Anomaly', marker=dict(color='red', size=8))
        st.plotly_chart(fig_temp, use_container_width=True)

    with col2:
        st.write("### Pressure vs Humidity")
        fig_scatter = px.scatter(data, x='surface_pressure', y='relative_humidity_2m', 
                                 color='is_anomaly', title="Anomaly Clustering",
                                 color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'})
        st.plotly_chart(fig_scatter, use_container_width=True)

    with st.expander("View Raw Atmospheric Data"):
        st.write(data)

else:
    st.error("City not found. Please check the spelling.")
