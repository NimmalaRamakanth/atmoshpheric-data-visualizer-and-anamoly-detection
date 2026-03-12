"""
utils.py
Utility functions: AQI classification, risk level, CSV storage.
"""

import pandas as pd
import os
from datetime import datetime

DATA_DIR = "data"


def calculate_aqi_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify AQI category based on PM2.5 concentration (µg/m³).
    Uses US EPA breakpoints adapted for hourly data.
    """
    def classify(pm25: float) -> str:
        if pd.isna(pm25) or pm25 < 0:
            return "Unknown"
        elif pm25 <= 12.0:
            return "Good"
        elif pm25 <= 35.4:
            return "Moderate"
        elif pm25 <= 55.4:
            return "Unhealthy for Sensitive Groups"
        elif pm25 <= 150.4:
            return "Unhealthy"
        elif pm25 <= 250.4:
            return "Very Unhealthy"
        else:
            return "Hazardous"

    df = df.copy()
    if "pm2_5" in df.columns:
        df["aqi_category"] = df["pm2_5"].apply(classify)
    else:
        df["aqi_category"] = "Unknown"
    return df


def calculate_risk_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate environmental risk level based on combined atmospheric parameters.
    Considers PM2.5, PM10, temperature extremes, and wind speed.
    """
    def risk(row) -> str:
        score = 0

        # PM2.5 contribution
        pm25 = row.get("pm2_5", 0) or 0
        if pm25 > 55:
            score += 3
        elif pm25 > 35:
            score += 2
        elif pm25 > 12:
            score += 1

        # PM10 contribution
        pm10 = row.get("pm10", 0) or 0
        if pm10 > 150:
            score += 2
        elif pm10 > 54:
            score += 1

        # Temperature extremes
        temp = row.get("temperature_2m", 20) or 20
        if temp > 40 or temp < -20:
            score += 2
        elif temp > 35 or temp < -10:
            score += 1

        # Wind speed
        wind = row.get("windspeed_10m", 0) or 0
        if wind > 60:
            score += 2
        elif wind > 40:
            score += 1

        # Classify
        if score >= 5:
            return "High Risk"
        elif score >= 2:
            return "Moderate Risk"
        else:
            return "Low Risk"

    df = df.copy()
    df["risk_level"] = df.apply(risk, axis=1)
    return df


def save_to_csv(df: pd.DataFrame, city_name: str) -> str:
    """Save DataFrame to a city-specific CSV file, appending new data."""
    os.makedirs(DATA_DIR, exist_ok=True)
    safe_city = city_name.lower().replace(" ", "_").replace(",", "")
    filepath = os.path.join(DATA_DIR, f"{safe_city}_atmospheric_data.csv")

    if os.path.exists(filepath):
        existing = pd.read_csv(filepath, parse_dates=["time"])
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["time"]).sort_values("time")
    else:
        combined = df

    combined.to_csv(filepath, index=False)
    print(f"Data saved to {filepath} ({len(combined)} total records)")
    return filepath


def load_historical_data(city_name: str) -> pd.DataFrame | None:
    """Load historical data for a city from CSV."""
    safe_city = city_name.lower().replace(" ", "_").replace(",", "")
    filepath = os.path.join(DATA_DIR, f"{safe_city}_atmospheric_data.csv")

    if not os.path.exists(filepath):
        return None

    df = pd.read_csv(filepath, parse_dates=["time"])
    return df


def format_datetime(dt) -> str:
    """Format datetime for display."""
    if isinstance(dt, str):
        dt = pd.to_datetime(dt)
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def get_aqi_color(category: str) -> str:
    """Return hex color for AQI category."""
    colors = {
        "Good": "#00c851",
        "Moderate": "#adff2f",
        "Unhealthy for Sensitive Groups": "#ffbb33",
        "Unhealthy": "#ff8800",
        "Very Unhealthy": "#cc0000",
        "Hazardous": "#7d0023",
        "Unknown": "#8892a4",
    }
    return colors.get(category, "#8892a4")
