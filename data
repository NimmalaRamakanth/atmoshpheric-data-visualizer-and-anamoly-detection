"""
data_collector.py
Fetches real-time atmospheric and air quality data from Open-Meteo APIs.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


def get_coordinates(city_name: str):
    """Resolve city name to lat/lon using Open-Meteo Geocoding API."""
    try:
        resp = requests.get(GEOCODING_URL, params={"name": city_name, "count": 1, "language": "en", "format": "json"}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if "results" not in data or not data["results"]:
            return None, None, None
        result = data["results"][0]
        return result["latitude"], result["longitude"], result.get("name", city_name)
    except Exception as e:
        print(f"Geocoding error: {e}")
        return None, None, None


def fetch_weather_data(lat: float, lon: float, days_back: int = 3) -> pd.DataFrame:
    """Fetch hourly weather data from Open-Meteo."""
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days_back)

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relativehumidity_2m,pressure_msl,windspeed_10m,precipitation",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "timezone": "UTC",
    }

    try:
        resp = requests.get(WEATHER_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        hourly = data.get("hourly", {})
        if not hourly:
            return pd.DataFrame()
        df = pd.DataFrame(hourly)
        df["time"] = pd.to_datetime(df["time"])
        return df
    except Exception as e:
        print(f"Weather API error: {e}")
        return pd.DataFrame()


def fetch_air_quality_data(lat: float, lon: float, days_back: int = 3) -> pd.DataFrame:
    """Fetch hourly air quality data from Open-Meteo Air Quality API."""
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days_back)

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,ozone",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "timezone": "UTC",
    }

    try:
        resp = requests.get(AIR_QUALITY_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        hourly = data.get("hourly", {})
        if not hourly:
            return pd.DataFrame()
        df = pd.DataFrame(hourly)
        df["time"] = pd.to_datetime(df["time"])
        return df
    except Exception as e:
        print(f"Air Quality API error: {e}")
        return pd.DataFrame()


def fetch_atmospheric_data(city_name: str, days_back: int = 3) -> pd.DataFrame | None:
    """
    Main entry point — fetches and merges weather + air quality data.
    Returns a merged DataFrame or None on failure.
    """
    lat, lon, resolved_name = get_coordinates(city_name)
    if lat is None:
        print(f"Could not resolve coordinates for city: {city_name}")
        return None

    print(f"Fetching data for {resolved_name} ({lat:.4f}, {lon:.4f})")

    weather_df = fetch_weather_data(lat, lon, days_back=days_back)
    aq_df = fetch_air_quality_data(lat, lon, days_back=days_back)

    if weather_df.empty:
        print("Weather data fetch failed.")
        return None

    if not aq_df.empty:
        merged = pd.merge(weather_df, aq_df, on="time", how="left")
    else:
        merged = weather_df.copy()
        for col in ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "ozone"]:
            merged[col] = np.nan

    # Fill missing air quality with 0 for AQI computation
    for col in ["pm2_5", "pm10"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    # Drop rows where all key atmospheric params are missing
    key_cols = ["temperature_2m", "relativehumidity_2m", "pressure_msl"]
    merged = merged.dropna(subset=[c for c in key_cols if c in merged.columns])
    merged = merged.reset_index(drop=True)

    # Add metadata
    merged["city"] = resolved_name
    merged["latitude"] = lat
    merged["longitude"] = lon
    merged["fetch_timestamp"] = datetime.utcnow().isoformat()

    print(f"Fetched {len(merged)} records for {resolved_name}")
    return merged
