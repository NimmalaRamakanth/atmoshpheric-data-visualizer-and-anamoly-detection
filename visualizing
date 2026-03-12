"""
forecaster.py
Temperature forecasting using Random Forest Regressor.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
from datetime import timedelta

MODEL_DIR = "models"
FORECAST_MODEL_PATH = os.path.join(MODEL_DIR, "rf_forecaster.pkl")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer time-based features for the forecaster."""
    df = df.copy()
    df["timestamp"] = df["time"].astype(np.int64) // 10**9
    df["hour"] = df["time"].dt.hour
    df["day"] = df["time"].dt.day
    df["month"] = df["time"].dt.month
    df["dayofweek"] = df["time"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Lag features (if enough data)
    if len(df) >= 24:
        df["temp_lag_1h"] = df["temperature_2m"].shift(1)
        df["temp_lag_3h"] = df["temperature_2m"].shift(3)
        df["temp_lag_6h"] = df["temperature_2m"].shift(6)
        df["temp_lag_24h"] = df["temperature_2m"].shift(24)
        df["temp_rolling_3h"] = df["temperature_2m"].rolling(3).mean()
        df["temp_rolling_6h"] = df["temperature_2m"].rolling(6).mean()
    else:
        for col in ["temp_lag_1h","temp_lag_3h","temp_lag_6h","temp_lag_24h","temp_rolling_3h","temp_rolling_6h"]:
            df[col] = df["temperature_2m"]

    return df


FEATURE_COLS = [
    "timestamp", "hour", "day", "month", "dayofweek",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "temp_lag_1h", "temp_lag_3h", "temp_lag_6h", "temp_lag_24h",
    "temp_rolling_3h", "temp_rolling_6h",
]


def train_model(df: pd.DataFrame):
    """Train Random Forest Regressor on historical temperature data."""
    df = build_features(df)
    df = df.dropna(subset=FEATURE_COLS + ["temperature_2m"])

    if len(df) < 20:
        print("Not enough data to train forecast model.")
        return None, None

    X = df[FEATURE_COLS]
    y = df["temperature_2m"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Forecast model — MAE: {mae:.2f}°C, R²: {r2:.3f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, FORECAST_MODEL_PATH)

    return model, {"mae": mae, "r2": r2}


def forecast_temperature(df: pd.DataFrame, hours: int = 24) -> pd.DataFrame | None:
    """
    Forecast temperature for the next `hours` hours.
    Returns a DataFrame with 'time' and 'temperature_forecast' columns.
    """
    if df is None or len(df) < 20:
        return None

    model, metrics = train_model(df)
    if model is None:
        return None

    # Generate future timestamps
    last_time = df["time"].max()
    future_times = [last_time + timedelta(hours=i + 1) for i in range(hours)]
    future_df = pd.DataFrame({"time": future_times})

    # Seed future data using rolling average from recent history
    recent_avg = df["temperature_2m"].tail(6).mean()
    future_df["temperature_2m"] = recent_avg

    future_df = build_features(future_df)

    avail_features = [c for c in FEATURE_COLS if c in future_df.columns]
    X_future = future_df[avail_features].fillna(recent_avg)

    # Handle missing features
    for col in FEATURE_COLS:
        if col not in X_future.columns:
            X_future[col] = 0

    X_future = X_future[FEATURE_COLS]
    predictions = model.predict(X_future)

    # Simple uncertainty estimate
    all_preds = np.array([est.predict(X_future) for est in model.estimators_])
    std = all_preds.std(axis=0)

    forecast_df = pd.DataFrame({
        "time": future_times,
        "temperature_forecast": predictions,
        "temp_upper": predictions + 1.96 * std,
        "temp_lower": predictions - 1.96 * std,
    })

    print(f"Generated {hours}-hour temperature forecast.")
    return forecast_df
