"""
anomaly_detector.py
Anomaly detection using Isolation Forest with StandardScaler preprocessing.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "isolation_forest.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

FEATURE_COLS = [
    "temperature_2m",
    "relativehumidity_2m",
    "pressure_msl",
    "windspeed_10m",
    "pm2_5",
    "pm10",
]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and fill feature columns for anomaly detection."""
    avail = [c for c in FEATURE_COLS if c in df.columns]
    X = df[avail].copy()
    X = X.fillna(X.median())
    return X


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """
    Detect anomalies using Isolation Forest.
    Adds columns: 'anomaly' (-1 = anomaly, 1 = normal), 'anomaly_score'
    """
    df = df.copy()
    X = prepare_features(df)

    if len(X) < 10:
        df["anomaly"] = 1
        df["anomaly_score"] = 0.0
        return df

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train / apply Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        n_estimators=200,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    predictions = iso_forest.fit_predict(X_scaled)
    scores = iso_forest.decision_function(X_scaled)  # higher = more normal

    df["anomaly"] = predictions          # -1 = anomaly, 1 = normal
    df["anomaly_score"] = scores

    # Save models for reuse
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(iso_forest, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    n_anomalies = int((predictions == -1).sum())
    print(f"Anomaly detection complete: {n_anomalies}/{len(df)} anomalies detected ({n_anomalies/len(df)*100:.1f}%)")

    return df


def load_and_predict(df: pd.DataFrame) -> pd.DataFrame:
    """Load saved model and predict on new data (if model exists)."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return detect_anomalies(df)

    df = df.copy()
    X = prepare_features(df)
    scaler = joblib.load(SCALER_PATH)
    iso_forest = joblib.load(MODEL_PATH)

    X_scaled = scaler.transform(X)
    df["anomaly"] = iso_forest.predict(X_scaled)
    df["anomaly_score"] = iso_forest.decision_function(X_scaled)
    return df


def get_anomaly_summary(df: pd.DataFrame) -> dict:
    """Return a summary dict of anomaly statistics."""
    if "anomaly" not in df.columns:
        return {}

    total = len(df)
    n_anomalies = int((df["anomaly"] == -1).sum())
    pct = round(n_anomalies / total * 100, 2) if total > 0 else 0

    anomaly_df = df[df["anomaly"] == -1]
    summary = {
        "total_records": total,
        "anomaly_count": n_anomalies,
        "anomaly_percentage": pct,
        "avg_anomaly_temp": round(anomaly_df["temperature_2m"].mean(), 2) if not anomaly_df.empty else None,
        "avg_anomaly_pm25": round(anomaly_df["pm2_5"].mean(), 2) if not anomaly_df.empty and "pm2_5" in anomaly_df.columns else None,
    }
    return summary
