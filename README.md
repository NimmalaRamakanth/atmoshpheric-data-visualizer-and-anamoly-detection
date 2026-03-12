# 🌍 Atmospheric Data Visualizer & Anomaly Detection System

A real-time atmospheric monitoring dashboard built with **Python** and **Streamlit**.  
It collects data from Open-Meteo APIs, visualizes environmental conditions, detects anomalies using machine learning, and forecasts temperature trends.

---

## 🚀 Features

| Feature | Details |
|---|---|
| 📡 Real-time Data | Open-Meteo Weather + Air Quality APIs |
| 🤖 Anomaly Detection | Isolation Forest (scikit-learn) |
| 📈 Temperature Forecast | Random Forest Regressor (next 1–48 hours) |
| 🌫️ AQI Classification | 6-tier US EPA PM2.5 breakpoints |
| ⚠️ Risk Levels | Low / Moderate / High based on combined parameters |
| 💾 Data Storage | Local CSV per city, auto-deduplicating |
| 📊 Interactive Charts | Plotly scatter, line, area, heatmap, pie, bar |
| ⬇️ Export | Download anomaly reports & forecasts as CSV |

---

## 📦 Installation

### 1. Clone / unzip the project

```bash
cd atmospheric_visualizer
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501** in your browser.

---

## 🖥️ Usage

1. Enter a **city name** in the sidebar (e.g. `London`, `New York`, `Tokyo`).
2. Adjust **Historical Days** (1–7), **Anomaly Sensitivity**, and **Forecast Hours**.
3. Click **🚀 Fetch & Analyze Data**.
4. Explore the five dashboard tabs:
   - **🌡️ Temperature** — scatter plot with anomaly markers + humidity/pressure subplots
   - **🌫️ Air Quality** — PM2.5/PM10 trend with WHO guidelines + wind speed
   - **📊 AQI Distribution** — bar + pie + risk level charts
   - **⚠️ Anomalies** — feature correlation heatmap + anomaly table + download
   - **🔮 Forecast** — next N-hour temperature prediction with uncertainty band

---

## 📁 Project Structure

```
atmospheric_visualizer/
├── app.py               # Streamlit dashboard (main entry point)
├── data_collector.py    # Open-Meteo API integration & data merging
├── anomaly_detector.py  # Isolation Forest anomaly detection
├── forecaster.py        # Random Forest temperature forecasting
├── utils.py             # AQI/risk classification, CSV I/O
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── data/                # Auto-created; stores city CSVs
└── models/              # Auto-created; stores trained ML models
```

---

## 🌐 APIs Used

| API | Endpoint | Data |
|---|---|---|
| Open-Meteo Weather | `api.open-meteo.com/v1/forecast` | Temperature, Humidity, Pressure, Wind |
| Open-Meteo Air Quality | `air-quality-api.open-meteo.com/v1/air-quality` | PM2.5, PM10, CO, NO₂, O₃ |
| Open-Meteo Geocoding | `geocoding-api.open-meteo.com/v1/search` | City → Lat/Lon |

All APIs are **free** and require **no API key**.

---

## 🧠 Machine Learning

### Anomaly Detection — Isolation Forest
- Features: temperature, humidity, pressure, wind speed, PM2.5, PM10
- Preprocessing: `StandardScaler` normalization
- Contamination: adjustable via sidebar (default 5%)
- Output: `-1` = anomaly, `1` = normal, plus anomaly score

### Temperature Forecasting — Random Forest Regressor
- Features: timestamp, hour, day, month, day-of-week, cyclic encodings, lag features (1h/3h/6h/24h), rolling means
- Output: predicted temperature + 95% confidence band
- Horizon: configurable 6–48 hours

---

## 📋 AQI Categories (PM2.5)

| PM2.5 (µg/m³) | Category |
|---|---|
| 0 – 12.0 | 🟢 Good |
| 12.1 – 35.4 | 🟡 Moderate |
| 35.5 – 55.4 | 🟠 Unhealthy for Sensitive Groups |
| 55.5 – 150.4 | 🔴 Unhealthy |
| 150.5 – 250.4 | 🟣 Very Unhealthy |
| > 250.5 | ⚫ Hazardous |

---

## 🛠️ Technologies

`Python 3.10+` · `Streamlit` · `Pandas` · `NumPy` · `Plotly` · `scikit-learn` · `Joblib` · `Requests`

---

## 📄 License

MIT License — free to use, modify, and distribute.
