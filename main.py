import streamlit as st
import pandas as pd
import numpy as np
import time
import random
import requests
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib


# Function to simulate live IoT sensor data
def generate_sensor_data():
    return {
        "Engine Temperature (°C)": random.randint(80, 120),
        "Battery Voltage (V)": round(random.uniform(11.0, 14.5), 2),
        "Oil Pressure (PSI)": random.randint(20, 70),
        "Tire Pressure (PSI)": random.randint(28, 35),
        "Vibration Level (G)": round(random.uniform(0.1, 1.5), 2),
        "Mileage (KM)": random.randint(20000, 200000)
    }


# Load or Train ML Model for Predictive Maintenance
MODEL_FILE = "predictive_maintenance_model.pkl"
sensor_columns = ["Engine Temperature (°C)", "Battery Voltage (V)", "Oil Pressure (PSI)", "Tire Pressure (PSI)",
                  "Vibration Level (G)", "Mileage (KM)"]

try:
    model = joblib.load(MODEL_FILE)
except:
    data = pd.DataFrame([generate_sensor_data() for _ in range(500)])
    data["Failure"] = np.where(
        (data["Engine Temperature (°C)"] > 110) |
        (data["Battery Voltage (V)"] < 12) |
        (data["Oil Pressure (PSI)"] < 30) |
        (data["Vibration Level (G)"] > 1.2), 1, 0
    )
    X = data[sensor_columns]
    y = data["Failure"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_FILE)

# Mock API for Telemetry Data
FREE_TELEMETRY_API = "https://mocki.io/v1/84f6d8c3-2b2c-4c3f-8220-2c1c1b89a36f"


def fetch_mock_telemetry():
    try:
        response = requests.get(FREE_TELEMETRY_API)
        return response.json() if response.status_code == 200 else generate_sensor_data()
    except:
        return generate_sensor_data()


# Streamlit Sidebar for Navigation
st.sidebar.title("🚚 Fleet Management Dashboard")
menu = st.sidebar.radio("Select Feature", ["Live Vehicle Tracking", "Predictive Maintenance", "Vehicle Diagnostics"])

if menu == "Live Vehicle Tracking":
    st.title("📍 Live Vehicle Tracking")

    # Define route from Swindon to Chippenham
    route = [
        (51.5601, -1.7806),  # Swindon
        (51.5400, -1.7300),
        (51.5200, -1.6800),
        (51.5000, -1.6300),
        (51.4800, -1.5800),
        (51.4600, -1.5300),
        (51.4400, -1.4800),
        (51.4280, -1.4600),  # Chippenham
    ]

    # Initialize session state for vehicle position
    if "vehicle_index" not in st.session_state:
        st.session_state.vehicle_index = 0

    # Create a folium map
    m = folium.Map(location=route[st.session_state.vehicle_index], zoom_start=10)
    folium.PolyLine(route, color="blue", weight=2.5, opacity=1).add_to(m)

    # Add a marker for the current vehicle position
    folium.Marker(route[st.session_state.vehicle_index],
                  icon=folium.Icon(color="red", icon="car", prefix="fa")).add_to(m)

    # Display the map
    st_folium(m, width=700, height=500)

    # Move vehicle automatically
    if st.session_state.vehicle_index < len(route) - 1:
        st.session_state.vehicle_index += 1
        time.sleep(100.01)  # Delay to simulate real-time movement
        st.rerun()


elif menu == "Predictive Maintenance":
    st.title("🛠️ Predictive Maintenance Dashboard")

    # Fetch and display multiple rows
    live_data = [fetch_mock_telemetry() for _ in range(5)]
    df_live = pd.DataFrame(live_data)

    # Fix the warning: Ensure feature names match when predicting
    X_live = df_live[sensor_columns]  # Ensure DataFrame format
    failure_predictions = model.predict(X_live)
    failure_probs = model.predict_proba(X_live)[:, 1] * 100  # Convert to percentage

    df_live["Failure Probability (%)"] = failure_probs
    df_live["Failure Risk"] = ["⚠️ High" if pred == 1 else "✅ Low" for pred in failure_predictions]

    st.write(df_live)

elif menu == "Vehicle Diagnostics":
    st.title("🔧 Vehicle Diagnostics")
    st.write("Fetching live data...")

    time.sleep(2)

    # Fetch and display multiple rows
    live_data = [fetch_mock_telemetry() for _ in range(5)]
    df_live = pd.DataFrame(live_data)

    st.write(df_live)
