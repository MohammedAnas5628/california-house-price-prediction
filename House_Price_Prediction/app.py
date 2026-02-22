import streamlit as st
import numpy as np
import pickle
import os

# Load model safely
model_path = os.path.join(os.path.dirname(__file__), "house_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

st.title("🏠 California House Price Prediction")
st.write("Enter only the essential details:")

# --- Clean Minimal Inputs ---
col1, col2 = st.columns(2)

with col1:
    MedInc = st.number_input("Median Income", min_value=0.0, format="%.2f")
    HouseAge = st.number_input("House Age", min_value=0.0, format="%.0f")
    AveRooms = st.number_input("Average Rooms", min_value=0.0, format="%.2f")

with col2:
    Latitude = st.number_input("Latitude", format="%.4f")
    Longitude = st.number_input("Longitude", format="%.4f")

# --- Hidden default values (not shown to user) ---
AveBedrms = 1.0
Population = 1000.0
AveOccup = 3.0

# --- Feature Engineering ---
RoomsPerHousehold = AveRooms / AveOccup if AveOccup != 0 else 0
BedroomsPerRoom = AveBedrms / AveRooms if AveRooms != 0 else 0
PopulationPerHousehold = Population / AveOccup if AveOccup != 0 else 0

# Prepare input
input_data = np.array([[
    MedInc,
    HouseAge,
    AveRooms,
    AveBedrms,
    Population,
    AveOccup,
    Latitude,
    Longitude,
    RoomsPerHousehold,
    BedroomsPerRoom,
    PopulationPerHousehold
]])

# --- Predict Button ---
if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated House Price: ${prediction * 100000:,.2f}")
