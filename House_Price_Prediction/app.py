import streamlit as st
import numpy as np
import pickle

# Load trained model
with open("house_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🏠 California House Price Prediction")
st.write("Enter housing details below:")

# User Inputs
MedInc = st.number_input("Median Income", value=3.0)
HouseAge = st.number_input("House Age", value=20.0)
AveRooms = st.number_input("Average Rooms", value=5.0)
AveBedrms = st.number_input("Average Bedrooms", value=1.0)
Population = st.number_input("Population", value=1000.0)
AveOccup = st.number_input("Average Occupancy", value=3.0)
Latitude = st.number_input("Latitude", value=34.0)
Longitude = st.number_input("Longitude", value=-118.0)

# Feature Engineering (same as training)
RoomsPerHousehold = AveRooms / AveOccup
BedroomsPerRoom = AveBedrms / AveRooms
PopulationPerHousehold = Population / AveOccup

# Prepare input data
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

# Predict
if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated House Price: ${prediction * 100000:,.2f}")
