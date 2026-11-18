import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model
model = joblib.load('predictor.pkl')

# Define mappings based on the dataset
airline_map = {
    'IndiGo': 3, 'Air India': 1, 'Jet Airways': 4, 'Multiple carriers': 6,
    'SpiceJet': 8, 'GoAir': 2, 'Vistara': 10, 'Air Asia': 0
}
source_map = {
    'Banglore': 0, 'Kolkata': 3, 'Delhi': 2, 'Chennai': 1, 'Mumbai': 4
}
destination_map = {
    'Delhi': 2, 'Banglore': 0, 'Cochin': 1, 'Kolkata': 4, 'Hyderabad': 3
}

def predict_price(airline, source, destination, date, month, year, hour, minute):
    arr = np.array([
        airline_map[airline],
        source_map[source],
        destination_map[destination],
        date,
        month,
        year,
        hour,
        minute
    ])
    arr = arr.reshape(1, -1)
    pred = model.predict(arr)
    return pred[0]

st.set_page_config(page_title="Flight Price Predictor", page_icon="✈️")
st.title("✈️ Flight Price Predictor")
st.write("Enter flight details to estimate the price.")

col1, col2 = st.columns(2)
with col1:
    airline = st.selectbox('Airline', list(airline_map.keys()))
    source = st.selectbox('Source', list(source_map.keys()))
    destination = st.selectbox('Destination', list(destination_map.keys()))
    date = st.number_input('Date', min_value=1, max_value=31, value=1)
    month = st.number_input('Month', min_value=1, max_value=12, value=1)

with col2:
    year = st.number_input('Year', min_value=2019, max_value=2020, value=2019)
    hour = st.number_input('Departure Hour', min_value=0, max_value=23, value=12)
    minute = st.number_input('Departure Minute', min_value=0, max_value=59, value=0)

if st.button("Predict Price", type="primary"):
    price = predict_price(airline, source, destination, date, month, year, hour, minute)
    st.success(f"Estimated Price: ₹{price:,.2f}")

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <p>Made for Flight Price Prediction</p>
</div>
""", unsafe_allow_html=True)

