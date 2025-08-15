import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import datetime

# Load model
model = load_model("diesel_rnn_model.h5")  # change file name if different

st.title("🚛 Diesel Price Prediction Using RNN")
st.write("Predict diesel prices based on trained RNN model.")

# Date input
selected_date = st.date_input("Select date for prediction", datetime.date.today())

dummy_input = np.random.rand(1, 120, 1)  # Replace with your preprocessing

if st.button("Predict Price"):
    prediction = model.predict(dummy_input)
    st.success(f"Predicted Diesel Price: ₹{prediction[0][0]:.2f}")
