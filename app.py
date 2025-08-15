import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import datetime

# -------------------------------
# SETTINGS
# -------------------------------
MODEL_PATH = "Diesel_RNN_Model.h5"  # change if your model filename is different
DATA_PATH = "diesel.csv"           # change if your dataset filename is different
SEQUENCE_LENGTH = 120

# -------------------------------
# LOAD MODEL & DATA
# -------------------------------
st.title("🚛 Diesel Price Prediction Using RNN")
st.write("Predict future diesel prices based on historical data using an RNN model.")

# Load trained model
model = load_model(MODEL_PATH)

# Load and preprocess dataset
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

# Extract price values
prices = df['Price'].values.reshape(-1, 1)

# Scale prices
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# -------------------------------
# STREAMLIT INPUTS
# -------------------------------
start_date = st.date_input("Select starting date for prediction", datetime.date.today())
days_to_predict = st.number_input("Number of days to predict", min_value=1, max_value=30, value=7)

# -------------------------------
# PREDICTION LOGIC
# -------------------------------
if st.button("Predict Diesel Prices"):
    try:
        # Prepare initial sequence
        last_sequence = scaled_prices[-SEQUENCE_LENGTH:]
        
        predictions_scaled = []
        sequence = last_sequence.copy()

        for _ in range(days_to_predict):
            pred_scaled = model.predict(sequence.reshape(1, SEQUENCE_LENGTH, 1), verbose=0)
            predictions_scaled.append(pred_scaled[0, 0])
            sequence = np.append(sequence[1:], pred_scaled).reshape(SEQUENCE_LENGTH, 1)

        # Inverse transform predictions
        predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()

        # Prepare results DataFrame
        date_range = pd.date_range(start=start_date, periods=days_to_predict)
        results_df = pd.DataFrame({
            "Date": date_range,
            "Predicted Diesel Price (₹)": predictions
        })

        # Display results
        st.subheader("📊 Prediction Results")
        st.dataframe(results_df.style.format({"Predicted Diesel Price (₹)": "{:.2f}"}))
        st.line_chart(results_df.set_index("Date"))

    except Exception as e:
        st.error(f"Error during prediction: {e}")
