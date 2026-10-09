import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# Configure page style
st.set_page_config(page_title="Energy Consumption Predictor", layout="centered")

st.title("⚡ Energy Consumption Prediction App")
st.write("Deploying the trained SimpleRNN model to forecast energy consumption based on the last 30 days of data.")

# Load the saved model and scaler
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('rnn_energy_model.keras')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

try:
    model, scaler = load_assets()
    st.success("Model and Scaler loaded successfully!")
except Exception as e:
    st.error(f"Error loading assets: {e}")

# Input options on sidebar
st.sidebar.header("Input Settings")
input_mode = st.sidebar.selectbox("Choose Input Method", ["Manual Input", "Upload CSV"])

if input_mode == "Manual Input":
    st.subheader("Enter last 30 days of daily consumption (kWh)")
    
    # Setup a form for manual values
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        user_inputs = []
        for i in range(30):
            col = col1 if i < 15 else col2
            # Providing a reasonable baseline around 3000 kWh
            val = col.number_input(f"Day {i+1} (kWh)", value=3000.0, step=10.0)
            user_inputs.append(val)
            
        submitted = st.form_submit_button("Predict Next Sequences")
        
    if submitted:
        # Format inputs for scaler and model
        input_array = np.array(user_inputs).reshape(-1, 1)
        scaled_input = scaler.transform(input_array)
        
        # Reshape to 3D shape (1, 30, 1) expected by the SimpleRNN layer
        rnn_input = scaled_input.reshape(1, 30, 1)
        
        # Generate and inverse-transform predictions
        prediction_scaled = model.predict(rnn_input)
        prediction_2d = prediction_scaled.reshape(-1, 1)
        prediction_unscaled = scaler.inverse_transform(prediction_2d)
        
        st.subheader("🔮 Predicted Sequences (kWh):")
        results_df = pd.DataFrame({
            'Sequence Day': [f"Forecast Day {i+1}" for i in range(30)],
            'Predicted Value (kWh)': prediction_unscaled.flatten()
        })
        st.dataframe(results_df)
        st.metric("Average Forecasted Consumption", f"{prediction_unscaled.mean():.2f} kWh")

elif input_mode == "Upload CSV":
    st.subheader("Upload CSV containing historical data")
    st.write("Upload a CSV file containing at least 30 rows with a column named `Consumption (kWh)`.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        if 'Consumption (kWh)' in data.columns:
            if len(data) >= 30:
                last_30 = data['Consumption (kWh)'].values[-30:]
                st.write("Loaded Last 30 Days successfully.")
                
                input_array = last_30.reshape(-1, 1)
                scaled_input = scaler.transform(input_array)
                rnn_input = scaled_input.reshape(1, 30, 1)
                
                prediction_scaled = model.predict(rnn_input)
                prediction_2d = prediction_scaled.reshape(-1, 1)
                prediction_unscaled = scaler.inverse_transform(prediction_2d)
                
                st.subheader("🔮 Predicted Sequences (kWh):")
                results_df = pd.DataFrame({
                    'Sequence Day': [f"Forecast Day {i+1}" for i in range(30)],
                    'Predicted Value (kWh)': prediction_unscaled.flatten()
                })
                st.dataframe(results_df)
                st.metric("Average Forecasted Consumption", f"{prediction_unscaled.mean():.2f} kWh")
            else:
                st.error("The uploaded CSV must contain at least 30 records.")
        else:
            st.error("The CSV must contain a column named 'Consumption (kWh)'.")
