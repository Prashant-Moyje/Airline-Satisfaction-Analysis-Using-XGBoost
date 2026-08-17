import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load the model artifacts efficiently
@st.cache_resource
def load_artifacts():
    return joblib.load('airline_model_artifacts.pkl')

artifacts = load_artifacts()
encoders = artifacts['encoders']
scaler = artifacts['scaler']
model = artifacts['model']
feature_names = artifacts['feature_names']

# 2. Build the User Interface
st.title("✈️ Airline Passenger Satisfaction Predictor")
st.markdown("Enter the flight details below to predict if the passenger will be satisfied.")

# Create two columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics & Flight Details")
    gender = st.selectbox("Gender", options=list(encoders['Gender'].keys()))
    age = st.slider("Age", 18, 90, 35)
    customer_type = st.selectbox("Customer Type", options=list(encoders['Customer Type'].keys()))
    travel_type = st.selectbox("Type of Travel", options=list(encoders['Type of Travel'].keys()))
    flight_class = st.selectbox("Class", options=list(encoders['Class'].keys()))
    distance = st.number_input("Flight Distance", min_value=0, value=800)
    dep_delay = st.number_input("Departure Delay (minutes)", min_value=0, value=0)
    arr_delay = st.number_input("Arrival Delay (minutes)", min_value=0, value=0)

with col2:
    st.subheader("Service Ratings (0-5)")
    # The highest correlating features based on your analysis
    online_boarding = st.slider("Online Boarding", 0, 5, 3)
    entertainment = st.slider("In-flight Entertainment", 0, 5, 3)
    seat_comfort = st.slider("Seat Comfort", 0, 5, 3)
    on_board_service = st.slider("On-board Service", 0, 5, 3)
    leg_room = st.slider("Leg Room Service", 0, 5, 3)
    
    # Remaining ratings
    cleanliness = st.slider("Cleanliness", 0, 5, 3)
    wifi = st.slider("In-flight Wifi Service", 0, 5, 3)
    baggage = st.slider("Baggage Handling", 0, 5, 3)
    inflight_service = st.slider("In-flight Service", 0, 5, 3)
    checkin = st.slider("Check-in Service", 0, 5, 3)
    food = st.slider("Food and Drink", 0, 5, 3)
    time_conv = st.slider("Departure/Arrival Time Convenience", 0, 5, 3)
    ease_booking = st.slider("Ease of Online Booking", 0, 5, 3)
    gate_location = st.slider("Gate Location", 0, 5, 3)

# 3. Prediction Logic
if st.button("Predict Satisfaction"):
    # Create a dictionary of inputs
    input_data = {
        'Gender': gender,
        'Age': age,
        'Customer Type': customer_type,
        'Type of Travel': travel_type,
        'Class': flight_class,
        'Flight Distance': distance,
        'Departure Delay': dep_delay,
        'Arrival Delay': arr_delay,
        'Departure and Arrival Time Convenience': time_conv,
        'Ease of Online Booking': ease_booking,
        'Check-in Service': checkin,
        'Online Boarding': online_boarding,
        'Gate Location': gate_location,
        'On-board Service': on_board_service,
        'Seat Comfort': seat_comfort,
        'Leg Room Service': leg_room,
        'Cleanliness': cleanliness,
        'Food and Drink': food,
        'In-flight Service': inflight_service,
        'In-flight Wifi Service': wifi,
        'In-flight Entertainment': entertainment,
        'Baggage Handling': baggage
    }
    
    # 1. Safely map categorical strings to integers
    for col in encoders:
        input_data[col] = encoders[col][input_data[col]]
        
    # 2. Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # 3. Force exact column order and cast to float
    input_df = input_df[feature_names].astype(float)
    
    # 4. STRIP Pandas metadata to prevent silent XGBoost column shuffling
    input_array = input_df.values
    
    # Predict directly on the raw numpy array
    prediction = model.predict(input_array)[0]
    probabilities = model.predict_proba(input_array)[0]
    
    satisfied_prob = probabilities[1] * 100
    dissatisfied_prob = probabilities[0] * 100
    
    # Display Result
    st.markdown("---")
    if prediction == 1:
        st.success(f"### ✅ The passenger is likely to be **Satisfied** ({satisfied_prob:.1f}% confidence)")
    else:
        st.error(f"### ❌ The passenger is likely to be **Neutral or Dissatisfied** ({dissatisfied_prob:.1f}% confidence)")