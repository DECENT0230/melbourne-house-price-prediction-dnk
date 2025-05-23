import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model and scaler
model = joblib.load('final_predictor/best_model.joblib')
scaler = joblib.load('final_predictor/scaler.joblib')

# Streamlit UI
st.title("🏡 Melbourne House Price Prediction")
st.markdown("""
Enter the property details below to estimate its price.  
**Note**: Bedroom Discrepancy = Rooms - Bedrooms.
""")

# Input Section
with st.expander("📋 Enter Property Details"):
    rooms = st.number_input('Number of Rooms', min_value=1, max_value=8, value=3, step=1)
    bedroom2 = st.number_input('Number of Bedrooms', min_value=0, max_value=8, value=3, step=1)
    bathroom = st.number_input('Number of Bathrooms', min_value=0, max_value=5, value=1, step=1)
    car = st.number_input('Number of Car Spaces', min_value=0, max_value=5, value=1, step=1)
    distance = st.number_input('Distance from CBD (km)', min_value=0.0, max_value=50.0, value=10.0, step=0.1)

# Calculate Bedroom Discrepancy
bedroom_discrepancy = rooms - bedroom2
st.write(f'**Bedroom Discrepancy**: {bedroom_discrepancy} (Rooms - Bedrooms)')

# Predict
if st.button('Predict Price'):
    # Validation rules
    if bedroom2 > rooms:
        st.error("🚫 Bedrooms cannot exceed total rooms.")
    elif all(val == 0 for val in [bedroom2, bathroom, car]) and rooms <= 1:
        st.warning("⚠️ Please enter realistic property features. All zero or minimal values will produce invalid predictions.")
    elif bedroom_discrepancy < -8 or bedroom_discrepancy > 8:
        st.warning("⚠️ Bedroom Discrepancy out of valid range (-8 to 8). Adjust Rooms or Bedrooms.")
    else:
        try:
            # Prepare input
            input_data = pd.DataFrame([[rooms, distance, bedroom2, bathroom, car, bedroom_discrepancy]], 
                                      columns=['Rooms', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 'Bedroom_Discrepancy'])

            # Scale input
            input_scaled = scaler.transform(input_data)

            # Predict
            prediction = model.predict(input_scaled)[0]

            # Reasonableness check
            if prediction < 10000:
                st.error("❌ Predicted price is unrealistically low. Please check your inputs.")
            elif prediction > 10_000_000:
                st.warning(f"⚠️ Unusually high prediction: ${prediction:,.2f}. This may indicate out-of-distribution input values.")
            else:
                st.success(f"💰 Estimated House Price: ${prediction:,.2f}")
                st.caption("Note: Prediction MAE ≈ $226,510 based on model performance.")
        except Exception as e:
            st.error(f"Prediction failed. Error: {e}")

# Model Information
st.markdown("""
---  
### 📊 Model Info
- **Model**: XGBoost Regressor  
- **Performance**: R² = 0.573, MAE ≈ $226,510  
- **Key Features**: Distance to CBD, Rooms, Bedrooms, Bathrooms, Car spaces  
- **Note**: Accuracy improves with more features (e.g., land size, suburb).
""")
