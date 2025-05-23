import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model and scaler
model = joblib.load('final_predictor/best_model.joblib')
scaler = joblib.load('scaler.joblib')

# Streamlit app
st.title("🏡 Melbourne House Price Prediction")
st.markdown("""
Enter the property details below to estimate its price.  
Values must match realistic ranges based on the Melbourne housing dataset.  
**Note**: Bedroom Discrepancy is automatically calculated as Rooms - Bedrooms.
""")

# Input fields with constraints matching preprocessing
rooms = st.number_input('Number of Rooms', min_value=1, max_value=8, value=3, step=1, help="Total rooms, capped at 8.")
distance = st.number_input('Distance from CBD (km)', min_value=0.0, max_value=50.0, value=10.0, step=0.1, help="Distance to city center, typically 0–48 km.")
bedroom2 = st.number_input('Number of Bedrooms', min_value=0, max_value=8, value=3, step=1, help="Bedroom count, capped at 8.")
bathroom = st.number_input('Number of Bathrooms', min_value=0, max_value=5, value=1, step=1, help="Bathrooms, capped at 5.")
car = st.number_input('Number of Car Spaces', min_value=0, max_value=5, value=1, step=1, help="Car spaces, capped at 5.")

# Auto-calculate Bedroom_Discrepancy
bedroom_discrepancy = rooms - bedroom2
st.write(f'**Bedroom Discrepancy**: {bedroom_discrepancy} (Rooms - Bedrooms)')

# Predict button and validation
if st.button('Predict Price'):
    # Validate inputs
    if bedroom_discrepancy < -8 or bedroom_discrepancy > 8:
        st.warning("Bedroom Discrepancy is unrealistic (should be between -8 and 8). Adjust Rooms or Bedrooms.")
    else:
        # Prepare input data
        input_data = pd.DataFrame([[rooms, distance, bedroom2, bathroom, car, bedroom_discrepancy]], 
                                  columns=['Rooms', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 'Bedroom_Discrepancy'])
        # Scale inputs (model expects standardized features)
        input_data_scaled = scaler.transform(input_data)
        # Predict
        prediction = model.predict(input_data_scaled)[0]
        if prediction < 0:
            st.error("Prediction is negative, which is invalid. Please check input values.")
        else:
            st.success(f"💰 Estimated House Price: ${prediction:,.2f}")
            st.write(f"**Note**: Prediction has an average error of ~$226,510 based on model performance (MAE).")

# Model information
st.markdown("""
### Model Details
- **Model**: XGBoost Regressor
- **Performance**: 
  - R² = 0.5726 (explains ~57.3% of price variance)
  - Mean Absolute Error (MAE) = $226,510.18
- **Key Predictors**: Distance from CBD, Number of Rooms
- **Limitations**: Accuracy may improve with additional features (e.g., land size, suburb).
""")
