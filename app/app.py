import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
@st.cache_resource
def load_model():
    with open('wine_model.pkl', 'rb') as file:
        model, scaler = pickle.load(file)
    return model, scaler

# Set page title
st.set_page_config(page_title="Wine Quality Prediction", page_icon="🍷")

# Page header
st.title("Premium Wine Quality Predictor")
st.write("#### A tool for Mr. Sanborn's quality assurance team")
st.write("Enter the chemical properties of your wine sample to predict if it meets premium quality standards (7+ rating).")

# Define input fields
st.subheader("Wine Chemical Properties")

col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, max_value=20.0, value=7.5, step=0.1)
    volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, max_value=2.0, value=0.5, step=0.01)
    citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    residual_sugar = st.number_input("Residual Sugar", min_value=0.0, max_value=15.0, value=2.0, step=0.1)
    chlorides = st.number_input("Chlorides", min_value=0.0, max_value=1.0, value=0.08, step=0.001)
    
with col2:
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, max_value=100.0, value=15.0, step=1.0)
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, max_value=300.0, value=40.0, step=1.0)
    density = st.number_input("Density", min_value=0.9, max_value=1.1, value=0.997, step=0.0001)
    pH = st.number_input("pH", min_value=2.0, max_value=5.0, value=3.4, step=0.01)
    sulphates = st.number_input("Sulphates", min_value=0.0, max_value=2.0, value=0.6, step=0.01)
    alcohol = st.number_input("Alcohol", min_value=8.0, max_value=15.0, value=10.0, step=0.1)

# Predict button
if st.button("Predict Quality"):
    # Load model
    model, scaler = load_model()
    
    # Create input array
    input_data = np.array([fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
                          chlorides, free_sulfur_dioxide, total_sulfur_dioxide, 
                          density, pH, sulphates, alcohol]).reshape(1, -1)
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]
    
    # Display result
    st.subheader("Prediction Result")
    
    # Create a container for the result
    result_container = st.container()
    
    with result_container:
        if prediction == 1:
            st.success("✅ GOOD QUALITY: This wine meets premium quality standards!")
            confidence = prediction_proba[1] * 100
        else:
            st.error("❌ NOT GOOD QUALITY: This wine does not meet premium quality standards.")
            confidence = prediction_proba[0] * 100
        
        st.write(f"Confidence Score: {confidence:.2f}%")
        
        # Display gauge chart for confidence
        st.write("Confidence Meter:")
        st.progress(confidence/100)
    
    # Feature importance section
    st.subheader("Quality Factors")
    
    # Feature importance (works only for tree-based models)
    if hasattr(model, 'feature_importances_'):
        feature_names = ['Fixed Acidity', 'Volatile Acidity', 'Citric Acid', 'Residual Sugar', 
                         'Chlorides', 'Free Sulfur Dioxide', 'Total Sulfur Dioxide', 'Density', 'pH', 'Sulphates', 'Alcohol']
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        st.subheader("Quality Factors")
        st.write("Top factors influencing wine quality:")
        st.bar_chart(feature_importance.set_index('Feature').head())
        # Suggestion based on prediction
        st.subheader("Quality Improvement Suggestions")
        if prediction == 0:
            top_feature = feature_importance.iloc[0]['Feature']
            second_feature = feature_importance.iloc[1]['Feature']
            st.write(f"To improve quality, consider adjusting these key factors:")
            st.write(f"- {top_feature}")
            st.write(f"- {second_feature}")
    else:
        st.info("Feature importance is not available for this model type.")

# Add explanation section
with st.expander("About this app"):
    st.write("""
    This application uses machine learning to predict wine quality based on chemical properties.
    
    **Quality definition:**
    - Good Quality: Wines with a rating of 7 or higher
    - Not Good Quality: Wines with a rating below 7
    
    The model was trained on a dataset of red wine samples with their chemical properties and quality ratings.
    """)