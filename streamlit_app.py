import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.preprocessing import StandardScaler
import os

# Suppress scikit-learn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Load the trained model and scaler
@st.cache_resource
def load_model():
    try:
        # Try to load from app directory first, then from root
        model_paths = ['app/wine_model.pkl', 'wine_model.pkl']
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                with open(model_path, 'rb') as file:
                    model, scaler = pickle.load(file)
                return model, scaler
        
        # If no model found, show error
        st.error("Model file 'wine_model.pkl' not found. Please ensure the model file is available.")
        st.stop()
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Set page title
st.set_page_config(page_title="Wine Quality Prediction", page_icon="🍷")

# Page header
st.title("🍷 Premium Wine Quality Predictor")
st.write("#### A tool for Mr. Sanborn's quality assurance team")
st.write("Enter the chemical properties of your wine sample to predict if it meets premium quality standards (7+ rating).")

# Add some styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #8B0000;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Define input fields
st.subheader("🧪 Wine Chemical Properties")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Acidity & pH**")
    fixed_acidity = st.number_input("Fixed Acidity (g/L)", min_value=0.0, max_value=20.0, value=7.5, step=0.1, help="Non-volatile acids that don't evaporate")
    volatile_acidity = st.number_input("Volatile Acidity (g/L)", min_value=0.0, max_value=2.0, value=0.5, step=0.01, help="Amount of acetic acid in wine")
    citric_acid = st.number_input("Citric Acid (g/L)", min_value=0.0, max_value=1.0, value=0.2, step=0.01, help="Adds freshness and flavor")
    pH = st.number_input("pH", min_value=2.0, max_value=5.0, value=3.4, step=0.01, help="Acidity/alkalinity level")
    
    st.markdown("**Sugar & Density**")
    residual_sugar = st.number_input("Residual Sugar (g/L)", min_value=0.0, max_value=15.0, value=2.0, step=0.1, help="Sugar remaining after fermentation")
    density = st.number_input("Density (g/mL)", min_value=0.9, max_value=1.1, value=0.997, step=0.0001, help="Density relative to water")
    
with col2:
    st.markdown("**Sulfur Compounds**")
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide (mg/L)", min_value=0.0, max_value=100.0, value=15.0, step=1.0, help="Prevents microbial growth")
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide (mg/L)", min_value=0.0, max_value=300.0, value=40.0, step=1.0, help="Total amount of SO2")
    sulphates = st.number_input("Sulphates (g/L)", min_value=0.0, max_value=2.0, value=0.6, step=0.01, help="Wine additive contributing to SO2")
    
    st.markdown("**Other Properties**")
    chlorides = st.number_input("Chlorides (g/L)", min_value=0.0, max_value=1.0, value=0.08, step=0.001, help="Amount of salt in wine")
    alcohol = st.number_input("Alcohol (%)", min_value=8.0, max_value=15.0, value=10.0, step=0.1, help="Alcohol percentage")

# Predict button
if st.button("🔮 Predict Wine Quality", type="primary"):
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
    st.subheader("🎯 Prediction Result")
    
    # Create a container for the result
    result_container = st.container()
    
    with result_container:
        if prediction == 1:
            st.success("✅ **GOOD QUALITY**: This wine meets premium quality standards!")
            confidence = prediction_proba[1] * 100
            st.balloons()
        else:
            st.error("❌ **NOT GOOD QUALITY**: This wine does not meet premium quality standards.")
            confidence = prediction_proba[0] * 100
        
        # Display confidence with better formatting
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.metric("Confidence Score", f"{confidence:.1f}%")
            st.progress(confidence/100)
    
    # Feature importance section
    st.subheader("📊 Quality Factors")
    
    # Feature importance (works only for tree-based models)
    if hasattr(model, 'feature_importances_'):
        feature_names = ['Fixed Acidity', 'Volatile Acidity', 'Citric Acid', 'Residual Sugar', 
                         'Chlorides', 'Free Sulfur Dioxide', 'Total Sulfur Dioxide', 'Density', 'pH', 'Sulphates', 'Alcohol']
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        st.write("**Top factors influencing wine quality:**")
        
        # Display top 5 features in a nicer format
        top_features = feature_importance.head(5)
        for idx, row in top_features.iterrows():
            st.write(f"• **{row['Feature']}**: {row['Importance']:.3f}")
        
        # Bar chart
        st.bar_chart(feature_importance.set_index('Feature'))
        
        # Suggestions based on prediction
        if prediction == 0:
            st.subheader("💡 Quality Improvement Suggestions")
            top_feature = feature_importance.iloc[0]['Feature']
            second_feature = feature_importance.iloc[1]['Feature']
            
            st.info(f"""
            **To improve wine quality, consider adjusting these key factors:**
            
            🎯 **{top_feature}** - Most important factor affecting quality
            
            🎯 **{second_feature}** - Second most important factor
            
            Focus on optimizing these chemical properties to enhance wine quality.
            """)
    else:
        st.info("Feature importance is not available for this model type.")

# Add explanation section
with st.expander("ℹ️ About this application"):
    st.write("""
    ### 🍷 Wine Quality Prediction System
    
    This application uses advanced machine learning algorithms to predict wine quality based on chemical properties.
    
    **🎯 Quality Definition:**
    - **Good Quality**: Wines with a rating of 7 or higher (out of 10)
    - **Not Good Quality**: Wines with a rating below 7
    
    **🤖 Model Information:**
    - **Algorithm**: Random Forest Classifier
    - **Training Data**: Red wine dataset with 1,599 samples
    - **Features**: 11 chemical properties
    - **Accuracy**: ~85% on test data
    
    **👨‍🔬 Developed for:**
    Mr. Sanborn's boutique winery quality assurance team to streamline the wine evaluation process.
    
    **📊 Dataset Source:**
    UCI Machine Learning Repository - Wine Quality Dataset
    """)

# Footer
st.markdown("---")
st.markdown("*Made with ❤️ for wine quality assurance*")
