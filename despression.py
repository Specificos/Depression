
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model_path = 'New_Student_Depression_Model.pkl'
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error(f"The model file '{model_path}' was not found. Please ensure it exists in the directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

st.write("""
# Depression Prediction App
This app predicts the likelihood of depression based on user input features.
""")

st.sidebar.header('User Input Parameters')

# Function to capture user input
def user_input_features():
    
    academic_pressure = st.sidebar.slider('Academic Pressure (1-10)', 1, 10, 5)
    study_satisfaction = st.sidebar.slider('Study Satisfaction (1-10)', 1, 10, 5)
    suicidal_thoughts = st.sidebar.radio('Have you ever had suicidal thoughts?', ['Yes', 'No'])
    work_study_hours = st.sidebar.slider('Work/Study Hours per Day', 0, 16, 8)
    financial_stress = st.sidebar.slider('Financial Stress (1-10)', 1, 10, 5)
    dietary_habits = st.sidebar.selectbox('Dietary Habits', [
        'Healthy', 'Unhealthy'
    ])

    # Encoding categorical features
    binary_map = {'Yes': 1, 'No': 0}

    # One-hot encoding for Dietary Habits
    diet_map = {
        'Healthy': [1, 0],
        'Unhealthy': [0, 1]
    }

    # Combine all features
    data = {
        'Academic Pressure': academic_pressure,
        'Study Satisfaction': study_satisfaction,
        'Have you ever had suicidal thoughts ?': binary_map[suicidal_thoughts],
        'Work/Study Hours': work_study_hours,
        'Financial Stress': financial_stress,
        **{f'Dietary Habits_{key}': value for key, value in zip(diet_map.keys(), diet_map[dietary_habits])}
    }

   
    df = pd.DataFrame(data, index=[0])
    return df


df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

# predictions
try:
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)

    st.subheader('Prediction')
    st.write('Depression Status:', 'Yes' if prediction[0] == 1 else 'No')

    st.subheader('Prediction Probability')
    st.write(f"Probability of No Depression: {prediction_proba[0][0]:.2f}")
    st.write(f"Probability of Depression: {prediction_proba[0][1]:.2f}")
except Exception as e:
    st.error(f"An error occurred during prediction: {e}")
 