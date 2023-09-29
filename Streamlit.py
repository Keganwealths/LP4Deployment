import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the preprocessed data and model
with open('preprocessed_data.pkl', 'rb') as f:
    preprocessed_data, all_feature_names = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit app
st.title('Ecuador Time Series Prediction')

# Input fields for the columns
user_input = {}

for col in preprocessed_data.columns:
    if col in ['store_nbr', 'onpromotion', 'cluster']:
        user_input[col] = st.number_input(f'Enter {col}', value=0)
    elif col in ['sales', 'price']:
        user_input[col] = st.number_input(f'Enter {col}', value=0.0)
    else:
        user_input[col] = st.text_input(f'Enter {col}')

# Convert user input to a DataFrame
user_input_df = pd.DataFrame([user_input])

# Make predictions using the pre-trained model
if st.button('Predict'):
    try:
        # Ensure the column order matches the model's expectations
        user_input_df = user_input_df[preprocessed_data.columns]
        
        # Predict using the loaded model
        prediction = model.predict(user_input_df)
        
        # Display the prediction
        st.subheader('Prediction')
        st.write(f'The predicted value is: {prediction[0]}')
    except Exception as e:
        st.error(f"Error during prediction: {e}")
