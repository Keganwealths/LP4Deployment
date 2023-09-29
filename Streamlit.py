import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the preprocessed data and model
with open('preprocessed_data.pkl', 'rb') as f:
    preprocessed_data, all_feature_names = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the columns you want to include in the input (excluding 'sales')
input_columns = [
    'store_nbr', 'family', 'onpromotion',
    'price', 'city', 'state', 'type_x', 'cluster', 'type_y', 'locale', 'transferred'
]

# Streamlit app
st.title('Ecuador Time Series Sales Prediction')

# Input fields for the selected columns
user_input = {}

for col in input_columns:
    if col in ['store_nbr', 'onpromotion', 'cluster']:
        user_input[col] = st.number_input(f'Enter {col}', value=0)
    elif col in ['price']:
        user_input[col] = st.number_input(f'Enter {col}', value=0.0)
    else:
        user_input[col] = st.text_input(f'Enter {col}')

# Convert user input to a DataFrame
user_input_df = pd.DataFrame([user_input])

# Load the preprocessor object (you should replace 'preprocessor.pkl' with the actual filename)
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Function to preprocess user input data
def preprocess_user_input(user_input_df, preprocessor):
    # Apply the same preprocessing transformations as the original data
    user_preprocessed_data = preprocessor.transform(user_input_df)
    return user_preprocessed_data

# Preprocess the user input data using the same preprocessing as the original data
user_preprocessed_data = preprocess_user_input(user_input_df, preprocessor)

# Make predictions using the pre-trained model
if st.button('Predict'):
    try:
        # Predict sales using the loaded model
        predicted_sales = model.predict(user_preprocessed_data)
        
        # Display the predicted sales
        st.subheader('Predicted Sales')
        st.write(f'The predicted sales value is: {predicted_sales[0]}')
    except Exception as e:
        st.error(f"Error during prediction: {e}")
