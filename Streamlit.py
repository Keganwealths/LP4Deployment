import streamlit as st
import pandas as pd
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

# Select only the relevant columns from all_feature_names
selected_feature_names = [col for col in all_feature_names if col in input_columns]

# Convert user input to a DataFrame with selected columns
user_input_df = pd.DataFrame([user_input])[selected_feature_names]

# Make predictions using the pre-trained model
if st.button('Predict'):
    try:
        # Ensure that user_input_df is in the correct format (2D array)
        user_input_array = user_input_df.values
        
        # Predict sales using the loaded model
        predicted_sales = model.predict(user_input_array)
        
        # Display the predicted sales
        st.subheader('Predicted Sales')
        st.write(f'The predicted sales value is: {predicted_sales[0]}')
    except Exception as e:
        st.error(f"Error during prediction: {e}")
