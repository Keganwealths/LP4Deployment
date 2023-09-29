import streamlit as st
import pandas as pd
import pickle

# Load the preprocessed data and model
with open('preprocessed_data.pkl', 'rb') as f:
    preprocessed_data, all_feature_names = pickle.load(f)

# Load the preprocessing transformers
with open('preprocessing_transformers.pkl', 'rb') as f:
    preprocessing_transformers = pickle.load(f)

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

# Create a DataFrame with user input
user_input_df = pd.DataFrame([user_input])

# Remove the 'sales' column from user input (if it's present)
if 'sales' in user_input_df.columns:
    user_input_df = user_input_df.drop(columns=['sales'])

# Remove the 'sales' column from all_feature_names
if 'sales' in all_feature_names:
    all_feature_names.remove('sales')

# Apply the same preprocessing transformations to user input data
user_input_preprocessed = preprocessing_transformers.transform(user_input_df)

# Make predictions using the loaded model
if st.button('Predict'):
    try:
        # Predict sales using the loaded model
        predicted_sales = model.predict(user_input_preprocessed)

        # Display the predicted sales
        st.subheader('Predicted Sales')
        st.write(f'The predicted sales value is: {predicted_sales[0]}')
    except Exception as e:
        st.error(f"Error during prediction: {e}")
