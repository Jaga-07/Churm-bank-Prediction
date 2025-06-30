import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load all models
model_files = {
    'Logistic Regression': 'logistic_regression_model.pkl',
    'Naive Bayes': 'naive_bayes_model.pkl',
    'Random Forest': 'random_forest_model.pkl',
    'K-Nearest Neighbors': '/content/k-nearest_neighbors_model.pkl',
    'Support Vector Machine': 'support_vector_machine_model.pkl'
}

models = {name: joblib.load(file) for name, file in model_files.items()}

# Load cleaned data to get feature names
data = pd.read_csv('cleaned_bank_data.csv')
feature_names = list(data.columns)
feature_names.remove('y')

# Streamlit App UI
st.title('Bank Term Deposit Prediction - Multi Model Comparison App')
st.write('Provide customer details below. Each model has a separate tab for predictions.')

# Dynamic input form
st.sidebar.header('Customer Details')
input_data = {}
for feature in feature_names:
    if data[feature].dtype == 'int64' or data[feature].dtype == 'float64':
        input_data[feature] = st.sidebar.number_input(f'Enter {feature}', value=float(data[feature].median()))
    else:
        input_data[feature] = st.sidebar.selectbox(f'Select {feature}', options=sorted(list(data[feature].unique())))

input_df = pd.DataFrame([input_data])

if st.button('Predict'):
    input_df = pd.DataFrame([input_data])
    prediction = selected_model.predict(input_df)[0]
# Create Tabs for Each Model
tabs = st.tabs(list(models.keys()))

for tab, (model_name, model) in zip(tabs, models.items()):
    with tab:
        st.subheader(f'{model_name} Prediction')
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

        if prediction == 1:
            st.success('The customer is likely to subscribe to the term deposit.')
        else:
            st.error('The customer is unlikely to subscribe to the term deposit.')

        if probability is not None:
            st.write(f'Prediction Probability (Yes): {probability:.2%}')

# Optional: Display input data
if st.sidebar.checkbox('Show Input Data'):
    st.write('Customer Input Data:', input_data)
