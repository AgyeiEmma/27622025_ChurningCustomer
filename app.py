import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow import keras

# Load the scaler
with open('scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

# Load the Keras model
model_file = 'best_model.h5'
model = keras.models.load_model(model_file)

# User input
tenure = st.number_input("Tenure")
OnlineSecurity = st.radio("OnlineSecurity", ('No', 'Yes', 'No internet service'))
OnlineBackup = st.radio("OnlineBackup", ('No', 'Yes', 'No internet service'))
TechSupport = st.radio("TechSupport", ('No', 'Yes', 'No internet service'))
Contract = st.radio("Contract", ('Month-to-month', 'Two year', 'One year'))
PaperlessBilling = st.radio("PaperlessBilling", ('Yes', 'No'))
TotalCharges = st.number_input("TotalCharges")

# Convert categorical variables to numerical values
# Use one-hot encoding for 'Contract'
contract_mapping = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
contract_values = [contract_mapping[Contract]]

# Gather missing feature values
missing_features = {
    'OnlineSecurity': [1 if OnlineSecurity == 'Yes' else 0],
    'OnlineBackup': [1 if OnlineBackup == 'Yes' else 0],
    'TechSupport': [1 if TechSupport == 'Yes' else 0],
    'PaperlessBilling': [1 if PaperlessBilling == 'Yes' else 0]
}

# Combine with existing prediction data
user_inputs = pd.DataFrame({
    'tenure': [tenure],
    'TotalCharges': [TotalCharges]
})

# Check for data inconsistencies
if not isinstance(user_inputs, pd.DataFrame) or user_inputs.dtypes.eq(np.dtype('object')).any():
    st.error("Invalid data type. Expected numeric values.")
else:
    # Add 'TotalCharges' and 'tenure' back to prediction_data
    prediction_data = pd.concat([user_inputs[['tenure', 'TotalCharges']], pd.DataFrame(missing_features)], axis=1)

    # One-hot encode 'Contract'
    contract_dummies = pd.get_dummies(contract_values)

    # Add one-hot encoded columns to prediction_data
    prediction_data = pd.concat([prediction_data, contract_dummies], axis=1)

    # Check for missing values in categorical features
    if prediction_data.isnull().any().any():
        st.error("Missing values detected in categorical features. Please ensure all categorical features have valid values.")
    else:
        # Scale missing features using the loaded scaler
        scaled_missing_features = loaded_scaler.transform(prediction_data[['OnlineSecurity', 'OnlineBackup', 'TechSupport', 'PaperlessBilling']])

        # Update prediction data with scaled missing features
        prediction_data.update(pd.DataFrame(scaled_missing_features, columns=['OnlineSecurity', 'OnlineBackup', 'TechSupport', 'PaperlessBilling']))

        # Get the feature names used during training
        feature_names = ['tenure', 'TotalCharges'] + list(contract_dummies.columns) + ['OnlineSecurity', 'OnlineBackup', 'TechSupport', 'PaperlessBilling']

        # Make sure the feature names match
        if set(prediction_data.columns) != set(feature_names):
            st.error(f"Invalid feature names. Expected: {feature_names}. Got: {list(prediction_data.columns)}")
        else:
            # Make predictions
            y_pred = model.predict(prediction_data)

            if st.button('SUBMIT'):
                st.write("Predicted Churn Probability:", y_pred[0][0])
