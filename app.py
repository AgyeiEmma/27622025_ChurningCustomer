import streamlit as st
import pandas as pd
import pickle as pkl
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('churn_assign.h5')

# Load the scaler
with open('scalar.pkl', 'rb') as file:
    scaler = pkl.load(file)

# Streamlit UI
st.title("Churn Prediction App")

# Create input fields for user to enter feature values
st.sidebar.header("User Input Features")

# Add input fields based on your features
tenure = st.sidebar.slider("Tenure", min_value=0, max_value=100, value=50)
monthly_charges = st.sidebar.slider("Monthly Charges", min_value=0, max_value=200, value=100)
total_charges = st.sidebar.slider("Total Charges", min_value=0, max_value=8000, value=4000)
contract = st.sidebar.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
payment_method = st.sidebar.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
online_security = st.sidebar.selectbox("Online Security", ['No', 'Yes'])

# Create a button to make predictions
if st.sidebar.button("Predict"):
    # Create a DataFrame with user input
    user_input = pd.DataFrame({
        'TotalCharges': [total_charges],
        'MonthlyCharges': [monthly_charges],
        'tenure': [tenure],
        'Contract': [contract],
        'PaymentMethod': [payment_method],
        'OnlineSecurity': [online_security],
    })

    # Encode categorical features using LabelEncoder
    le = LabelEncoder()
    encoded_contract = le.fit_transform(user_input['Contract'])
    encoded_payment_method = le.fit_transform(user_input['PaymentMethod'])
    encoded_online_security = le.fit_transform(user_input['OnlineSecurity'])

    # Add encoded features to the user_input DataFrame
    user_input['Contract_Encoded'] = encoded_contract
    user_input['PaymentMethod_Encoded'] = encoded_payment_method
    user_input['OnlineSecurity_Encoded'] = encoded_online_security

    # Preprocess input with the scaler
    processed_data = scaler.fit_transform(user_input[['TotalCharges', 'MonthlyCharges', 'tenure', 'Contract_Encoded', 'PaymentMethod_Encoded', 'OnlineSecurity_Encoded']])

    # Make predictions
    prediction = model.predict(processed_data)

    # Display the prediction
    st.subheader("Prediction Probability")
    st.write(prediction[0, 0])

    # Display a message based on the prediction
    if prediction[0, 0] > 0.5:
        st.success("The customer is predicted to churn.")
    else:
        st.success("The customer is predicted not to churn.")
