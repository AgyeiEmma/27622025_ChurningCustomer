import streamlit as st
import pandas as pd
import numpy as np
import pickle, joblib
from scikeras.wrappers import KerasClassifier

model_file = 'churnmodel.pkl'
file = open(model_file, 'rb')
model = pickle.load(file)
#model = joblib.load(open('./churnmodel.joblib','rb'))

tenure = int(st.number_input("tenure"))
OnlineSecurity= st.radio("OnlineSecurity", ('No','Yes','No internet service'))
OnlineBackup = st.radio("OnlineBackup", ('No','Yes','No internet service'))
TechSupport = st.radio("TechSupport", ('No','Yes','No internet service'))
Contract = st.radio("Contract", ('Month-to-month','Two year','One year'))
PaperlessBilling = st.radio("PaperlessBilling", ('Yes','No'))
TotalCharges = int(st.number_input("TotalCharges"))

userInputs = [tenure, OnlineSecurity, OnlineBackup, TechSupport, Contract, PaperlessBilling, TotalCharges]
userInputs = np.array(userInputs)

userInputs = pd.DataFrame(userInputs.reshape(1, -1), columns=['tenure','OnlineSecurity','OnlineBackup','TechSupport','Contract','PaperlessBilling','TotalCharges'])

y_pred = model.predict(userInputs)

if st.button('SUBMIT'):
    st.write(y_pred)