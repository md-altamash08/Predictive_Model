import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load the model and data
data = load_iris()
scaler = StandardScaler().fit(data.data)
model = LogisticRegression().fit(scaler.transform(data.data), data.target)

st.title("Iris Predictive Model")
st.write("Enter the features of the Iris flower to predict its class.")

# User Input
sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, step=0.1)

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)
    st.write(f"Predicted Class: {data.target_names[prediction[0]]}")
