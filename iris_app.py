import numpy as np
import joblib as dump
import streamlit as st

model = dump.load('iris_model.pkl')

st.set_page_config(page_title="Iris Species Classifier", page_icon="🌸", layout="centered")

st.title("🌸 Iris Species Classifier")
st.write("Enter the flower measurements to predict the Iris species.")

sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.8)
sepal_width  = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.0)
petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 4.3)
petal_width  = st.slider('Petal Width (cm)', 0.1, 2.5, 1.3)

if st.button('Predict'):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    species = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica'][prediction]
    st.success(f'Predicted Species: **{species}**')
