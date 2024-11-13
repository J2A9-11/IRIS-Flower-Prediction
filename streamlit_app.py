# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load your trained KNN model (replace 'knn_model.pkl' with the correct file path)
model = joblib.load('IRIS_model.pkl')

# Load your dataset to map predicted labels to species names (assumes it's a CSV file with species column)
df = pd.read_csv('IRIS.csv')  # Replace 'iris_dataset.csv' with the correct filename

# Ensure unique species labels
species_mapping = df['species'].unique()

st.markdown(
    """
    <style>
    .stApp {
        background-color: #000;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
st.title("Iris Flower Species Prediction")
st.write("Welcome to the **IRIS Flower Species Prediction App**!\n\nThis app predicts the **species** of Iris flowers based on input features.\n\nFor selecting the parameters simply adjust the sliders.")

st.header("Characteristics of Iris Species")
st.write("**Iris-setosa**: Usually has shorter petals and sepals compared to other species.\n\n"
         "**Iris-versicolor**: Intermediate size petals and sepals, with purple or blue coloration.\n\n"
         "**Iris-virginica**: Larger petals and sepals, generally deep purple or blue in color.\n\n")


# User inputs
st.subheader('Input Features')
sepal_length = st.slider('Sepal Length (cm)', min_value=4.0, max_value=8.0, value=5.0)
sepal_width = st.slider('Sepal Width (cm)', min_value=2.0, max_value=5.0, value=3.0)
petal_length = st.slider('Petal Length (cm)', min_value=1.0, max_value=7.0, value=1.5)
petal_width = st.slider('Petal Width (cm)', min_value=0.1, max_value=3.0, value=0.5)

# Predict button
if st.button('Predict'):
    # Convert inputs to numpy array for prediction
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Prediction
    prediction = model.predict(input_data)

    # Ensure prediction is an integer index (e.g., array([0]) -> 0)
    predicted_index = int(prediction[0])

    # Map predicted numeric label to species name
    predicted_species = species_mapping[predicted_index]

    # Display results
    st.write(f"The predicted species is: **{predicted_species}**")

    # Input summary
    st.subheader('Input Summary')
    st.write(f"**Sepal Length**: {sepal_length} cm\n\n"
         f"**Sepal Width**: {sepal_width} cm\n\n"
         f"**Petal Length**: {petal_length} cm\n\n"
         f"**Petal Width**: {petal_width} cm\n\n")
