import streamlit as st
import pickle
import numpy as np

# App Title
st.title("🌸 Iris Flower Species Identifier")
st.write("Please enter the flower measurements to predict its species.")

# User Inputs with Min, Max, and Default Values
sepL = st.number_input("Sepal Length (cm)", min_value=4.3, max_value=7.9, value=5.8)
sepW = st.number_input("Sepal Width (cm)", min_value=2.0, max_value=4.4, value=3.0)
petL = st.number_input("Petal Length (cm)", min_value=1.0, max_value=6.9, value=4.3)
petW = st.number_input("Petal Width (cm)", min_value=0.1, max_value=2.5, value=1.3)

# Load the Trained Model
with open('iris.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict the Species
if st.button("🔍 Predict Species"):
    try:
        input_features = np.array([[sepL, sepW, petL, petW]])
        species_idx = model.predict(input_features)[0]

        # Mapping numerical output to species names
        species_map = {0: "Iris Setosa", 1: "Iris Versicolor", 2: "Iris Virginica"}
        species_name = species_map.get(species_idx, "Unknown")

        st.success(f"🌿 The predicted species is: **{species_name}**")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
