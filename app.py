
import streamlit as st
import joblib
import pandas as pd
import json
from pathlib import Path

st.set_page_config(page_title="Decision Tree Predictor")
st.title("🌳 Decision Tree Predictor")
st.write("Predict Iris flower species using Machine Learning.")

model = joblib.load("decision_tree.joblib")

st.subheader("Enter flower measurements")

sepal_length = st.number_input("Sepal length (cm)", 0.0, 10.0, 5.1)
sepal_width = st.number_input("Sepal width (cm)", 0.0, 10.0, 3.5)
petal_length = st.number_input("Petal length (cm)", 0.0, 10.0, 1.4)
petal_width = st.number_input("Petal width (cm)", 0.0, 10.0, 0.2)

if st.button("Predict"):
    features = pd.DataFrame([[
        sepal_length, sepal_width, petal_length, petal_width
    ]], columns=[
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ])

    prediction = int(model.predict(features)[0])
    names = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"Predicted flower: {names[prediction]}")

if Path("metrics.json").exists():
    with open("metrics.json") as f:
        metrics = json.load(f)
    st.subheader("Model accuracy")
    st.write(metrics)

if Path("loss_graph.png").exists():
    st.subheader("Training and Testing Error Graph")
    st.image("loss_graph.png")
