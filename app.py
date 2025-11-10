import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Title
st.title("Lightweight ML Dashboard - Iris Species Prediction")

# Load Iris Dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Sidebar: User input
st.sidebar.header("Input Features")
def user_input_features():
    sepal_length = st.sidebar.slider("Sepal length (cm)", float(X['sepal length (cm)'].min()), float(X['sepal length (cm)'].max()), float(X['sepal length (cm)'].mean()))
    sepal_width = st.sidebar.slider("Sepal width (cm)", float(X['sepal width (cm)'].min()), float(X['sepal width (cm)'].max()), float(X['sepal width (cm)'].mean()))
    petal_length = st.sidebar.slider("Petal length (cm)", float(X['petal length (cm)'].min()), float(X['petal length (cm)'].max()), float(X['petal length (cm)'].mean()))
    petal_width = st.sidebar.slider("Petal width (cm)", float(X['petal width (cm)'].min()), float(X['petal width (cm)'].max()), float(X['petal width (cm)'].mean()))
    
    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Map prediction to target names
target_names = iris.target_names
predicted_species = target_names[prediction[0]]

# Show results
st.subheader("Predicted Species")
st.write(predicted_species)

st.subheader("Prediction Probabilities")
st.write(pd.DataFrame(prediction_proba, columns=target_names))

# Dataset visualization
st.subheader("Iris Dataset Overview")
st.write(X.head())

st.subheader("Pairplot of Iris Dataset")
sns.pairplot(pd.concat([X, pd.Series(y, name='species')], axis=1), hue='species')
st.pyplot(plt.gcf())

