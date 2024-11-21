import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    # Load data
    data = pd.read_csv("data/normalized_dataset.csv")

    # Define predictor columns and target column
    X = data[['Zip Code Score', 'Type of Property Score', 'Number of Rooms Score', 'Livable Space (m2) Score', 'Surface of the Land (m2) Score']]
    y = data['Price']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    return data, X_train, X_test, y_train, y_test

# Train Linear Regression model
@st.cache_resource
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Find nearest Zip Code within ±5
def find_nearest_zip_code(zip_code, zip_code_scores):
    valid_zip_codes = zip_code_scores.index
    nearby_zip_codes = valid_zip_codes[(valid_zip_codes >= zip_code - 5) & (valid_zip_codes <= zip_code + 5)]
    if not nearby_zip_codes.empty:
        return nearby_zip_codes[0]  # Return the first valid Zip Code
    else:
        return None  # No matching Zip Code found

# Load and preprocess data
data, X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Train the Linear Regression model
lr_model = train_model(X_train, y_train)

# Evaluate the model
r2_score = lr_model.score(X_test, y_test)

# Streamlit app interface
st.title("Real Estate Price Predictor")

# Display R^2 score
st.write(f"### Model Regression score on test data: {r2_score:.4f}")

# User inputs
zip_code = st.number_input("Enter Zip Code", min_value=1000, max_value=9999, step=1)
prop_type = st.selectbox("Select Type of Property", options=data['Type of Property'].unique())
livable_space = st.number_input("Enter Livable Space (m2)", min_value=10.0, step=1.0)
land_area = st.number_input("Enter Surface of the Land (m2)", min_value=0.0, step=1.0)
rooms_number = st.number_input("Enter Number of Rooms", min_value=0.0, step=1.0)

if st.button("Predict Price"):
    # Calculate scores for inputs
    zip_code_scores = data.groupby('Zip Code')['Price Ratio'].median()
    nearest_zip_code = find_nearest_zip_code(zip_code, zip_code_scores)
    if nearest_zip_code is not None:
        zip_score = zip_code_scores[nearest_zip_code]
        st.write(f"Using nearest Zip Code: {nearest_zip_code}")
    else:
        zip_score = zip_code_scores.median()
        st.write("No nearby Zip Code found. Using median Zip Code score.")

    prop_type_scores = data.groupby('Type of Property')['Price Ratio'].mean()
    building_scores = data.groupby('State of the Building')['Price Ratio'].mean()
    livable_space_scores = data.groupby('Livable Space (m2)')['Price Ratio'].median()
    land_scores = data.groupby('Surface of the Land (m2)')['Price Ratio'].median()
    room_scores = data.groupby('Number of Rooms')['Price Ratio'].mean()

    # Normalize scores
    zip_score /= zip_code_scores.max()
    prop_type_score = prop_type_scores.get(prop_type, 0) / prop_type_scores.max()
    livable_space_score = livable_space_scores.get(livable_space, 0) / livable_space_scores.max()
    land_score = land_scores.get(land_area, 0) / land_scores.max()
    rooms_score = room_scores.get(rooms_number, 0) / room_scores.max()

    # Prepare input features
    input_features = np.array([[zip_score, prop_type_score, rooms_score, livable_space_score, land_score]])

    # Predict using the trained model
    predicted_price = lr_model.predict(input_features)[0]

    # Display the prediction
    st.write(f"### Predicted Price: €{predicted_price:,.2f}")