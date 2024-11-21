import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load and preprocess data

def load_and_preprocess_data():
    # Load data
    data = pd.read_csv("data/normalized_dataset.csv")

    # Define predictor columns and target column
    X = data[['Zip Code Score', 'Type of Property Score', 'State of the Building Score', 'Livable Space (m2) Score', 'Surface of the Land (m2) Score']]
    y = data['Price']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    return data, X_train, X_test, y_train, y_test

# Train Linear Regression model

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

print(r2_score)
