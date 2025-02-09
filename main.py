import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import json

# Load and preprocess data from JSON files
def load_data(file_paths):
    data = []
    for file_path in file_paths:
        with open(file_path) as f:
            data.extend(json.load(f))
    return pd.DataFrame(data)

# List of JSON file paths from archive folder
json_files = ['archive/dev.json', 'archive/test.json', 'archive/train.json']

# Load the data
df = load_data(json_files)

# Data preprocessing
# Note: Adjust these steps based on your actual data structure
def preprocess_data(df):
    # Handle missing values
    df = df.dropna()
    
    # Convert categorical variables to numerical
    categorical_columns = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_columns)
    
    return df

# Preprocess the data
processed_df = preprocess_data(df)

# Split features and target
# Assuming last column is target - adjust as needed
X = processed_df.iloc[:, :-1]
y = processed_df.iloc[:, -1]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Save the model
import joblib
joblib.dump(model, 'trained_model.joblib')
print("\nModel saved as 'trained_model.joblib'")

def extract_feature1(text):
    """Extract first feature from text"""
    # Add your feature extraction logic here
    return len(text)  # Example: using text length as a feature

def extract_feature2(text):
    """Extract second feature from text"""
    # Add your feature extraction logic here
    return len(text.split())  # Example: using word count as a feature

def process_user_input(user_text, scaler):
    """
    Process user text input and prepare it for model prediction
    """
    # Convert user input to DataFrame format
    # Note: Modify this according to your feature extraction logic
    user_data = {
        'feature1': [extract_feature1(user_text)],
        'feature2': [extract_feature2(user_text)],
        # ... add all required features
    }
    user_df = pd.DataFrame(user_data)
    
    # Apply the same preprocessing steps as training data
    user_df = preprocess_data(user_df)
    
    # Ensure columns match training data
    missing_cols = set(X.columns) - set(user_df.columns)
    for col in missing_cols:
        user_df[col] = 0
    
    # Ensure column order matches training data
    user_df = user_df[X.columns]
    
    # Scale the features
    user_df_scaled = scaler.transform(user_df)
    
    return user_df_scaled

def get_prediction(user_text):
    """
    Get prediction for user input
    """
    # Load the saved model
    loaded_model = joblib.load('trained_model.joblib')
    
    # Process user input
    processed_input = process_user_input(user_text, scaler)
    
    # Make prediction
    prediction = loaded_model.predict(processed_input)[0]
    
    # Get prediction probability
    prob = loaded_model.predict_proba(processed_input)[0]
    
    return {
        'prediction': prediction,
        'confidence': float(max(prob))
    }

# Example usage for web interface:
"""
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    user_text = request.json.get('text', '')
    if not user_text:
        return jsonify({'error': 'No text provided'}), 400
        
    result = get_prediction(user_text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
"""
