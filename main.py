import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import json
import os
from flask import Flask, request, jsonify, render_template

# Create sample data since JSON files don't exist yet
sample_data = [
    {"text": "invest stocks", "label": "investment"},
    {"text": "check balance", "label": "balance"},
    {"text": "transfer money", "label": "transfer"},
    # Add more sample data as needed
]

# Save sample data to JSON file
with open('train.json', 'w') as f:
    json.dump(sample_data, f)

# Load data from single JSON file
def load_data(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return pd.DataFrame(data)

# Load the data
df = load_data('train.json')

def preprocess_data(df):
    # Create features from text
    df['feature1'] = df['text'].apply(lambda x: len(x))
    df['feature2'] = df['text'].apply(lambda x: len(x.split()))
    
    # Convert label to numerical
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    
    return df[['feature1', 'feature2']], df['label'], le

# Preprocess the data
X, y, label_encoder = preprocess_data(df)

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model
import joblib
joblib.dump((model, scaler, label_encoder), 'trained_model.joblib')

def process_user_input(user_text):
    """Process user text input"""
    features = pd.DataFrame({
        'feature1': [len(user_text)],
        'feature2': [len(user_text.split())]
    })
    return features

def get_prediction(user_text):
    """Get prediction for user input"""
    # Load model and preprocessors
    model, scaler, label_encoder = joblib.load('trained_model.joblib')
    
    # Process input
    features = process_user_input(user_text)
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0]
    
    return {
        'prediction': label_encoder.inverse_transform([prediction])[0],
        'confidence': float(max(prob))
    }

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_text = request.json.get('text', '')
    if not user_text:
        return jsonify({'error': 'No text provided'}), 400
    
    result = get_prediction(user_text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
