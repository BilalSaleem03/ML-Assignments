

import sys
import json
import numpy as np
import pandas as pd
from keras.models import load_model
import joblib  # For loading .pkl files
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
bank_leaving_dir = os.path.join(script_dir, '..', '..', 'trained-model', 'insurance-charges-prediction')

def load_preprocessing_objects():
    """Load all preprocessing objects from the bank-leaving-prediction folder"""
    try:
        # Load the scaler
        scaler = joblib.load(os.path.join(bank_leaving_dir, 'scaler.pkl'))
        print("Scaler loaded successfully")
    except FileNotFoundError:
        print("Warning: Scaler file not found")
        scaler = None
    
    try:
        # Load the label encoder
        le = joblib.load(os.path.join(bank_leaving_dir, 'label_encoder.pkl'))
        print("Label encoder loaded successfully")
    except FileNotFoundError:
        print("Warning: Label encoder file not found")
        le = None
    
    try:
        # Load column order
        column_order = joblib.load(os.path.join(bank_leaving_dir, 'column_order.pkl'))
        print("Column order loaded successfully")
    except FileNotFoundError:
        print("Warning: Column order file not found")
        column_order = None
    
    return scaler, le, column_order

model = load_model(os.path.join(bank_leaving_dir, 'ANN_model_for_calculating_insurance_charges.h5'))
scaler, le, column_order = load_preprocessing_objects()

def preprocess_input(input_dict, scaler, le, column_order):
    # Convert input dict to DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # Apply the same preprocessing steps as to the training data

    # Standard Scaling
    numerical_cols = ['age', 'bmi', 'children']
    # Use the scaler fitted on the training data
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Label Encoding for sex and smoker
    input_df['sex'] = le.transform(input_df['sex'])
    input_df['smoker'] = le.transform(input_df['smoker'])

    # One-Hot Encoding for region
    # We need to manually create columns for all regions and set the appropriate one to 1
    # Ensure all region columns are present, even if the new data doesn't contain all regions
    region_cols = ['region_northeast', 'region_northwest', 'region_southeast', 'region_southwest']
    for col in region_cols:
        input_df[col] = 0

    # Set the value to 1 for the specified region
    input_df[f'region_{input_df["region"].iloc[0]}'] = 1

    # Drop the original 'region' column
    input_df = input_df.drop(columns=['region'])

    # Reorder columns to match the training data if necessary
    x_cols = ['age', 'bmi', 'children', 'smoker', 'sex', 'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest']
    input_df = input_df[x_cols]

    return input_df

def predict(input_data):
    # Parse input dict from JSON (if it's a string)
    if isinstance(input_data, str):
        input_dict = json.loads(input_data)
    else:
        input_dict = input_data
    
    # Preprocess input
    input_array = preprocess_input(input_dict, scaler, le, column_order)
    
    # Make prediction
    prediction = model.predict(input_array)
    
    # Convert to list and return
    return prediction.tolist()

if __name__ == '__main__':
    # For command line usage: python script.py '{"age": 53, "sex": "female", "bmi": 35.9, "children": 2, "smoker": "no", "region": "southwest"}'
    if len(sys.argv) > 1:
        input_data = sys.argv[1]
    else:
        # Default test data
        input_data = {
            'age': 53,
            'sex': 'female',
            'bmi': 35.9,
            'children': 2,
            'smoker': 'no',
            'region': 'southwest'
        }
    
    result = predict(input_data)
    print(json.dumps(result))