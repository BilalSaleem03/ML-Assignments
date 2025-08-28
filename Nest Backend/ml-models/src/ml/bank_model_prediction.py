# If you see "No module named 'sklearn'", install scikit-learn:
# pip install scikit-learn

try:
    import sklearn
except ImportError:
    print("ERROR: scikit-learn is not installed. Please run 'pip install scikit-learn' in your Python environment.")
    sys.exit(1)

# import sys
# import json
# import numpy as np
# from tensorflow.keras.models import load_model

# # Load model
# model = load_model('./trained-model/ANN_model_for_bank_leaving_prediction.h5')


import sys
import json
import numpy as np
import pandas as pd
from keras.models import load_model
import joblib  # For loading .pkl files
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
trained_model_dir = os.path.join(script_dir, 'trained-model')
# trained_model_dir = os.path.join(script_dir, '..', 'trained-model')

def load_preprocessing_objects():
    """Load all preprocessing objects from the trained-model folder"""
    try:
        # Load the scaler
        scaler = joblib.load(os.path.join(trained_model_dir, 'scaler.pkl'))
        print("Scaler loaded successfully")
    except FileNotFoundError:
        print("Warning: Scaler file not found")
        scaler = None
    
    try:
        # Load the label encoder
        le = joblib.load(os.path.join(trained_model_dir, 'label_encoder.pkl'))
        print("Label encoder loaded successfully")
    except FileNotFoundError:
        print("Warning: Label encoder file not found")
        le = None
    
    try:
        # Load column order
        column_order = joblib.load(os.path.join(trained_model_dir, 'column_order.pkl'))
        print("Column order loaded successfully")
    except FileNotFoundError:
        print("Warning: Column order file not found")
        column_order = None
    
    return scaler, le, column_order

# Load model and preprocessing objects
model = load_model(os.path.join(trained_model_dir, 'ANN_model_for_Bank_leaving_prediction.h5'))
scaler, le, column_order = load_preprocessing_objects()

def preprocess_input(input_dict, scaler, le, column_order):
    """Preprocess input using the saved preprocessing objects"""
    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])
    print("222")
    # Dynamically get scaler columns
    if scaler is not None and hasattr(scaler, 'feature_names_in_'):
        scaler_cols = list(scaler.feature_names_in_)
    else:
        # fallback to previous default
        scaler_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

    # Scale only the columns the scaler was trained on
    if scaler is not None:
        input_df[scaler_cols] = scaler.transform(input_df[scaler_cols])
    else:
        print("Warning: Using default scaling")
        input_df[scaler_cols] = (input_df[scaler_cols] - input_df[scaler_cols].mean()) / input_df[scaler_cols].std()
    
    # Encode categorical features (Geography)
    input_df = pd.get_dummies(input_df, columns=['Geography'])
    
    # Ensure all geography columns are present
    for col in ['Geography_France', 'Geography_Germany', 'Geography_Spain']:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Apply label encoding to Gender if label encoder is available
    if le is not None:
        input_df['Gender'] = le.transform(input_df['Gender'])
    else:
        print("Warning: Using manual gender encoding")
        input_df['Gender'] = input_df['Gender'].apply(lambda x: 0 if x == 'Female' else 1)
    
    # Apply label encoding to geography columns if needed
    for col in ['Geography_France', 'Geography_Germany', 'Geography_Spain']:
        if col in input_df.columns:
            if le is not None:
                input_df[col] = le.transform(input_df[col])
            else:
                input_df[col] = input_df[col].astype(int)
    
    # Use the saved column order if available, otherwise use default order
    if column_order is not None:
        expected_columns = column_order
    else:
        print("Warning: Using default column order")
        expected_columns = [
            'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 
            'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
            'Geography_France', 'Geography_Germany', 'Geography_Spain'
        ]
    
    # Ensure all expected columns are present
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns to match training data
    input_df = input_df[expected_columns]

    return input_df.values

def predict(input_data):
    # Parse input dict from JSON
    input_dict = json.loads(input_data)
    # Preprocess input
    input_array = preprocess_input(input_dict, scaler, le, column_order)
    # Make prediction
    prediction = model.predict(input_array)
    # Convert to list and return
    return prediction.tolist()

if __name__ == '__main__':
    input_data = sys.argv[1]
    result = predict(input_data)
    print(json.dumps(result))