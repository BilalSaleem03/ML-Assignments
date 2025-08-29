import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow INFO and WARNING messages

import warnings
warnings.filterwarnings("ignore")  # Suppress sklearn and other warnings

import sys
import json
import numpy as np
import pandas as pd
from keras.models import load_model
import joblib 
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
bitcoin_price_dir = os.path.join(script_dir, '..', '..', 'trained-model', 'bitcoin-price-prediction')


def load_preprocessing_objects():
    """Load all preprocessing objects from the bank-leaving-prediction folder"""
    try:
        # Load the scaler
        scaler = joblib.load(os.path.join(bitcoin_price_dir, 'scaler.pkl'))
        print("Scaler loaded successfully")
    except FileNotFoundError:
        print("Warning: Scaler file not found")
        scaler = None
    
    try:
        # Load column order
        column_order = joblib.load(os.path.join(bitcoin_price_dir, 'column_order.pkl'))
        print("Column order loaded successfully")
    except FileNotFoundError:
        print("Warning: Column order file not found")
        column_order = None
    
    return scaler, column_order

# Load model and preprocessing objects
model = load_model(os.path.join(bitcoin_price_dir, 'ANN_bitcoin_price_prediction_model.h5'))
scaler, feature_column = load_preprocessing_objects()



def preprocess_input(future_date_str, model, feature_columns,scaler):
    try:
        # Load recent historical data from CSV in the same folder as this script
        recent_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recent_data.csv')
        recent_data_for_features = pd.read_csv(recent_data_path, index_col='Timestamp', parse_dates=True)
        print(f"Number of rows in recent_data.csv: {len(recent_data_for_features)}")

        # 3. Prepare future data point
        try:
            future_date = pd.to_datetime(future_date_str)
        except ValueError:
            print("Invalid future date format. Please use YYYY-MM-DD.")
            return None

        last_date_in_recent_data = recent_data_for_features.index.max()

        if future_date <= last_date_in_recent_data:
             print(f"Error: The entered date {future_date_str} is not after the latest date in recent data ({last_date_in_recent_data.strftime('%Y-%m-%d')}).")
             return None

        # Create a new DataFrame row for the future date
        future_data_point = pd.DataFrame(index=[future_date])

        # Calculate time features for the future date
        future_data_point['hour'] = future_data_point.index.hour
        future_data_point['day_of_week'] = future_data_point.index.dayofweek
        future_data_point['day_of_month'] = future_data_point.index.day
        future_data_point['month'] = future_data_point.index.month
        future_data_point['is_weekend'] = (future_data_point.index.dayofweek >= 5).astype(int)
        future_data_point['hour_sin'] = np.sin(2 * np.pi * future_data_point['hour']/24)
        future_data_point['hour_cos'] = np.cos(2 * np.pi * future_data_point['hour']/24)
        future_data_point['day_sin'] = np.sin(2 * np.pi * future_data_point['day_of_week']/7)
        future_data_point['day_cos'] = np.cos(2 * np.pi * future_data_point['day_of_week']/7)

        # Calculate other features based on recent data and the future date
        cols_to_fill = [col for col in recent_data_for_features.columns if col not in future_data_point.columns]
        for col in cols_to_fill:
             future_data_point[col] = np.nan

        future_data_point = future_data_point[recent_data_for_features.columns]

        combined_data_for_calc = pd.concat([recent_data_for_features, future_data_point])

        last_close_recent = recent_data_for_features['Close'].iloc[-1]
        last_volume_recent = recent_data_for_features['Volume'].iloc[-1]

        combined_data_for_calc.loc[future_date, 'Open'] = last_close_recent
        combined_data_for_calc.loc[future_date, 'High'] = last_close_recent
        combined_data_for_calc.loc[future_date, 'Low'] = last_close_recent
        combined_data_for_calc.loc[future_date, 'Volume'] = last_volume_recent

        # Recalculate features with placeholders
        combined_data_for_calc['returns'] = combined_data_for_calc['Close'].pct_change()
        combined_data_for_calc['volatility_7'] = combined_data_for_calc['returns'].rolling(7).std()
        combined_data_for_calc['MA_7'] = combined_data_for_calc['Close'].rolling(7).mean()
        combined_data_for_calc['MA_24'] = combined_data_for_calc['Close'].rolling(24).mean()
        combined_data_for_calc['MA_7_24_ratio'] = combined_data_for_calc['MA_7'] / combined_data_for_calc['MA_24']
        combined_data_for_calc['price_MA_deviation'] = (combined_data_for_calc['Close'] - combined_data_for_calc['MA_7']) / combined_data_for_calc['MA_7']
        combined_data_for_calc['volume_MA_7'] = combined_data_for_calc['Volume'].rolling(7).mean()
        combined_data_for_calc['volume_ratio'] = combined_data_for_calc['Volume'] / combined_data_for_calc['volume_MA_7']
        combined_data_for_calc['high_low_range'] = (combined_data_for_calc['High'] - combined_data_for_calc['Low']) / combined_data_for_calc['Open']
        combined_data_for_calc['close_open_range'] = (combined_data_for_calc['Close'] - combined_data_for_calc['Open']) / combined_data_for_calc['Open']

        if 'numeric_timestamp' in feature_columns:
             # This assumes the original min timestamp is available, or calculate from recent_data_for_features if sufficient
             # For robustness in a backend, save original min_timestamp with scaler/column_order
             # Assuming min_timestamp_original is somehow available if numeric_timestamp is a feature
             # For this simplified example, we'll skip if it's not easily calculable from recent_data
             print("Warning: 'numeric_timestamp' feature calculation is simplified. Ensure original min_timestamp is available in a backend.")
             pass

        future_data_point_final = combined_data_for_calc.iloc[[-1]][feature_columns]
        future_data_point_final = future_data_point_final[feature_columns]
        print("qqqqqqqqq")
        print(future_data_point_final)

        # Fill NaN values with last available value from recent_data_for_features
        if future_data_point_final.isnull().any().any():
            print("Warning: Future data point contains NaN values in features. Filling with last available value from recent_data.csv.")
            last_values = recent_data_for_features.iloc[-1][feature_columns]
            future_data_point_final = future_data_point_final.fillna(last_values)
            print(future_data_point_final)

        print("\nPrepared future data point with calculated features:")
        print(future_data_point_final)

        # 4. Scale the future data point
        scaled_columns_for_scaler = list(scaler.feature_names_in_)

        # Fill all columns with last available value from recent_data_for_features
        last_values_all = recent_data_for_features.iloc[-1].reindex(scaled_columns_for_scaler)
        dummy_future_data = pd.DataFrame([last_values_all], index=future_data_point_final.index)

        # Overwrite with future data values for feature_columns
        for col in feature_columns:
            if col in dummy_future_data.columns:
                dummy_future_data[col] = future_data_point_final[col].values

        scaled_future_data_full = scaler.transform(dummy_future_data)

        feature_indices_for_scaler = [scaled_columns_for_scaler.index(col) for col in feature_columns if col in scaled_columns_for_scaler]
        scaled_future_features_ordered = scaled_future_data_full[:, feature_indices_for_scaler]

        print("\nScaled future data point features (subset used for prediction):")
        print(scaled_future_features_ordered)

        # 5. Make prediction
        prediction_scaled_future = model.predict(scaled_future_features_ordered)

        # 6. Inverse transform the prediction back to the original scale
        dummy_prediction_data_future = pd.DataFrame(np.zeros((len(prediction_scaled_future), len(scaled_columns_for_scaler))), columns=scaled_columns_for_scaler)

        try:
            target_column_index_future = list(scaler.feature_names_in_).index('target')
        except ValueError:
            print("Warning: 'target' column not found in scaler's feature names. Inverse transformation may not be possible.")
            target_column_index_future = None

        if target_column_index_future is not None:
            dummy_prediction_data_future['target'] = prediction_scaled_future.flatten()

            original_scale_prediction_data_future = scaler.inverse_transform(dummy_prediction_data_future)

            prediction_original_scale_future = original_scale_prediction_data_future[:, target_column_index_future]

            # 7. Display predicted price
            print(f"\nPredicted Close price for {future_date.strftime('%Y-%m-%d')}: {prediction_original_scale_future[0]}")
            return prediction_original_scale_future[0]
        else:
             print(f"\nPredicted value (scaled) for {future_date.strftime('%Y-%m-%d')}: {prediction_scaled_future[0][0]}")
             return None

    except FileNotFoundError as fnf_error:
        print(f"File not found error: {fnf_error}. Make sure the model, scaler, column order, and recent data files exist at the specified paths.")
        return None
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None




def predict(input_data):
    # Parse input dict from JSON
    input_dict = json.loads(input_data)
    future_date_str = input_dict['date']

    prediction = preprocess_input(future_date_str, model, feature_column, scaler)
    
    if prediction is None:
        print("Prediction aborted due to preprocessing error.")
        # Output a valid JSON error for frontend
        return {"error": "Preprocessing failed. Check your input and recent_data.csv."}
    
    # Return the prediction
    return {"prediction": float(prediction)}

if __name__ == '__main__':
    try:
        input_data = sys.argv[1]
        result = predict(input_data)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)