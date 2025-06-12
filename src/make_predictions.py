#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prediction Script for Squamish Housing Price Prediction Model

This script loads a trained model and makes predictions on new data.
It can be used to predict housing prices for new properties or for
properties with missing assessed values.

Author: am-eskandari
"""

import os
import pandas as pd
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Import PyCaret regression module
try:
    from pycaret.regression import load_model, predict_model
except ImportError:
    print("PyCaret is not installed. Installing it now...")
    import subprocess
    subprocess.check_call(["pip", "install", "pycaret"])
    from pycaret.regression import load_model, predict_model

def load_model_from_path(model_path):
    """
    Load a trained model from the specified path
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        object: The loaded model
    """
    print(f"Loading model from {model_path}")
    loaded_model = load_model(model_path)
    return loaded_model

def load_prediction_data(file_path):
    """
    Load data for prediction from the specified path
    
    Args:
        file_path (str): Path to the CSV file containing data for prediction
        
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    print(f"Loading prediction data from {file_path}")
    dataset = pd.read_csv(file_path)
    return dataset

def select_prediction_features(dataset):
    """
    Select the features required for prediction
    
    Args:
        dataset (pandas.DataFrame): Dataset containing features for prediction
        
    Returns:
        pandas.DataFrame: Dataset with selected features
    """
    print("Selecting features for prediction...")
    
    # Features selected to best predict the 2023 Assessed Total
    selected_features = [
        'PID',
        '2023 Assessed Land',
        '2022 Assessed Total',
        '2023 Assessed Improvements',
        'Last Sold',
        '2022 Assessed Land',
        '2021 Assessed Total',
    ]
    
    # Check if '2023 Assessed Total' is in the dataset (for evaluation)
    if '2023 Assessed Total' in dataset.columns:
        selected_features.append('2023 Assessed Total')
    
    # Check which features are available in the dataset
    available_features = [col for col in selected_features if col in dataset.columns]
    
    if len(available_features) < len(selected_features) - 1:  # -1 for the target column
        missing_features = set(selected_features) - set(available_features) - {'2023 Assessed Total'}
        print(f"Warning: Missing required features: {', '.join(missing_features)}")
    
    # Create a new DataFrame with only the selected columns
    prediction_data = dataset[available_features]
    
    return prediction_data

def make_predictions(model, data):
    """
    Make predictions using the loaded model
    
    Args:
        model: The loaded model
        data (pandas.DataFrame): Data for prediction
        
    Returns:
        pandas.DataFrame: DataFrame with predictions
    """
    print("Making predictions...")
    
    # Check if PID column exists
    pid_column = None
    if 'PID' in data.columns:
        pid_column = data['PID'].copy()
        data = data.drop(columns=['PID'])
    
    # Make predictions
    predictions = predict_model(model, data=data)
    
    # Rename the prediction column from 'prediction_label' to '2023 Assessed Total'
    if 'prediction_label' in predictions.columns:
        predictions = predictions.rename(columns={'prediction_label': '2023 Assessed Total Predicted'})
    
    # Reintroduce the PID column if it was present
    if pid_column is not None:
        predictions.insert(0, 'PID', pid_column.reset_index(drop=True))
    
    return predictions

def save_predictions(predictions, output_path):
    """
    Save predictions to a CSV file
    
    Args:
        predictions (pandas.DataFrame): DataFrame with predictions
        output_path (str): Path to save the predictions
    """
    predictions.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

def predict_single_property(model, property_data):
    """
    Make a prediction for a single property
    
    Args:
        model: The loaded model
        property_data (dict): Dictionary containing property features
        
    Returns:
        float: Predicted price
    """
    print("Making prediction for a single property...")
    
    # Convert the input data into a DataFrame
    input_df = pd.DataFrame(property_data)
    
    # Make prediction
    prediction = predict_model(model, data=input_df)
    
    # Rename the prediction column
    if 'prediction_label' in prediction.columns:
        prediction = prediction.rename(columns={'prediction_label': '2023 Assessed Total Predicted'})
    
    return prediction

def predict_properties(model_path, input_file_path, output_file_path=None):
    """
    Main function to make predictions
    
    Args:
        model_path (str): Path to the saved model
        input_file_path (str): Path to the input CSV file
        output_file_path (str, optional): Path to save the predictions
        
    Returns:
        pandas.DataFrame: DataFrame with predictions
    """
    # Load the model
    model = load_model_from_path(model_path)
    
    # Load the data
    data = load_prediction_data(input_file_path)
    
    # Select features
    prediction_data = select_prediction_features(data)
    
    # Make predictions
    predictions = make_predictions(model, prediction_data)
    
    # Save predictions if output path is provided
    if output_file_path:
        save_predictions(predictions, output_file_path)
    
    return predictions

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Get file paths from environment variables or use defaults
    model_path = os.getenv('MODEL_PATH', './models/squamish-pipeline')
    input_file_path = os.getenv('MYSTERY_CSV_PATH', './data/mystery-parcels.csv')
    output_file_path = os.getenv('PREDICTIONS_PATH', './data/predictions.csv')
    
    # Make predictions
    predictions = predict_properties(model_path, input_file_path, output_file_path)
    
    # Display the first few predictions
    print("\nSample predictions:")
    print(predictions.head())
    
    # Example of predicting a single property
    print("\nExample of single property prediction:")
    single_property = {
        '2023 Assessed Land': [1068000],
        '2022 Assessed Total': [1391000],
        '2023 Assessed Improvements': [289000],
        'Last Sold': ["2018-06-21"],
        '2022 Assessed Land': [1070000],
        '2021 Assessed Total': [1225000],
    }
    single_prediction = predict_single_property(load_model_from_path(model_path), single_property)
    print(single_prediction) 