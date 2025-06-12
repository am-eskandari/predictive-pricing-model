#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Training Script for Squamish Housing Price Prediction

This script builds and trains a machine learning model to predict housing prices
in Squamish, BC. It uses the PyCaret library to automate the ML workflow including
feature selection, model comparison, hyperparameter tuning, and model evaluation.

Author: [Your Name]
Date: [Current Date]
"""

import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Import PyCaret regression module
try:
    from pycaret.regression import setup, compare_models, tune_model, finalize_model, save_model
except ImportError:
    print("PyCaret is not installed. Installing it now...")
    import subprocess
    subprocess.check_call(["pip", "install", "pycaret"])
    from pycaret.regression import setup, compare_models, tune_model, finalize_model, save_model

def load_processed_data(file_path):
    """
    Load the preprocessed dataset
    
    Args:
        file_path (str): Path to the preprocessed CSV file
        
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    print(f"Loading preprocessed dataset from {file_path}")
    dataset = pd.read_csv(file_path)
    return dataset

def select_features(dataset):
    """
    Select the most relevant features for model training
    
    Args:
        dataset (pandas.DataFrame): Preprocessed dataset
        
    Returns:
        pandas.DataFrame: Dataset with selected features
    """
    print("Selecting features for model training...")
    
    # Drop columns with high correlation or low importance
    columns_to_drop = [
        'Previous Sold2', 'Previous Price2', 'Type2',
        'Previous Sold3', 'Previous Price3', 'Type3',
        'Previous Sold4', 'Previous Price4', 'Type4',
        'Previous Sold5', 'Previous Price5', 'Type5',
        '1st Floor', '2nd Floor', '3rd Floor', '4th Floor',
        'Finished', 'Unfinshed', 'Total Size', 'Storage',
        'Half Bathrooms',
    ]
    
    # Check if columns exist before dropping
    existing_columns = [col for col in columns_to_drop if col in dataset.columns]
    if existing_columns:
        dataset = dataset.drop(columns=existing_columns)
        print(f"Dropped {len(existing_columns)} columns")
    
    # Keep track of the PID column for later reference if it exists
    pid_column = None
    if 'PID' in dataset.columns:
        pid_column = dataset['PID'].copy()
        dataset = dataset.drop(columns=['PID'])
        print("PID column saved for reference and removed from training data")
    
    return dataset, pid_column

def setup_pycaret(dataset, target_column='2023 Assessed Total'):
    """
    Set up PyCaret for regression modeling
    
    Args:
        dataset (pandas.DataFrame): Dataset with selected features
        target_column (str): Name of the target column to predict
        
    Returns:
        tuple: Tuple containing the setup object and the prepared dataset
    """
    print(f"Setting up PyCaret with target column: {target_column}")
    
    # Check if target column exists
    if target_column not in dataset.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    # Set up PyCaret
    setup_obj = setup(
        data=dataset,
        target=target_column,
        session_id=123,  # For reproducibility
        normalize=True,
        transformation=True,
        ignore_features=['Status'],  # Ignore non-predictive columns
        use_gpu=False,  # Set to True if GPU is available
        silent=True,  # Set to False for more verbose output
    )
    
    return setup_obj

def train_and_evaluate_models():
    """
    Train and evaluate multiple regression models
    
    Returns:
        object: The best model after comparison
    """
    print("Training and evaluating models...")
    
    # Compare models and select the best one
    best_model = compare_models(
        sort='R2',  # Sort by R-squared
        n_select=3,  # Select top 3 models
        verbose=True
    )
    
    print(f"Best model selected: {type(best_model).__name__}")
    return best_model

def tune_best_model(best_model):
    """
    Tune the hyperparameters of the best model
    
    Args:
        best_model: The best model from compare_models
        
    Returns:
        object: The tuned model
    """
    print("Tuning the best model...")
    
    # Tune the model with 5-fold cross-validation
    tuned_model = tune_model(
        best_model,
        optimize='R2',
        n_iter=10,  # Number of iterations for hyperparameter tuning
        verbose=True
    )
    
    print("Model tuning completed")
    return tuned_model

def finalize_and_save_model(tuned_model, model_path):
    """
    Finalize the model and save it to disk
    
    Args:
        tuned_model: The tuned model
        model_path (str): Path to save the model
        
    Returns:
        object: The finalized model
    """
    print("Finalizing and saving the model...")
    
    # Finalize the model (train on entire dataset)
    final_model = finalize_model(tuned_model)
    
    # Save the model
    save_model(final_model, model_path)
    print(f"Model saved to {model_path}")
    
    return final_model

def train_model(input_file_path, model_output_path):
    """
    Main function to train the model
    
    Args:
        input_file_path (str): Path to the preprocessed CSV file
        model_output_path (str): Path to save the trained model
        
    Returns:
        object: The trained and finalized model
    """
    # Load the preprocessed data
    dataset = load_processed_data(input_file_path)
    
    # Select features
    dataset, pid_column = select_features(dataset)
    
    # Set up PyCaret
    setup_pycaret(dataset)
    
    # Train and evaluate models
    best_model = train_and_evaluate_models()
    
    # Tune the best model
    tuned_model = tune_best_model(best_model)
    
    # Finalize and save the model
    final_model = finalize_and_save_model(tuned_model, model_output_path)
    
    return final_model

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Get file paths from environment variables or use defaults
    input_file_path = os.getenv('PROCESSED_CSV_PATH', './data/processed-squamish-parcels.csv')
    model_output_path = os.getenv('MODEL_PATH', './models/squamish-pipeline')
    
    # Train the model
    trained_model = train_model(input_file_path, model_output_path)
    
    print("Model training completed successfully") 