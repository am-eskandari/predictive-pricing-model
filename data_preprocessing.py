#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Preprocessing Script for Squamish Housing Price Prediction Model

This script handles the preprocessing of real estate data for Squamish, BC.
It cleans the dataset by removing irrelevant columns, handling missing values,
and preparing the data for model training.

Author: [Your Name]
Date: [Current Date]
"""

import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import warnings

# Suppress FutureWarnings to keep output clean
warnings.filterwarnings("ignore", category=FutureWarning)

def load_data(file_path):
    """
    Load the dataset from CSV and handle basic cleaning
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Loaded and initially cleaned dataset
    """
    print(f"Loading dataset from {file_path}")
    dataset = pd.read_csv(file_path, on_bad_lines='skip')
    
    # Adjust display settings for ease of viewing
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    # Handle missing values in target column
    dataset = dataset.dropna(subset=['2023 Assessed Total'])
    print(f"Missing 2023 Assessed Total values: {dataset['2023 Assessed Total'].isnull().sum()}")
    
    return dataset

def remove_irrelevant_columns(dataset):
    """
    Remove columns that are not relevant for the analysis
    
    Args:
        dataset (pandas.DataFrame): Original dataset
        
    Returns:
        pandas.DataFrame: Dataset with irrelevant columns removed
    """
    print("Removing irrelevant columns...")
    
    # Identify columns to remove
    columns_to_remove = [col for col in dataset.columns if 'Photo' in col]
    columns_to_remove.extend([col for col in dataset.columns if 'Unnamed' in col])
    columns_to_remove.extend([col for col in dataset.columns if 'Virtual Tour' in col])
    columns_to_remove.extend([col for col in dataset.columns if 'LEGAL_DETAIL' in col])
    columns_to_remove.extend([col for col in dataset.columns if 'CIVIC_ADDRESS' in col])
    columns_to_remove.extend([col for col in dataset.columns if 'STREET' in col])
    columns_to_remove.extend([col for col in dataset.columns if 'Last MLS' in col])
    columns_to_remove.extend([col for col in dataset.columns if 'MLS_Images' in col])
    
    # Drop the selected columns from the dataset
    cleaned_dataset = dataset.drop(columns=columns_to_remove)
    
    print(f"Removed {len(columns_to_remove)} irrelevant columns")
    return cleaned_dataset

def analyze_missing_values(dataset):
    """
    Analyze missing values in the dataset
    
    Args:
        dataset (pandas.DataFrame): Dataset to analyze
        
    Returns:
        pandas.Series: Series containing counts of missing values for each column
    """
    print("Analyzing missing values...")
    
    # Determine the total number of rows in the dataset
    total_rows = dataset.shape[0]
    print(f"Total number of rows in the dataset: {total_rows}")
    
    # Calculate the number of missing values in each column
    missing_values = dataset.isnull().sum()
    # Filter to keep only columns with missing values
    missing_values = missing_values[missing_values > 0]
    
    # Print out columns with missing values
    if len(missing_values) > 0:
        for column, n_missing in missing_values.items():
            missing_percentage = (n_missing / total_rows) * 100
            print(f"{column}: {n_missing} missing values ({missing_percentage:.2f}%)")
    else:
        print("No missing values found.")
    
    return missing_values

def handle_missing_values(dataset, missing_values, threshold=10):
    """
    Handle missing values in the dataset based on a threshold percentage
    
    Args:
        dataset (pandas.DataFrame): Dataset with missing values
        missing_values (pandas.Series): Series containing counts of missing values
        threshold (int): Percentage threshold for dropping columns
        
    Returns:
        pandas.DataFrame: Dataset with handled missing values
    """
    print(f"Handling missing values with threshold {threshold}%...")
    
    total_rows = dataset.shape[0]
    
    # Identify columns with high percentage of missing values
    columns_to_drop = []
    for column, n_missing in missing_values.items():
        missing_percentage = (n_missing / total_rows) * 100
        if missing_percentage > threshold:
            columns_to_drop.append(column)
    
    if columns_to_drop:
        print(f"Dropping columns with >{threshold}% missing values: {', '.join(columns_to_drop)}")
        dataset = dataset.drop(columns=columns_to_drop)
    
    # For remaining columns with missing values, use appropriate imputation
    # Numeric columns: use median
    # Categorical columns: use mode
    numeric_cols = dataset.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = dataset.select_dtypes(include=['object']).columns
    
    for col in numeric_cols:
        if dataset[col].isnull().sum() > 0:
            median_value = dataset[col].median()
            dataset[col].fillna(median_value, inplace=True)
            print(f"Filled missing values in {col} with median: {median_value}")
    
    for col in categorical_cols:
        if dataset[col].isnull().sum() > 0:
            mode_value = dataset[col].mode()[0]
            dataset[col].fillna(mode_value, inplace=True)
            print(f"Filled missing values in {col} with mode: {mode_value}")
    
    return dataset

def clean_price_columns(dataset):
    """
    Clean price columns by removing '$' and ',' characters and converting to numeric
    
    Args:
        dataset (pandas.DataFrame): Dataset with price columns
        
    Returns:
        pandas.DataFrame: Dataset with cleaned price columns
    """
    print("Cleaning price columns...")
    
    # Identify price columns (those with 'Price' or 'Assessed' in their name)
    price_columns = [col for col in dataset.columns if 'Price' in col or 'Assessed' in col]
    
    for col in price_columns:
        if dataset[col].dtype == 'object':
            # Remove '$' and ',' characters and convert to numeric
            dataset[col] = dataset[col].astype(str).str.replace('$', '', regex=False)
            dataset[col] = dataset[col].str.replace(',', '', regex=False)
            dataset[col] = pd.to_numeric(dataset[col], errors='coerce')
            print(f"Cleaned price column: {col}")
    
    return dataset

def preprocess_data(input_file_path, output_file_path=None):
    """
    Main function to preprocess the data
    
    Args:
        input_file_path (str): Path to input CSV file
        output_file_path (str, optional): Path to save processed CSV file
        
    Returns:
        pandas.DataFrame: Preprocessed dataset
    """
    # Load the dataset
    dataset = load_data(input_file_path)
    
    # Remove irrelevant columns
    dataset = remove_irrelevant_columns(dataset)
    
    # Analyze missing values
    missing_values = analyze_missing_values(dataset)
    
    # Handle missing values
    dataset = handle_missing_values(dataset, missing_values)
    
    # Clean price columns
    dataset = clean_price_columns(dataset)
    
    # Save processed data if output path is provided
    if output_file_path:
        dataset.to_csv(output_file_path, index=False)
        print(f"Preprocessed data saved to {output_file_path}")
    
    return dataset

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Get file paths from environment variables or use defaults
    input_file_path = os.getenv('INPUT_CSV_PATH', './data/squamish-parcels.csv')
    output_file_path = os.getenv('PROCESSED_CSV_PATH', './data/processed-squamish-parcels.csv')
    
    # Preprocess the data
    preprocessed_data = preprocess_data(input_file_path, output_file_path)
    
    print(f"Preprocessing complete. Dataset shape: {preprocessed_data.shape}")
    print(f"Columns in preprocessed data: {', '.join(preprocessed_data.columns)}") 