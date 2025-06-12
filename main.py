#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main Script for Squamish Housing Price Prediction Model

This script orchestrates the entire pipeline for the Squamish housing price prediction model:
1. Data preprocessing
2. Model training
3. Making predictions

Author: [Your Name]
Date: [Current Date]
"""

import os
import argparse
from dotenv import load_dotenv

# Import functions from other modules
from data_preprocessing import preprocess_data
from model_training import train_model
from make_predictions import predict_properties

def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Squamish Housing Price Prediction Pipeline')
    
    parser.add_argument('--preprocess', action='store_true',
                        help='Run data preprocessing')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--predict', action='store_true',
                        help='Make predictions')
    parser.add_argument('--all', action='store_true',
                        help='Run the entire pipeline')
    
    parser.add_argument('--input-file', type=str,
                        help='Path to input CSV file')
    parser.add_argument('--processed-file', type=str,
                        help='Path to save processed CSV file')
    parser.add_argument('--model-path', type=str,
                        help='Path to save/load model')
    parser.add_argument('--mystery-file', type=str,
                        help='Path to mystery CSV file for predictions')
    parser.add_argument('--output-file', type=str,
                        help='Path to save predictions')
    
    return parser.parse_args()

def main():
    """
    Main function to run the pipeline
    """
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Set default paths from environment variables if not provided
    input_file_path = args.input_file or os.getenv('INPUT_CSV_PATH', './data/squamish-parcels.csv')
    processed_file_path = args.processed_file or os.getenv('PROCESSED_CSV_PATH', './data/processed-squamish-parcels.csv')
    model_path = args.model_path or os.getenv('MODEL_PATH', './models/squamish-pipeline')
    mystery_file_path = args.mystery_file or os.getenv('MYSTERY_CSV_PATH', './data/mystery-parcels.csv')
    output_file_path = args.output_file or os.getenv('PREDICTIONS_PATH', './data/predictions.csv')
    
    # Run the pipeline based on arguments
    if args.all or args.preprocess:
        print("=== Data Preprocessing ===")
        preprocessed_data = preprocess_data(input_file_path, processed_file_path)
        print(f"Preprocessing complete. Dataset shape: {preprocessed_data.shape}")
    
    if args.all or args.train:
        print("\n=== Model Training ===")
        trained_model = train_model(processed_file_path, model_path)
        print("Model training complete")
    
    if args.all or args.predict:
        print("\n=== Making Predictions ===")
        predictions = predict_properties(model_path, mystery_file_path, output_file_path)
        print(f"Predictions complete. Generated {len(predictions)} predictions")
    
    # If no specific task is selected, print help
    if not (args.all or args.preprocess or args.train or args.predict):
        print("No task specified. Use --all to run the entire pipeline or specify individual tasks.")
        print("Run 'python main.py -h' for help.")

if __name__ == "__main__":
    main() 