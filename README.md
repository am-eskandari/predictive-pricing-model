# Predictive Pricing Model for Squamish Housing

A machine learning model for predicting housing prices in Squamish, British Columbia. This project uses historical property assessment data to build a predictive model that can estimate property values based on various features.

## Project Overview

This project analyzes real estate data from Squamish, BC to predict property values. The model utilizes historical assessment data, property characteristics, and location information to generate accurate price predictions. The system can be used by real estate professionals, property assessors, or homeowners to estimate property values.

## Technical Approach

The project follows a structured data science workflow:

1. **Data Preprocessing**: 
   - Cleaning and preparing the raw CSV data
   - Handling missing values through imputation or removal
   - Feature selection and engineering
   - Removing irrelevant columns

2. **Model Development**:
   - Automated machine learning with PyCaret
   - Model comparison and selection
   - Hyperparameter tuning
   - Model evaluation using metrics like R², MAE, and RMSE

3. **Prediction**:
   - Making predictions on new properties
   - Evaluating model performance on test data
   - Single property prediction functionality

## Technologies Used

- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **PyCaret**: Automated machine learning workflow
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib & Seaborn**: Data visualization
- **Python-dotenv**: Environment variable management

## Project Structure

```
predictive-pricing-model/
├── data/                      # Data directory
│   ├── squamish-parcels.csv   # Input data
│   └── mystery-parcels.csv    # Test data
├── models/                    # Saved models
├── .env                       # Environment variables
├── data_preprocessing.py      # Data preprocessing script
├── model_training.py          # Model training script
├── make_predictions.py        # Prediction script
├── main.py                    # Main script to run the pipeline
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## How to Use

1. **Setup**:
   ```
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   Create a `.env` file with your local file paths:
   ```
   DATA_PATH=./data
   MODEL_PATH=./models
   INPUT_CSV_PATH=./data/squamish-parcels.csv
   MYSTERY_CSV_PATH=./data/mystery-parcels.csv
   ```

3. **Run the Pipeline**:
   ```
   # Run the entire pipeline
   python main.py --all
   
   # Or run individual steps
   python main.py --preprocess
   python main.py --train
   python main.py --predict
   ```

4. **Custom Paths**:
   ```
   python main.py --all --input-file path/to/input.csv --output-file path/to/output.csv
   ```

## Results

The model achieves strong predictive performance on Squamish housing data, with high R² values indicating good fit. The most important features for prediction include recent assessed values, land value, and improvement value.

## Future Improvements

- Incorporate more geographical features
- Add time-series analysis for price trends
- Develop a web interface for easy predictions
- Expand the model to other regions in British Columbia

