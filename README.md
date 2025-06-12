# Predictive Pricing Model for Squamish Housing

A machine learning model for predicting housing prices in Squamish, British Columbia. This project uses historical property assessment data to build a predictive model that can estimate property values based on various features.

## Project Overview

This project analyzes real estate data from Squamish, BC to predict property values. The model utilizes historical assessment data, property characteristics, and location information to generate accurate price predictions. The system can be used by real estate professionals, property assessors, or homeowners to estimate property values.

## Technical Approach

The project follows a structured data science workflow:

1. **Data Preprocessing** (`data_preprocessing.py`): 
   - **Data Cleaning**:
     - Removed irrelevant columns (photos, virtual tours, legal details)
     - Handled missing values using median imputation for numeric data and mode imputation for categorical data
     - Cleaned price columns by removing currency symbols and standardizing formats
   - **Feature Engineering**:
     - Selected key features based on correlation with target variable
     - Focused on recent assessment values, land values, and property characteristics
     - Removed features with high percentage (>10%) of missing values
   - **Data Validation**:
     - Ensured target variable ('2023 Assessed Total') has no missing values
     - Verified data types and formats are consistent
     - Removed duplicate entries if present

2. **Model Development** (`model_training.py`):
   - **Feature Selection**:
     - Used PyCaret's built-in feature importance analysis
     - Selected features with strongest correlation to target variable:
       - Recent assessed values (2021-2023)
       - Land and improvement values
       - Property characteristics (bedrooms, bathrooms, etc.)
   - **Model Selection**:
     - Utilized PyCaret's automated ML workflow to compare multiple algorithms:
       - Linear Regression
       - Random Forest
       - XGBoost
       - LightGBM
     - Selected best performing model based on R² score
   - **Hyperparameter Tuning**:
     - Performed automated hyperparameter optimization using PyCaret
     - Used 5-fold cross-validation to prevent overfitting
     - Optimized for R² metric

3. **Prediction Pipeline** (`make_predictions.py`):
   - **Model Deployment**:
     - Saved trained model using PyCaret's save_model function
     - Implemented model loading and prediction functions
   - **Prediction Features**:
     - Focused on key predictive features:
       - 2023 Assessed Land
       - 2022 Assessed Total
       - 2023 Assessed Improvements
       - Last Sold Date
       - 2022 Assessed Land
       - 2021 Assessed Total
   - **Output Generation**:
     - Created functions for both batch and single property predictions
     - Included property identifiers (PID) in predictions
     - Generated CSV output with predicted values

4. **Pipeline Integration** (`main.py`):
   - **Command Line Interface**:
     - Implemented argparse for flexible command-line usage
     - Added options for running full pipeline or individual steps
   - **Environment Management**:
     - Used python-dotenv for configuration
     - Separated file paths and configurations into .env file
   - **Error Handling**:
     - Added robust error checking for file paths
     - Included input validation for all parameters
     - Provided clear error messages and logging

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
│   ├── squamish-parcels.csv   # Raw input data
│   ├── cleaned-squamish-parcels.csv  # Preprocessed data
│   └── mystery-parcels.csv    # Test data
├── models/                    # Saved models
│   └── squamish-pipeline.pkl  # Trained model
├── notebooks/                 # Jupyter notebooks
│   ├── preprocessing.ipynb    # Original preprocessing notebook
│   ├── model_training.ipynb   # Original model training notebook
│   └── testing.ipynb         # Original testing notebook
├── src/                      # Source code
│   ├── data_preprocessing.py  # Data preprocessing script
│   ├── model_training.py     # Model training script
│   ├── make_predictions.py   # Prediction script
│   └── main.py              # Main pipeline script
├── .env                      # Environment variables (not in repo)
├── .env.example             # Example environment file
├── requirements.txt         # Project dependencies
├── setup.ps1               # Setup script for Windows
└── README.md              # Project documentation
```

## How to Use

1. **Setup**:
   ```powershell
   # Run the setup script (Windows)
   .\setup.ps1
   ```
   This will:
   - Create necessary directories
   - Set up the environment file
   - Install required packages

2. **Configure Environment**:
   The setup script will create a `.env` file with default paths. Update them if needed:
   ```
   DATA_PATH=./data
   MODEL_PATH=./models
   INPUT_CSV_PATH=./data/squamish-parcels.csv
   MYSTERY_CSV_PATH=./data/mystery-parcels.csv
   ```

3. **Run the Pipeline**:
   ```bash
   # Run the entire pipeline
   python src/main.py --all
   
   # Or run individual steps
   python src/main.py --preprocess
   python src/main.py --train
   python src/main.py --predict
   ```

4. **Custom Paths**:
   ```bash
   python src/main.py --all --input-file path/to/input.csv --output-file path/to/output.csv
   ```

## Results

The model achieves strong predictive performance on Squamish housing data, with high R² values indicating good fit. The most important features for prediction include recent assessed values, land value, and improvement value.

## Future Improvements

- Incorporate more geographical features
- Add time-series analysis for price trends
- Develop a web interface for easy predictions
- Expand the model to other regions in British Columbia

## Author

[am-eskandari](https://github.com/am-eskandari)

