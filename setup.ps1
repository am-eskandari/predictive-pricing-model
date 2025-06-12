# PowerShell script for setting up the Squamish Housing Price Prediction project

# Create directories if they don't exist
if (-not (Test-Path -Path ".\data")) {
    New-Item -ItemType Directory -Path ".\data"
    Write-Host "Created data directory"
}

if (-not (Test-Path -Path ".\models")) {
    New-Item -ItemType Directory -Path ".\models"
    Write-Host "Created models directory"
}

# Create .env file if it doesn't exist
if (-not (Test-Path -Path ".\.env")) {
    $envContent = @"
# Environment variables for the predictive pricing model

# Base paths
DATA_PATH=./data
MODEL_PATH=./models

# Input/output file paths
INPUT_CSV_PATH=./data/squamish-parcels.csv
PROCESSED_CSV_PATH=./data/processed-squamish-parcels.csv
MYSTERY_CSV_PATH=./data/mystery-parcels.csv
PREDICTIONS_PATH=./data/predictions.csv
"@
    Set-Content -Path ".\.env" -Value $envContent
    Write-Host "Created .env file"
}

# Install required packages
Write-Host "Installing required packages..."
pip install -r requirements.txt

Write-Host "`nSetup complete!"
Write-Host "Please place your data files in the 'data' directory:"
Write-Host "  - squamish-parcels.csv (main dataset)"
Write-Host "  - mystery-parcels.csv (test dataset)"
Write-Host "`nTo run the full pipeline, use:"
Write-Host "  python main.py --all" 