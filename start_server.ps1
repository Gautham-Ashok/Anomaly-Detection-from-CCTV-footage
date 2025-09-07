Write-Host "Starting Anomaly Detection System..." -ForegroundColor Green
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path "anomaly_venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv anomaly_venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
.\anomaly_venv\Scripts\Activate.ps1

# Install requirements
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Check if processed data exists
if (-not (Test-Path "data\processed\features\processed_data.pkl")) {
    Write-Host "Processing dataset..." -ForegroundColor Yellow
    python scripts\prepare_dataset.py
}

# Check if model exists
if (-not (Test-Path "data\processed\models\anomaly_detection_model.joblib")) {
    Write-Host "Training model..." -ForegroundColor Yellow
    python scripts\quick_train.py
}

# Start the API server
Write-Host "Starting API server..." -ForegroundColor Green
Write-Host "Server will be available at: http://localhost:5000" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
python api\app.py