@echo off
echo Starting Anomaly Detection System...
echo.

REM Activate virtual environment
call anomaly_venv\Scripts\activate

REM Check if processed data exists
if not exist "data\processed\features\processed_data.pkl" (
    echo Processing dataset...
    python scripts\prepare_dataset.py
)

REM Check if model exists
if not exist "data\processed\models\anomaly_detection_model.joblib" (
    echo Training model...
    python scripts\quick_train.py
)

REM Start the API server
echo Starting API server...
python api\app.py

pause