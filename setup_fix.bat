@echo off
REM Quick Setup Script for Phishing Detector AI
REM Fixes the inverted labels and HTTPS bias issues

echo ================================================================================
echo PHISHING DETECTOR AI - COMPLETE FIX
echo ================================================================================
echo.
echo This script will:
echo   1. Create balanced dataset (removes HTTPS bias)
echo   2. Train model with correct labels
echo   3. Test predictions
echo.
pause

echo.
echo Step 1: Creating balanced dataset...
echo --------------------------------------------------------------------------------
cd ml-api
python scripts\fast_augment.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to create balanced dataset
    pause
    exit /b 1
)

echo.
echo Step 2: Training model with balanced dataset...
echo --------------------------------------------------------------------------------
python scripts\train_model.py --dataset ../data/final_dataset_balanced.csv
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to train model
    pause
    exit /b 1
)

echo.
echo Step 3: Testing predictions...
echo --------------------------------------------------------------------------------
echo.
echo Test 1: Phishing URL with HTTP
python scripts\predict_model.py "http://www.shprakserf.gq"

echo.
echo Test 2: Same phishing URL with HTTPS (should also be phishing!)
python scripts\predict_model.py "https://www.shprakserf.gq"

echo.
echo Test 3: Legitimate URL
python scripts\predict_model.py "https://www.google.com"

echo.
echo ================================================================================
echo SETUP COMPLETE!
echo ================================================================================
echo.
echo Model trained and ready to use.
echo Run comprehensive tests: python ..\test_phishing_detector.py
echo.
pause
