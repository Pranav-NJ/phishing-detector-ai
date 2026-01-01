# Quick Start Script for Phishing Detector
# PowerShell script to start all services

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "🚀 PHISHING DETECTOR - QUICK START" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if model exists
$modelPath = "ml-api\models\phishing_model.pkl"
if (-Not (Test-Path $modelPath)) {
    Write-Host "❌ Model not found!" -ForegroundColor Red
    Write-Host "   Training model first..." -ForegroundColor Yellow
    Write-Host ""
    
    Set-Location ml-api
    python scripts/train_model.py
    Set-Location ..
    
    if (-Not (Test-Path $modelPath)) {
        Write-Host "❌ Model training failed!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host ""
    Write-Host "✅ Model trained successfully!" -ForegroundColor Green
    Write-Host ""
}

Write-Host "✅ Model found: $modelPath" -ForegroundColor Green
Write-Host ""

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "📋 STARTING SERVICES" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "This will open 3 terminal windows:" -ForegroundColor Yellow
Write-Host "  1. ML API (FastAPI) on http://localhost:8000" -ForegroundColor White
Write-Host "  2. Backend Server (Express) on http://localhost:5000" -ForegroundColor White
Write-Host "  3. Frontend Client (React) on http://localhost:3000" -ForegroundColor White
Write-Host ""

Read-Host "Press Enter to continue..."

# Start ML API
Write-Host "🔧 Starting ML API..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\ml-api'; Write-Host '🤖 Starting ML API (FastAPI)...' -ForegroundColor Cyan; uvicorn main:app --reload --port 8000"

Start-Sleep -Seconds 2

# Start Backend Server
Write-Host "🔧 Starting Backend Server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\server'; Write-Host '🖥️  Starting Backend Server (Express)...' -ForegroundColor Cyan; npm start"

Start-Sleep -Seconds 2

# Start Frontend Client
Write-Host "🔧 Starting Frontend Client..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\client'; Write-Host '🌐 Starting Frontend Client (React)...' -ForegroundColor Cyan; npm start"

Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Green
Write-Host "✅ ALL SERVICES STARTED!" -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Access the application:" -ForegroundColor Cyan
Write-Host "  🌐 Frontend:  http://localhost:3000" -ForegroundColor White
Write-Host "  🖥️  Backend:   http://localhost:5000" -ForegroundColor White
Write-Host "  🤖 ML API:    http://localhost:8000" -ForegroundColor White
Write-Host "  📚 API Docs:  http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "To stop services: Close each terminal window" -ForegroundColor Yellow
Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "Press any key to exit this window..." -ForegroundColor White
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
