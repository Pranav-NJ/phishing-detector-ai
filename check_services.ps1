# Service Health Check Script for Phishing Detector
# Run this to check if all services are running properly

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " PHISHING DETECTOR - SERVICE STATUS" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$allHealthy = $true

# Check React Client (port 3000)
Write-Host "1. React Client (port 3000)..." -NoNewline
try {
    $client = Invoke-RestMethod -Uri "http://localhost:3000" -Method Get -TimeoutSec 2 -ErrorAction Stop
    Write-Host " ✓ RUNNING" -ForegroundColor Green
} catch {
    Write-Host " ✗ NOT RUNNING" -ForegroundColor Red
    $allHealthy = $false
}

# Check Node Server (port 5000)
Write-Host "2. Node Server (port 5000)..." -NoNewline
try {
    $server = Invoke-RestMethod -Uri "http://localhost:5000/api/health" -Method Get -TimeoutSec 2 -ErrorAction Stop
    Write-Host " ✓ RUNNING ($($server.status))" -ForegroundColor Green
} catch {
    Write-Host " ✗ NOT RUNNING" -ForegroundColor Red
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Yellow
    $allHealthy = $false
}

# Check ML API (port 8000)
Write-Host "3. ML API (port 8000)..." -NoNewline
try {
    $ml = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get -TimeoutSec 2 -ErrorAction Stop
    Write-Host " ✓ RUNNING (Model: $($ml.model_version))" -ForegroundColor Green
} catch {
    Write-Host " ✗ NOT RUNNING" -ForegroundColor Red
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Yellow
    $allHealthy = $false
}

# Check MongoDB
Write-Host "4. MongoDB..." -NoNewline
try {
    $mongo = Get-Service -ErrorAction Stop | Where-Object { $_.Name -like "*mongo*" }
    if ($mongo -and $mongo.Status -eq "Running") {
        Write-Host " ✓ RUNNING" -ForegroundColor Green
    } else {
        Write-Host " ✗ NOT RUNNING" -ForegroundColor Red
        $allHealthy = $false
    }
} catch {
    Write-Host " ? UNKNOWN" -ForegroundColor Yellow
}

Write-Host "`n========================================" -ForegroundColor Cyan

if ($allHealthy) {
    Write-Host " ALL SERVICES ARE HEALTHY ✓" -ForegroundColor Green
    Write-Host "`n Testing prediction endpoint..." -ForegroundColor Yellow
    try {
        $body = @{ url = "https://netfIix.com/billing/update" } | ConvertTo-Json
        $result = Invoke-RestMethod -Uri "http://localhost:5000/api/predict" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 5
        Write-Host " ✓ Prediction API working!" -ForegroundColor Green
        Write-Host "   Result: $(if($result.prediction){'PHISHING'}else{'SAFE'}) - Risk: $($result.risk_level)" -ForegroundColor White
    } catch {
        Write-Host " ✗ Prediction API failed: $($_.Exception.Message)" -ForegroundColor Red
    }
} else {
    Write-Host " SOME SERVICES ARE DOWN!" -ForegroundColor Red
    Write-Host "`n To start all services, run:" -ForegroundColor Yellow
    Write-Host "   .\START_ALL.ps1" -ForegroundColor White
}

Write-Host "========================================`n" -ForegroundColor Cyan
