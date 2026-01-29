# Troubleshooting Guide - Phishing Detector

## Common Errors and Solutions

### Error 1: 404 (Not Found)

**Symptom:** Browser console shows `Failed to load resource: the server responded with a status of 404 (Not Found)`

**Cause:** Missing favicon.ico file

**Solution:** ✅ **FIXED** - Created `client/public/favicon.svg` and updated `index.html`

---

### Error 2: 503 (Service Unavailable) on /api/predict

**Symptom:** 
```
Failed to load resource: the server responded with a status of 503 (Service Unavailable)
:3000/api/predict:1
PhishingDetector.js:54  Prediction error:
```

**Cause:** ML API (Python FastAPI) is not running or crashed

**Solution:**

1. **Check if ML API is running:**
   ```powershell
   .\check_services.ps1
   ```

2. **Start the ML API manually:**
   ```powershell
   cd ml-api
   python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Or use the automated startup script:**
   ```powershell
   .\START_ALL.ps1
   ```

---

### Error 3: ML API Keeps Shutting Down

**Symptom:** ML API starts but then immediately shuts down

**Common Causes:**
- Port 8000 is already in use
- Python environment issues
- Model file is missing

**Solutions:**

1. **Kill existing process on port 8000:**
   ```powershell
   $port = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
   if ($port) {
       Stop-Process -Id $port.OwningProcess -Force
   }
   ```

2. **Check if model exists:**
   ```powershell
   Test-Path ml-api\models\phishing_model.pkl
   ```
   
   If FALSE, train the model:
   ```powershell
   cd ml-api
   python scripts/train_model.py
   ```

3. **Verify Python environment:**
   ```powershell
   python --version  # Should be Python 3.8+
   pip list | Select-String "fastapi|uvicorn"
   ```

---

### Error 4: Network Error / CORS Issues

**Symptom:** "Network error. Please check if the server is running."

**Cause:** Backend server (Express) is not running or CORS configuration issue

**Solution:**

1. **Check backend server:**
   ```powershell
   Get-NetTCPConnection -LocalPort 5000 -State Listen
   ```

2. **Start backend server:**
   ```powershell
   cd server
   npm start
   ```

3. **Verify .env file exists in server directory:**
   ```powershell
   Get-Content server\.env
   ```

---

### Error 5: MongoDB Connection Failed

**Symptom:** Backend server shows "MongoDB connection error"

**Solution:**

1. **Check MongoDB service:**
   ```powershell
   Get-Service | Where-Object { $_.Name -like "*mongo*" }
   ```

2. **Start MongoDB:**
   ```powershell
   net start MongoDB
   ```

---

## Quick Diagnosis Commands

### Check all services at once:
```powershell
.\check_services.ps1
```

### Check specific ports:
```powershell
Get-NetTCPConnection | Where-Object { $_.LocalPort -in @(3000, 5000, 8000) -and $_.State -eq 'Listen' } | Select-Object LocalPort, State, OwningProcess | Format-Table
```

### Test ML API directly:
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get | ConvertTo-Json
```

### Test prediction endpoint:
```powershell
$body = @{ url = "https://netfIix.com/billing/update" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:5000/api/predict" -Method Post -Body $body -ContentType "application/json" | ConvertTo-Json -Depth 10
```

---

## Service Architecture

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│  React Client   │ ------> │  Express Server │ ------> │   ML API        │
│  Port 3000      │         │  Port 5000      │         │   Port 8000     │
└─────────────────┘         └─────────────────┘         └─────────────────┘
                                     │
                                     ↓
                            ┌─────────────────┐
                            │    MongoDB      │
                            │  Port 27017     │
                            └─────────────────┘
```

**Flow:**
1. User enters URL in React app (port 3000)
2. React makes POST request to `/api/predict` → proxied to Express server (port 5000)
3. Express server validates and forwards to ML API (port 8000)
4. ML API loads model, extracts features, makes prediction
5. ML API returns result to Express
6. Express saves to MongoDB and returns to React
7. React displays result to user

---

## Files Fixed

### ✅ Created/Modified:
- **client/public/favicon.svg** - Created new favicon (red circle with "!" symbol)
- **client/public/index.html** - Updated to use favicon.svg instead of favicon.ico
- **check_services.ps1** - New health check script
- **TROUBLESHOOTING.md** - This file

---

## Recommended Startup Process

1. **Ensure MongoDB is running:**
   ```powershell
   Get-Service *mongo* | Start-Service
   ```

2. **Use the automated startup script:**
   ```powershell
   .\START_ALL.ps1
   ```

3. **Or start services individually in separate terminals:**
   
   **Terminal 1 - ML API:**
   ```powershell
   cd ml-api
   python -m uvicorn main:app --reload --port 8000
   ```
   
   **Terminal 2 - Backend Server:**
   ```powershell
   cd server
   npm start
   ```
   
   **Terminal 3 - React Client:**
   ```powershell
   cd client
   npm start
   ```

4. **Verify all services:**
   ```powershell
   .\check_services.ps1
   ```

5. **Access the app:**
   - Frontend: http://localhost:3000
   - API Docs: http://localhost:8000/docs

---

## Still Having Issues?

Run the comprehensive check:
```powershell
.\check_services.ps1
```

This will tell you exactly which service is down and what to do.
