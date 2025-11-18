# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a full-stack AI-powered phishing website detection system built with:
- **Frontend**: React (port 3000) 
- **Backend**: Express.js + MongoDB (port 5000)
- **ML Service**: FastAPI + scikit-learn (port 8000)

The system analyzes URLs using 40+ features through a Random Forest classifier to detect phishing websites in real-time.

## Architecture & Data Flow

```
React Client ←→ Express Server ←→ FastAPI ML Service
     ↓              ↓                    ↓
  User Input    MongoDB Storage    ML Model + Features
```

**Critical Dependencies:**
- The Express server proxies predictions to the ML service
- The ML service requires a trained model at `./ml-api/models/phishing_model.pkl`
- All three services must run concurrently for full functionality

## Essential Commands

### Initial Setup
```bash
# Environment setup (required first)
cp server/.env.template server/.env
cp ml-api/.env.template ml-api/.env

# Install dependencies for all services
cd client && npm install
cd ../server && npm install  
cd ../ml-api && pip install -r requirements.txt

# Train the ML model (required before first run)
cd ml-api
python scripts/data_collection.py
python scripts/train_model.py
```

### Development Workflow
```bash
# Start all services (requires 4 terminals)
# Terminal 1: MongoDB
mongod

# Terminal 2: ML Service
cd ml-api && python main.py

# Terminal 3: Backend API
cd server && npm run dev

# Terminal 4: Frontend
cd client && npm start
```

### Testing & Validation
```bash
# Test individual services
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"url": "https://www.google.com"}'
curl -X POST "http://localhost:5000/api/predict" -H "Content-Type: application/json" -d '{"url": "https://www.google.com"}'

# Health checks
curl http://localhost:8000/health
curl http://localhost:5000/api/health

# Run tests
cd client && npm test
cd ml-api && python -m pytest  # if tests exist
```

### Production Build
```bash
# Frontend production build
cd client && npm run build

# Backend production start
cd server && npm start

# ML Service production
cd ml-api && python main.py
```

## Key Configuration

### Environment Variables
- **Server**: MongoDB URI, ML API URL, CORS origin, rate limiting
- **ML API**: Model path, CORS origins, debug mode, host/port
- **Client**: Proxies to backend via package.json during development

### Critical Paths
- **Model File**: `ml-api/models/phishing_model.pkl` (must exist)
- **Feature Extractor**: `ml-api/utils/feature_extractor.py`
- **Training Scripts**: `ml-api/scripts/` (data_collection.py, train_model.py)

## ML Service Architecture

The ML service (`ml-api/`) is the core intelligence:
- **FastAPI** serves predictions via REST API
- **URLFeatureExtractor** extracts 40+ features from URLs
- **Random Forest model** trained on phishing vs legitimate URLs
- **Joblib** for model serialization/loading

**Feature Categories:**
- URL structure (length, special chars, IP addresses)
- Content analysis (brand keywords, suspicious patterns) 
- Domain analysis (TLD, subdomains, ports)

## Development Guidelines

### Service Dependencies
1. **ML Service must start first** - other services depend on it
2. **MongoDB required** for the Express server (graceful degradation in dev)
3. **Model training required** before ML service can function

### Testing Strategy
- Use provided test URLs in README.md for validation
- Test legitimate URLs: google.com, github.com, paypal.com  
- Test suspicious patterns: domains with suspicious TLDs, IP addresses, URL shorteners

### Common Troubleshooting
- **"Model file not found"**: Run training scripts in ml-api/
- **CORS errors**: Check CORS_ORIGIN in server/.env matches frontend URL
- **MongoDB connection failed**: Verify MongoDB is running and connection string is correct
- **Port conflicts**: Services use ports 3000, 5000, 8000, 27017

### Database Schema
The Express server uses Mongoose models in `server/models/` to store prediction history and statistics in MongoDB.

## API Endpoints Reference

**Express Backend (`/api/`):**
- `POST /predict` - Main prediction endpoint
- `GET /health` - Service health  
- `GET /history` - Prediction history
- `GET /stats` - Usage statistics

**ML Service (`/`):**
- `POST /predict` - Direct ML prediction
- `GET /health` - ML service status
- `GET /model/info` - Model metadata
- `POST /predict/batch` - Batch predictions (max 50)
- `GET /features/extract?url=...` - Feature extraction only

## Performance Characteristics
- **Response time**: <100ms for URL analysis
- **Model size**: ~500KB trained model file  
- **Memory usage**: ~200MB total for all services
- **Throughput**: ~1000 requests/minute per service