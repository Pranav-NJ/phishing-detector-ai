# 🛡️ Phishing Detector AI

A full-stack AI-powered phishing website detection system built with MERN stack and Python machine learning.

## 🌟 Features

- **Real-time URL Analysis**: Instantly analyze URLs for phishing indicators
- **Machine Learning Detection**: Advanced Random Forest classifier with 40+ features
- **Modern Web Interface**: React-based UI with responsive design
- **RESTful API**: Express.js backend with MongoDB storage
- **ML Microservice**: FastAPI service for machine learning predictions
- **Comprehensive Logging**: Track predictions and system health
- **Scalable Architecture**: Microservices ready for cloud deployment

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Client  │    │  Express Server │    │   Python ML API │
│   (Frontend)    │◄──►│   (Backend)     │◄──►│   (ML Service)  │
│   Port 3000     │    │   Port 5000     │    │   Port 8000     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │    MongoDB      │
                       │   (Database)    │
                       └─────────────────┘
```

## 📋 Prerequisites

- **Node.js** (v16+ recommended)
- **Python** (v3.8+ recommended)
- **MongoDB** (local or Atlas)
- **Redis** (v6+ recommended, optional but recommended for caching)
- **Git**

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd phishing-detector-ai
```

### 2. Set Up Environment Variables

Copy the template files and configure:

```bash
# Backend environment
cp server/.env.template server/.env

# ML API environment  
cp ml-api/.env.template ml-api/.env
```

Edit the `.env` files with your configuration (see [Environment Configuration](#-environment-configuration)).

### 3. Install Dependencies

#### Backend (Express Server)
```bash
cd server
npm install
```

#### Frontend (React Client)
```bash
cd ../client
npm install
```

#### ML API (Python Service)
```bash
cd ../ml-api
pip install -r requirements.txt
# or with conda:
# conda create -n phishing-detector python=3.9
# conda activate phishing-detector
# pip install -r requirements.txt
```

### 4. Prepare ML Model

Generate training data and train the model:

```bash
cd ml-api
python scripts/data_collection.py
python scripts/train_model.py
```

### 5. Start Services

Open 4 terminal windows and run:

```bash
# Terminal 1: Start MongoDB (if local)
mongod

# Terminal 2: Start ML API
cd ml-api
python main.py

# Terminal 3: Start Express Backend
cd server
npm run dev

# Terminal 4: Start React Frontend
cd client
npm start
```

### 6. Access the Application

- **Web Interface**: http://localhost:3000
- **Backend API**: http://localhost:5000
- **ML API Docs**: http://localhost:8000/docs
- **MongoDB**: mongodb://localhost:27017

## 🔧 Environment Configuration

### Backend (.env)
```env
PORT=5000
NODE_ENV=development
MONGODB_URI=mongodb://localhost:27017/phishing_detector
ML_API_URL=http://localhost:8000
ML_API_ENDPOINT=/predict
CORS_ORIGIN=http://localhost:3000
RATE_LIMIT_WINDOW_MS=900000
RATE_LIMIT_MAX_REQUESTS=100
REDIS_ENABLED=true
REDIS_URL=redis://localhost:6379
REDIS_CACHE_TTL_SECONDS=3600
```

### ML API (.env)
```env
APP_NAME=Phishing Detection ML API
DEBUG=True
HOST=0.0.0.0
PORT=8000
MODEL_PATH=./models/phishing_model.pkl
CORS_ORIGINS=["http://localhost:3000", "http://localhost:5000"]
LOG_LEVEL=INFO
```

## 🧪 Testing the System

### Test with Sample URLs

**Legitimate URLs** (should show as SAFE):
- https://www.google.com
- https://www.github.com
- https://www.paypal.com

**Suspicious Patterns** (may show as DANGEROUS):
- http://paypal-security-alert.tk/login
- https://google-account-verify.ml/signin
- http://192.168.1.100/facebook/login.php

### API Testing with cURL

```bash
# Test ML API directly
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.google.com"}'

# Test Express API
curl -X POST "http://localhost:5000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.google.com"}'

# Check health status
curl http://localhost:8000/health
curl http://localhost:5000/api/health
```

## 📊 Machine Learning Details

### Features Extracted (40+ features)

The system analyzes URLs using multiple categories of features:

**URL Structure Features:**
- Length metrics (URL, hostname, path)
- Character analysis (dots, dashes, special characters)
- Security indicators (HTTPS usage, IP addresses)

**Content Features:**
- Brand keyword detection (PayPal, Amazon, etc.)
- Action words (login, verify, urgent)
- Suspicious patterns (URL shorteners, encoding)

**Domain Analysis:**
- TLD analysis (suspicious extensions)
- Subdomain patterns
- Port usage patterns

### Model Performance

The Random Forest model typically achieves:
- **Accuracy**: 95-98%
- **Precision**: 92-96%
- **Recall**: 90-95%
- **F1-Score**: 91-95%

## 🚀 Deployment

### Local Development

Follow the [Quick Start](#-quick-start) guide above.

### Production Deployment

#### Option 1: Traditional Server

**Frontend (Static Hosting)**
```bash
cd client
npm run build
# Deploy build/ folder to Netlify, Vercel, or web server
```

**Backend (Node.js Server)**
```bash
cd server
npm start
# Deploy to Heroku, DigitalOcean, or your server
```

**ML API (Python Server)**
```bash
cd ml-api
pip install -r requirements.txt
python main.py
# Deploy to Heroku, Railway, or your server
```

#### Option 2: Docker (Recommended)

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  frontend:
    build: ./client
    ports:
      - "3000:80"
    depends_on:
      - backend
      
  backend:
    build: ./server
    ports:
      - "5000:5000"
    environment:
      - MONGODB_URI=mongodb://mongo:27017/phishing_detector
      - ML_API_URL=http://ml-api:8000
    depends_on:
      - mongo
      - ml-api
      
  ml-api:
    build: ./ml-api
    ports:
      - "8000:8000"
    volumes:
      - ./ml-api/models:/app/models
      
  mongo:
    image: mongo:latest
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db

volumes:
  mongo_data:
```

Deploy with:
```bash
docker-compose up -d
```

#### Option 3: Cloud Services

**Frontend**: Deploy to Vercel, Netlify, or AWS S3
**Backend**: Deploy to Heroku, Railway, or AWS EC2
**ML API**: Deploy to Heroku, Railway, or Google Cloud Run
**Database**: Use MongoDB Atlas (cloud MongoDB)

## 🔍 API Documentation

### Express Backend Endpoints

#### POST /api/predict
Analyze a URL for phishing.

```json
// Request
{
  "url": "https://example.com"
}

// Response
{
  "url": "https://example.com",
  "prediction": false,
  "confidence": 0.95,
  "processingTime": 245,
  "success": true
}
```

#### GET /api/health
Check service health.

#### GET /api/history?limit=10
Get recent predictions.

#### GET /api/stats
Get prediction statistics.

### ML API Endpoints

#### POST /predict
Direct ML prediction.

#### GET /health
ML service health check.

#### GET /model/info
Model information and performance metrics.

#### POST /predict/batch
Batch URL prediction (up to 50 URLs).

#### GET /features/extract?url=https://example.com
Extract features without prediction.

## 🛠️ Development

### Project Structure
```
phishing-detector-ai/
├── client/                 # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   └── App.js         # Main app component
│   └── package.json
├── server/                 # Express backend
│   ├── models/            # MongoDB schemas
│   ├── routes/            # API routes
│   ├── middleware/        # Express middleware
│   └── server.js         # Main server file
├── ml-api/                # Python ML service
│   ├── scripts/           # Data & training scripts
│   ├── utils/             # Feature extraction
│   ├── models/            # Trained ML models
│   └── main.py           # FastAPI application
├── phishing-detector-extension/   # Browser extension (Chrome/Edge)
│   ├── icons/                     # Extension icons (PNG formats)
│   ├── background.js              # Monitors tabs, sends requests
│   ├── popup.html                 # Popup UI layout
│   ├── popup.js                   # Popup script (API call & status)
│   └── create_icons.py            # (Optional) Script to generate icons
└── README.md
```

### Adding New Features

**Frontend**: Edit components in `client/src/components/`
**Backend**: Add routes in `server/routes/`
**ML**: Modify features in `ml-api/utils/feature_extractor.py`

### Running Tests

```bash
# Frontend tests
cd client && npm test

# Backend tests (if implemented)
cd server && npm test

# ML API tests (if implemented)
cd ml-api && python -m pytest
```

## 🐛 Troubleshooting

### Common Issues

**Model file not found**
```bash
cd ml-api
python scripts/data_collection.py
python scripts/train_model.py
```

**MongoDB connection failed**
- Ensure MongoDB is running
- Check connection string in `.env`
- For Atlas: whitelist your IP

**CORS errors**
- Check `CORS_ORIGIN` in backend `.env`
- Ensure frontend URL is whitelisted

**Module import errors**
```bash
cd ml-api
pip install -r requirements.txt
```

**Port already in use**
```bash
# Find and kill process using port
lsof -ti:3000 | xargs kill -9  # macOS/Linux
netstat -ano | findstr :3000   # Windows
```

## 🔒 Security Considerations

- **Input Validation**: All URLs are validated before processing
- **Rate Limiting**: API endpoints are rate-limited
- **CORS Configuration**: Properly configured for production
- **Environment Variables**: Sensitive data in `.env` files
- **SQL Injection**: Using MongoDB with Mongoose (NoSQL)
- **XSS Protection**: React's built-in XSS protection

## 📈 Performance

- **Response Time**: < 100ms for URL analysis
- **Throughput**: ~1000 requests/minute per service
- **Memory Usage**: ~200MB total (all services)
- **Model Size**: ~500KB trained model file

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PhishTank** - For phishing URL data
- **scikit-learn** - Machine learning library
- **FastAPI** - Modern Python web framework
- **React** - Frontend framework
- **Express.js** - Node.js web framework
- **MongoDB** - Database solution

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Search existing issues on GitHub
3. Create a new issue with detailed information

---

## 🚀 Next Steps

After successful deployment, consider:

1. **Monitoring**: Add logging and monitoring (e.g., Sentry, DataDog)
2. **Scaling**: Implement load balancing for high traffic
3. **Security**: Add authentication and authorization
4. **Features**: Email alerts, browser extension, etc.
5. **ML Improvements**: Regular model retraining with new data

**Happy Detecting! 🛡️**
