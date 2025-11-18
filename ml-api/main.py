"""
FastAPI ML Service for Phishing Detection

This service provides a REST API for phishing URL detection using a trained
Random Forest model. It extracts features from URLs in real-time and returns
predictions with confidence scores.
"""

import os
import sys
import joblib
import logging
from datetime import datetime
from typing import Dict, List, Optional
import traceback

from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl, validator
import uvicorn
from dotenv import load_dotenv

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from feature_extractor import URLFeatureExtractor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for model and feature extractor
ml_model = None
feature_extractor = None
model_metadata = None

# Pydantic models for API
class URLRequest(BaseModel):
    url: str
    
    @validator('url')
    def validate_url(cls, v):
        if not v or not v.strip():
            raise ValueError('URL cannot be empty')
        
        # Basic URL validation
        url = v.strip()
        if not (url.startswith('http://') or url.startswith('https://')):
            url = 'http://' + url
        
        return url

class PredictionResponse(BaseModel):
    url: str
    prediction: str  # 'phishing' or 'legitimate'
    confidence: float
    processing_time_ms: float
    model_version: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str]
    uptime_seconds: float
    timestamp: str

class ModelInfoResponse(BaseModel):
    model_type: str
    version: str
    training_date: str
    feature_count: int
    performance_metrics: Dict

# Initialize FastAPI app
app = FastAPI(
    title="Phishing Detection ML API",
    description="AI-powered phishing website detection using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global variable to track startup time
startup_time = datetime.now()

# Configure CORS
cors_origins = os.getenv('CORS_ORIGINS', '["*"]')
if isinstance(cors_origins, str):
    try:
        import json
        cors_origins = json.loads(cors_origins)
    except:
        cors_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_model():
    """Load the trained ML model and feature extractor."""
    global ml_model, feature_extractor, model_metadata
    
    model_path = os.getenv('MODEL_PATH', './models/phishing_model.pkl')
    
    try:
        logger.info(f"Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            return False
        
        # Load model data
        model_data = joblib.load(model_path)
        
        # Extract components
        ml_model = model_data['model']
        model_metadata = {
            'type': model_data.get('model_type', 'Unknown'),
            'version': model_data.get('version', '1.0'),
            'training_date': model_data.get('training_date', 'Unknown'),
            'feature_names': model_data.get('feature_names', []),
            'performance': model_data.get('performance', {})
        }
        
        # Initialize feature extractor
        feature_extractor = URLFeatureExtractor()
        
        logger.info("Model loaded successfully!")
        logger.info(f"Model type: {model_metadata['type']}")
        logger.info(f"Features: {len(model_metadata['feature_names'])}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize the service on startup."""
    logger.info("🚀 Starting Phishing Detection ML API")
    
    success = load_model()
    if not success:
        logger.error("❌ Failed to load model. Service will not function properly.")
    else:
        logger.info("✅ ML API service started successfully!")

def get_model_dependency():
    """Dependency to ensure model is loaded."""
    if ml_model is None or feature_extractor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model not loaded. Please check server logs."
        )
    return ml_model, feature_extractor

@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Phishing Detection ML API",
        "version": "1.0.0",
        "description": "AI-powered phishing website detection",
        "status": "running",
        "model_loaded": ml_model is not None,
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health",
            "model_info": "GET /model/info"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_url(request: URLRequest, dependencies=Depends(get_model_dependency)):
    """
    Predict whether a URL is phishing or legitimate.
    
    This is the main endpoint that:
    1. Extracts features from the provided URL
    2. Uses the trained model to make a prediction
    3. Returns the result with confidence score
    """
    start_time = datetime.now()
    model, extractor = dependencies
    
    try:
        url = request.url
        logger.info(f"Processing prediction request for: {url}")
        
        # Extract features from URL
        features = extractor.extract_all_features(url)
        
        # Convert to the format expected by the model
        feature_names = model_metadata['feature_names']
        feature_vector = []
        
        for feature_name in feature_names:
            feature_vector.append(features.get(feature_name, 0))
        
        # Make prediction
        prediction_proba = model.predict_proba([feature_vector])[0]
        prediction_class = model.predict([feature_vector])[0]
        
        # Get confidence score (probability of the predicted class)
        confidence = float(max(prediction_proba))
        
        # Convert prediction to string
        prediction_str = "phishing" if prediction_class == 1 else "legitimate"
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        result = PredictionResponse(
            url=url,
            prediction=prediction_str,
            confidence=confidence,
            processing_time_ms=processing_time,
            model_version=model_metadata['version'],
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Prediction completed: {prediction_str} (confidence: {confidence:.3f})")
        return result
        
    except Exception as e:
        logger.error(f"Error processing prediction: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing URL: {str(e)}"
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring."""
    uptime = (datetime.now() - startup_time).total_seconds()
    
    return HealthResponse(
        status="healthy" if ml_model is not None else "unhealthy",
        model_loaded=ml_model is not None,
        model_version=model_metadata['version'] if model_metadata else None,
        uptime_seconds=uptime,
        timestamp=datetime.now().isoformat()
    )

@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info(dependencies=Depends(get_model_dependency)):
    """Get information about the loaded model."""
    return ModelInfoResponse(
        model_type=model_metadata['type'],
        version=model_metadata['version'],
        training_date=model_metadata['training_date'],
        feature_count=len(model_metadata['feature_names']),
        performance_metrics=model_metadata['performance']
    )

@app.post("/predict/batch")
async def predict_batch(urls: List[str], dependencies=Depends(get_model_dependency)):
    """
    Predict multiple URLs at once.
    
    Useful for batch processing or testing multiple URLs.
    """
    model, extractor = dependencies
    
    if len(urls) > 50:  # Limit batch size
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 50 URLs per batch request"
        )
    
    results = []
    
    for url in urls:
        try:
            # Create individual request and process
            url_request = URLRequest(url=url)
            result = await predict_url(url_request, dependencies)
            results.append(result.dict())
            
        except Exception as e:
            results.append({
                "url": url,
                "error": str(e),
                "prediction": None,
                "confidence": 0.0
            })
    
    return {
        "results": results,
        "total_processed": len(results),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/features/extract")
async def extract_features_endpoint(url: str):
    """
    Extract and return features from a URL without making a prediction.
    
    Useful for debugging and understanding what features the model uses.
    """
    try:
        if not feature_extractor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Feature extractor not available"
            )
        
        features = feature_extractor.extract_all_features(url)
        
        return {
            "url": url,
            "features": features,
            "feature_count": len(features),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error extracting features: {str(e)}"
        )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler for unexpected errors."""
    logger.error(f"Unexpected error: {str(exc)}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if os.getenv('DEBUG', 'False').lower() == 'true' else "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    # Configuration
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8000))
    debug = os.getenv('DEBUG', 'True').lower() == 'true'
    workers = int(os.getenv('WORKERS', 1))
    
    print("🚀 Starting Phishing Detection ML API")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Debug: {debug}")
    print(f"   Workers: {workers}")
    
    # Run the server
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        workers=1 if debug else workers,
        log_level="info"
    )