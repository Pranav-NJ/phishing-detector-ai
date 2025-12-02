"""
FIXED: FastAPI Phishing Detection Service
Properly handles numeric subdomain phishing with rule-based overrides
"""

import os
import joblib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
ml_model = None
feature_extractor = None
model_metadata = None
startup_time = datetime.now()


# ============================================================
# PYDANTIC MODELS
# ============================================================

class URLRequest(BaseModel):
    url: str
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('URL cannot be empty')
        return v.strip()


class PredictionResponse(BaseModel):
    url: str
    prediction: str
    confidence: float
    risk_level: str
    risk_score: int
    processing_time_ms: float
    model_version: str
    threshold_used: float
    timestamp: str
    details: Dict
    warnings: List[str]
    rule_triggered: Optional[str] = None


# ============================================================
# CRITICAL: RULE-BASED OVERRIDES FOR EXTREME PHISHING PATTERNS
# ============================================================

def check_rule_based_phishing(features: Dict, url: str) -> Optional[Dict[str, Any]]:
    """
    🚨 CRITICAL FUNCTION 🚨
    Detect extreme phishing patterns that should ALWAYS be flagged.
    This overrides ML model predictions for obvious phishing.
    """
    
    # Rule 1: Very long numeric-only subdomain (>15 chars)
    # Example: http://00000000883838383992929292222.ratingandreviews.in
    if features.get('subdomain_is_numeric_only', 0) > 0.5:
        subdomain_length = features.get('subdomain_length', 0)
        if subdomain_length > 15:
            logger.warning(f"🚨 RULE 1: Long numeric subdomain ({subdomain_length} chars)")
            return {
                'prediction': 'phishing',
                'confidence': 0.98,
                'rule': 'Long numeric-only subdomain (>15 chars)',
                'reason': f'Subdomain is {subdomain_length} characters of pure numbers'
            }
    
    # Rule 2: Extremely long subdomain with high numeric ratio
    subdomain_length = features.get('subdomain_length', 0)
    numeric_ratio = features.get('subdomain_numeric_ratio', 0)
    if subdomain_length > 20 and numeric_ratio > 0.7:
        logger.warning(f"🚨 RULE 2: Very long subdomain with {numeric_ratio:.0%} numbers")
        return {
            'prediction': 'phishing',
            'confidence': 0.96,
            'rule': 'Very long subdomain with high numeric content',
            'reason': f'{subdomain_length} char subdomain, {numeric_ratio:.0%} numeric'
        }
    
    # Rule 3: Long numeric subdomain feature triggered
    if features.get('very_long_numeric_subdomain', 0) > 0.5:
        logger.warning("🚨 RULE 3: Very long numeric subdomain feature")
        return {
            'prediction': 'phishing',
            'confidence': 0.95,
            'rule': 'Very long numeric subdomain pattern',
            'reason': 'Matches known phishing subdomain pattern'
        }
    
    # Rule 4: Multiple strong indicators (4+)
    suspicious_score = 0
    indicators = []
    
    if features.get('subdomain_is_numeric_only', 0) > 0.5:
        suspicious_score += 3
        indicators.append("numeric-only subdomain")
    if features.get('long_numeric_subdomain', 0) > 0.5:
        suspicious_score += 2
        indicators.append("long numeric subdomain")
    if features.get('brand_spoofing_pattern', 0) > 0:
        suspicious_score += 2
        indicators.append("brand spoofing")
    if features.get('suspicious_tld', 0) > 0:
        suspicious_score += 1
        indicators.append("suspicious TLD")
    if features.get('is_ip_address', 0) > 0:
        suspicious_score += 2
        indicators.append("IP address")
    if features.get('is_https', 0) == 0:
        suspicious_score += 1
        indicators.append("no HTTPS")
    
    if suspicious_score >= 4:
        logger.warning(f"🚨 RULE 4: Multiple indicators (score: {suspicious_score})")
        return {
            'prediction': 'phishing',
            'confidence': min(0.95, 0.70 + (suspicious_score * 0.05)),
            'rule': 'Multiple suspicious indicators',
            'reason': f'Found {len(indicators)} indicators: {", ".join(indicators)}'
        }
    
    return None


def analyze_url_warnings(url: str, features: Dict) -> List[str]:
    """Generate specific warnings based on URL analysis."""
    warnings = []
    
    # Numeric subdomain warnings
    if features.get('subdomain_is_numeric_only', 0) > 0.5:
        length = int(features.get('subdomain_length', 0))
        warnings.append(f"🚨 Subdomain is {length} characters of pure numbers - HIGHLY SUSPICIOUS")
    elif features.get('long_numeric_subdomain', 0) > 0.5:
        warnings.append("⚠️ Long numeric subdomain pattern detected")
    
    # Brand spoofing
    if features.get('brand_spoofing_pattern', 0) > 0:
        warnings.append("⚠️ Brand name in subdomain - possible impersonation")
    
    # TLD and domain warnings
    if features.get('suspicious_tld', 0) > 0:
        warnings.append("⚠️ Suspicious top-level domain (TLD)")
    
    if features.get('is_ip_address', 0) > 0:
        warnings.append("⚠️ Uses IP address instead of domain name")
    
    # Security warnings
    if features.get('is_https', 0) == 0:
        warnings.append("⚠️ Not using secure HTTPS protocol")
    
    # Keyword warnings
    if features.get('suspicious_keyword_count', 0) >= 2:
        warnings.append("⚠️ Multiple suspicious keywords detected")
    
    # Path warnings
    if features.get('suspicious_path', 0) > 0:
        warnings.append("⚠️ Suspicious path pattern (login/verify/account)")
    
    # URL characteristics
    if features.get('url_length', 0) > 150:
        warnings.append("⚠️ Unusually long URL")
    
    if features.get('is_url_shortener', 0) > 0:
        warnings.append("⚠️ URL shortener - hides true destination")
    
    return warnings


def calculate_risk_score(confidence: float, prediction: str, features: Dict) -> int:
    """Calculate 0-100 risk score."""
    if prediction == "legitimate":
        return int((1 - confidence) * 100)
    
    # Base score on confidence
    score = int(confidence * 100)
    
    # Boost for dangerous features
    if features.get('subdomain_is_numeric_only', 0) > 0.5:
        score = min(100, score + 20)  # Big boost for numeric subdomain
    if features.get('very_long_numeric_subdomain', 0) > 0.5:
        score = min(100, score + 15)
    if features.get('brand_spoofing_pattern', 0) > 0:
        score = min(100, score + 10)
    if features.get('is_ip_address', 0) > 0:
        score = min(100, score + 8)
    if features.get('suspicious_tld', 0) > 0:
        score = min(100, score + 5)
    
    return min(100, score)


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():
    """Load the trained model."""
    global ml_model, feature_extractor, model_metadata
    
    possible_paths = [
        'phishing_model.pkl',
        'models/phishing_model.pkl',
        '../models/phishing_model.pkl'
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        logger.error("❌ Model file not found!")
        return False
    
    try:
        logger.info(f"📂 Loading model from: {model_path}")
        
        model_data = joblib.load(model_path)
        
        ml_model = model_data['model']
        feature_extractor = model_data.get('feature_extractor')
        
        model_metadata = {
            'type': model_data.get('model_type', 'RandomForest'),
            'version': model_data.get('version', '3.1'),
            'training_date': model_data.get('training_date', 'Unknown'),
            'feature_names': model_data.get('feature_names', []),
            'performance': model_data.get('performance', {}),
            'optimal_threshold': model_data.get('optimal_threshold', 0.5)
        }
        
        logger.info("✓ Model loaded successfully!")
        logger.info(f"   Version: {model_metadata['version']}")
        logger.info(f"   Threshold: {model_metadata['optimal_threshold']:.3f}")
        
        if model_metadata['performance']:
            perf = model_metadata['performance']
            logger.info(f"   Accuracy: {perf.get('accuracy', 0)*100:.2f}%")
            logger.info(f"   Recall: {perf.get('recall', 0)*100:.2f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        return False


# ============================================================
# FASTAPI APP
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    logger.info("=" * 70)
    logger.info("🚀 Starting FIXED Phishing Detection API")
    logger.info("=" * 70)
    
    success = load_model()
    if not success:
        logger.error("❌ Failed to load model")
    else:
        logger.info("✅ Service ready!")
    
    logger.info("=" * 70)
    
    yield
    
    logger.info("🔄 Shutting down...")


app = FastAPI(
    title="Fixed Phishing Detection API",
    description="AI-powered phishing detection with rule-based overrides",
    version="3.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Fixed Phishing Detection API",
        "version": "3.1.0",
        "status": "running",
        "model_loaded": ml_model is not None,
        "features": "Numeric subdomain detection + rule-based overrides",
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_url(request: URLRequest):
    """
    Analyze URL for phishing.
    Uses ML model + rule-based overrides for extreme cases.
    """
    if ml_model is None or feature_extractor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    start_time = datetime.now()
    url = request.url
    
    try:
        logger.info(f"🔍 Analyzing: {url}")
        
        # Extract features
        features = feature_extractor.extract_all_features(url)
        
        # 🚨 CRITICAL: Check rule-based overrides FIRST
        rule_result = check_rule_based_phishing(features, url)
        
        threshold = model_metadata['optimal_threshold']
        rule_triggered = None
        
        if rule_result:
            # Use rule-based detection
            prediction = rule_result['prediction']
            confidence = rule_result['confidence']
            rule_triggered = rule_result['rule']
            
            logger.warning(f"⚠️ RULE TRIGGERED: {rule_triggered}")
            logger.warning(f"   Reason: {rule_result['reason']}")
            
            # Set probabilities based on rule
            prob_phish = confidence
            prob_legit = 1.0 - confidence
        else:
            # Use ML model
            feature_names = model_metadata['feature_names']
            feature_vector = [features.get(name, 0.0) for name in feature_names]
            
            prediction_proba = ml_model.predict_proba([feature_vector])[0]
            prob_legit = float(prediction_proba[0])
            prob_phish = float(prediction_proba[1])
            
            if prob_phish >= threshold:
                prediction = "phishing"
                confidence = prob_phish
            else:
                prediction = "legitimate"
                confidence = prob_legit
        
        # Calculate risk
        risk_score = calculate_risk_score(confidence, prediction, features)
        
        # Risk level
        if prediction == "phishing":
            if risk_score >= 90:
                risk_level = "CRITICAL"
            elif risk_score >= 75:
                risk_level = "HIGH"
            elif risk_score >= 60:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
        else:
            risk_level = "SAFE" if risk_score <= 20 else "LOW"
        
        # Warnings
        warnings = analyze_url_warnings(url, features)
        
        # Processing time
        proc_time = (datetime.now() - start_time).total_seconds() * 1000
        
        result = PredictionResponse(
            url=url,
            prediction=prediction,
            confidence=round(confidence, 4),
            risk_level=risk_level,
            risk_score=risk_score,
            processing_time_ms=round(proc_time, 2),
            model_version=model_metadata['version'],
            threshold_used=round(threshold, 3),
            timestamp=datetime.now().isoformat(),
            rule_triggered=rule_triggered,
            details={
                "probabilities": {
                    "legitimate": round(prob_legit, 4),
                    "phishing": round(prob_phish, 4)
                },
                "key_indicators": {
                    "numeric_subdomain": bool(features.get('subdomain_is_numeric_only', 0) > 0.5),
                    "long_numeric_subdomain": bool(features.get('long_numeric_subdomain', 0) > 0.5),
                    "subdomain_length": int(features.get('subdomain_length', 0)),
                    "numeric_ratio": round(features.get('subdomain_numeric_ratio', 0), 2),
                    "brand_spoofing": bool(features.get('brand_spoofing_pattern', 0)),
                    "suspicious_tld": bool(features.get('suspicious_tld', 0)),
                    "is_https": bool(features.get('is_https', 0))
                }
            },
            warnings=warnings
        )
        
        logger.info(f"✓ {prediction.upper()} | Risk: {risk_level} ({risk_score}) | Conf: {confidence:.3f}")
        if rule_triggered:
            logger.info(f"   Rule: {rule_triggered}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/health")
async def health():
    """Health check."""
    uptime = (datetime.now() - startup_time).total_seconds()
    return {
        "status": "healthy" if ml_model is not None else "unhealthy",
        "model_loaded": ml_model is not None,
        "uptime_seconds": round(uptime, 2),
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    port = int(os.getenv('PORT', 8000))
    
    print("\n" + "="*70)
    print("🚀 FIXED PHISHING DETECTION API")
    print("="*70)
    print(f"   Port: {port}")
    print(f"   Docs: http://localhost:{port}/docs")
    print("="*70 + "\n")
    
    uvicorn.run(
        "api_service:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )