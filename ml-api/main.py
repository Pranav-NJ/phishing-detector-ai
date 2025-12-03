"""
FIXED main.py - Complete Phishing Detection FastAPI Service
FORCES same model path and FRESH feature extractor as predict_model.py
"""

import os
import sys
import joblib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
import uvicorn
from dotenv import load_dotenv

# ============================================================
# CRITICAL: ADD PATH FOR IMPORTS (SAME AS predict_model.py)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(BASE_DIR, "utils")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, UTILS_DIR)

# ============================================================
# CRITICAL: IMPORT FRESH FEATURE EXTRACTOR
# ============================================================
from enhanced_feature_extraction import CompleteFeatureExtractor

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# CRITICAL: FORCE EXACT MODEL PATH (SAME AS predict_model.py)
# ============================================================
MODEL_PATH = os.path.join(BASE_DIR, "models", "phishing_model.pkl")

# Global variables
ml_model = None
feature_extractor = None
model_metadata = None
startup_time = datetime.now()


# ============================================================
# PYDANTIC MODELS (V2 Compatible)
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


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    uptime_seconds: float
    timestamp: str


# ============================================================
# CRITICAL: RULE-BASED PHISHING DETECTION (EXACT MATCH WITH predict_model.py)
# ============================================================

def check_rule_based_phishing(features):
    """Strong rule-based detection for numeric subdomains, brand spoofing, and suspicious TLDs.

    Logic is kept in sync with scripts/predict_model.py so both CLI and API
    make the same rule-based decisions.
    """

    # Rule 1: Subdomain all numbers & long (>15 chars)
    subdomain_numeric_only = features.get("subdomain_is_numeric_only", 0)
    subdomain_length = features.get("subdomain_length", 0)

    logger.debug(f"   subdomain_is_numeric_only: {subdomain_numeric_only}")
    logger.debug(f"   subdomain_length: {subdomain_length}")

    if subdomain_numeric_only > 0.5 and subdomain_length > 15:
        logger.warning(f"🚨 RULE 1: Long numeric subdomain ({subdomain_length} chars)")
        return True, "Long numeric-only subdomain", 0.98

    # Rule 2: High numeric ratio (>20 chars, >70% numeric)
    subdomain_numeric_ratio = features.get("subdomain_numeric_ratio", 0)
    logger.debug(f"   subdomain_numeric_ratio: {subdomain_numeric_ratio}")

    if subdomain_length > 20 and subdomain_numeric_ratio > 0.70:
        logger.warning(
            f"🚨 RULE 2: Very long numeric subdomain ({subdomain_length} chars, {subdomain_numeric_ratio:.0%} numeric)"
        )
        return True, "Very long numeric subdomain", 0.96

    # Rule 3: Generic numeric explosion
    very_long_numeric = features.get("very_long_numeric_subdomain", 0)
    logger.debug(f"   very_long_numeric_subdomain: {very_long_numeric}")

    if very_long_numeric > 0.5:
        logger.warning("🚨 RULE 3: Excessive numeric subdomain")
        return True, "Excessive numeric subdomain", 0.95

    # Rule 4: Brand spoofing with suspicious keywords (strong phishing indicator)
    brand_spoofing = features.get("brand_spoofing_pattern", 0)
    suspicious_kw_count = features.get("suspicious_keyword_count", 0)
    logger.debug(f"   brand_spoofing_pattern: {brand_spoofing}")
    logger.debug(f"   suspicious_keyword_count: {suspicious_kw_count}")

    if brand_spoofing > 0.5 and suspicious_kw_count >= 2:
        logger.warning("🚨 RULE 4: Brand impersonation with suspicious keywords")
        return True, "Brand impersonation with suspicious keywords", 0.90

    # Rule 5: Suspicious TLD with suspicious keywords
    suspicious_tld = features.get("suspicious_tld", 0)
    logger.debug(f"   suspicious_tld: {suspicious_tld}")

    if suspicious_tld > 0.5 and suspicious_kw_count >= 1:
        logger.warning("🚨 RULE 5: Suspicious TLD with suspicious keywords")
        return True, "Suspicious TLD with suspicious keywords", 0.85

    return False, None, 0.0


def analyze_url_warnings(url: str, features: Dict) -> List[str]:
    """Generate specific warnings based on URL analysis."""
    warnings = []
    
    # Critical: Numeric subdomain
    if features.get('subdomain_is_numeric_only', 0) > 0.5:
        length = int(features.get('subdomain_length', 0))
        warnings.append(f"🚨 CRITICAL: Subdomain is {length} characters of pure numbers")
    elif features.get('long_numeric_subdomain', 0) > 0.5:
        warnings.append("⚠️ Long numeric subdomain pattern detected")
    
    # Brand spoofing
    if features.get('brand_spoofing_pattern', 0) > 0:
        warnings.append("⚠️ Brand name in subdomain - possible impersonation")
    
    # Domain warnings
    if features.get('suspicious_tld', 0) > 0:
        warnings.append("⚠️ Suspicious top-level domain (TLD)")
    
    if features.get('is_ip_address', 0) > 0:
        warnings.append("⚠️ Uses IP address instead of domain name")
    
    # Security
    if features.get('is_https', 0) == 0:
        warnings.append("⚠️ Not using secure HTTPS protocol")
    
    # Keywords
    susp_count = int(features.get('suspicious_keyword_count', 0))
    if susp_count >= 2:
        warnings.append(f"⚠️ Contains {susp_count} suspicious keywords")
    
    # Path
    if features.get('suspicious_path', 0) > 0:
        warnings.append("⚠️ Suspicious path pattern (login/verify/account)")
    
    # Length
    if features.get('url_length', 0) > 150:
        warnings.append("⚠️ Unusually long URL")
    
    # URL shortener
    if features.get('is_url_shortener', 0) > 0:
        warnings.append("⚠️ URL shortener detected - hides destination")
    
    return warnings


def calculate_risk_score(confidence: float, prediction: str, features: Dict) -> int:
    """Calculate 0-100 risk score."""
    if prediction == "legitimate":
        return int((1 - confidence) * 100)
    
    # Base score on confidence
    score = int(confidence * 100)
    
    # Boost for dangerous features
    if features.get('subdomain_is_numeric_only', 0) > 0.5:
        score = min(100, score + 20)
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
# MODEL LOADING - FORCES SAME PATH AS predict_model.py
# ============================================================

def load_model():
    """
    Load the ML model with FRESH feature extractor.
    FORCES same model path as predict_model.py
    """
    global ml_model, feature_extractor, model_metadata
    
    logger.info("=" * 70)
    logger.info("🔧 MODEL LOADING CONFIGURATION")
    logger.info("=" * 70)
    logger.info(f"📂 Base Directory: {BASE_DIR}")
    logger.info(f"📂 Utils Directory: {UTILS_DIR}")
    logger.info(f"📂 Model Path: {MODEL_PATH}")
    logger.info("=" * 70)
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ Model file NOT found at:")
        logger.error(f"   {MODEL_PATH}")
        logger.error("")
        logger.error("   Train your model first:")
        logger.error("   python scripts/train_model.py")
        logger.error("")
        return False
    
    try:
        logger.info(f"📂 Loading model from: {MODEL_PATH}")
        
        # Load model data
        model_data = joblib.load(MODEL_PATH)
        
        # Get ML model
        ml_model = model_data['model']
        
        # ============================================================
        # CRITICAL: ALWAYS USE FRESH FEATURE EXTRACTOR
        # DO NOT use the one stored in the model file (it may be old)
        # ============================================================
        logger.info("🔧 Creating FRESH feature extractor...")
        feature_extractor = CompleteFeatureExtractor()
        logger.info("✓ Fresh feature extractor created!")
        
        # Get metadata
        model_metadata = {
            'type': model_data.get('model_type', 'RandomForest'),
            'version': model_data.get('version', '3.1'),
            'training_date': model_data.get('training_date', 'Unknown'),
            'feature_names': model_data.get('feature_names', []),
            'performance': model_data.get('performance', {}),
            'optimal_threshold': model_data.get('optimal_threshold', 0.5)
        }
        
        logger.info("=" * 70)
        logger.info("✅ MODEL LOADED SUCCESSFULLY!")
        logger.info("=" * 70)
        logger.info(f"   Type: {model_metadata['type']}")
        logger.info(f"   Version: {model_metadata['version']}")
        logger.info(f"   Training Date: {model_metadata['training_date']}")
        logger.info(f"   Features: {len(model_metadata['feature_names'])}")
        logger.info(f"   Optimal Threshold: {model_metadata['optimal_threshold']:.3f}")
        
        if model_metadata['performance']:
            perf = model_metadata['performance']
            logger.info(f"   Accuracy: {perf.get('accuracy', 0)*100:.2f}%")
            logger.info(f"   Recall: {perf.get('recall', 0)*100:.2f}%")
            logger.info(f"   F1-Score: {perf.get('f1_score', 0):.4f}")
        
        logger.info("=" * 70)
        
        return True
        
    except Exception as e:
        logger.error("=" * 70)
        logger.error("❌ ERROR LOADING MODEL")
        logger.error("=" * 70)
        logger.error(f"Error: {str(e)}")
        logger.error("")
        logger.error("Stack trace:")
        logger.error(traceback.format_exc())
        logger.error("=" * 70)
        return False


# ============================================================
# LIFESPAN MANAGEMENT
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    # Startup
    logger.info("\n" + "=" * 70)
    logger.info("🚀 STARTING PHISHING DETECTION API")
    logger.info("=" * 70)
    
    success = load_model()
    
    if not success:
        logger.error("=" * 70)
        logger.error("❌ FAILED TO START - MODEL NOT LOADED")
        logger.error("=" * 70)
    else:
        logger.info("=" * 70)
        logger.info("✅ SERVICE STARTED SUCCESSFULLY!")
        logger.info("=" * 70)
    
    yield
    
    # Shutdown
    logger.info("\n🔄 Shutting down service...")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Phishing Detection API",
    description="AI-powered phishing detection with enhanced numeric subdomain detection",
    version="3.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)


# ============================================================
# CORS MIDDLEWARE
# ============================================================

cors_origins = os.getenv('CORS_ORIGINS', '*')
if isinstance(cors_origins, str) and cors_origins != '*':
    try:
        import json
        cors_origins = json.loads(cors_origins)
    except:
        cors_origins = ["*"]
else:
    cors_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DEPENDENCY INJECTION
# ============================================================

def get_model_dependency():
    """Ensure model is loaded."""
    if ml_model is None or feature_extractor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs and ensure model is trained."
        )
    return ml_model


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Phishing Detection API",
        "version": "3.1.0",
        "status": "running",
        "model_loaded": ml_model is not None,
        "model_path": MODEL_PATH,
        "description": "AI-powered phishing detection with rule-based overrides for numeric subdomains",
        "features": [
            "40+ URL features extraction",
            "Rule-based detection for obvious phishing",
            "Random Forest ML model",
            "Real-time analysis"
        ],
        "endpoints": {
            "predict": "POST /predict - Analyze single URL",
            "predict_batch": "POST /predict/batch - Analyze multiple URLs",
            "health": "GET /health - Health check",
            "model_info": "GET /model/info - Model details"
        },
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc"
        },
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_url(request: URLRequest, model=Depends(get_model_dependency)):
    """
    Analyze a single URL for phishing detection.
    SYNCHRONIZED with predict_model.py for consistent results.
    """
    start_time = datetime.now()
    
    try:
        url = request.url
        logger.info("=" * 70)
        logger.info(f"🔍 ANALYZING URL")
        logger.info("=" * 70)
        logger.info(f"URL: {url}")
        logger.info("-" * 70)
        
        # Extract enhanced features using FRESH extractor
        features = feature_extractor.extract_all_features(url)
        
        # DEBUG: Log critical features
        logger.info("📊 EXTRACTED FEATURES:")
        logger.info(f"   subdomain_is_numeric_only: {features.get('subdomain_is_numeric_only', 0)}")
        logger.info(f"   subdomain_length: {features.get('subdomain_length', 0)}")
        logger.info(f"   subdomain_numeric_ratio: {features.get('subdomain_numeric_ratio', 0):.3f}")
        logger.info(f"   long_numeric_subdomain: {features.get('long_numeric_subdomain', 0)}")
        logger.info(f"   very_long_numeric_subdomain: {features.get('very_long_numeric_subdomain', 0)}")
        logger.info(f"   brand_spoofing_pattern: {features.get('brand_spoofing_pattern', 0)}")
        logger.info(f"   suspicious_tld: {features.get('suspicious_tld', 0)}")
        logger.info(f"   is_https: {features.get('is_https', 0)}")
        logger.info("-" * 70)
        
        # Prepare feature vector in correct order
        feature_names = model_metadata['feature_names']
        feature_vector = [features.get(name, 0.0) for name in feature_names]
        
        # Get optimal threshold
        threshold = model_metadata['optimal_threshold']
        
        # ===================================================
        # RULE-BASED CHECK FIRST (EXACT SAME AS predict_model.py)
        # ===================================================
        logger.info("🔍 Checking rule-based detection...")
        is_rule, rule_name, rule_conf = check_rule_based_phishing(features)
        
        if is_rule:
            # Use rule-based detection
            prediction = "phishing"
            confidence = rule_conf
            rule_triggered = rule_name
            
            logger.info("=" * 70)
            logger.info("🚨 RULE-BASED DETECTION TRIGGERED!")
            logger.info("=" * 70)
            logger.info(f"   Rule: {rule_triggered}")
            logger.info(f"   Confidence: {confidence:.3f}")
            logger.info("=" * 70)
            
            # Set probabilities based on rule
            prob_phish = confidence
            prob_legit = 1.0 - confidence
            
        else:
            # Use ML model
            logger.info("🤖 Using ML model prediction...")
            rule_triggered = None
            prediction_proba = model.predict_proba([feature_vector])[0]
            prob_legit = float(prediction_proba[0])
            prob_phish = float(prediction_proba[1])
            
            if prob_phish >= threshold:
                prediction = "phishing"
                confidence = prob_phish
            else:
                prediction = "legitimate"
                confidence = prob_legit
            
            logger.info(f"   ML Prediction: {prediction.upper()}")
            logger.info(f"   Confidence: {confidence:.3f}")
            logger.info(f"   Prob Legitimate: {prob_legit:.3f}")
            logger.info(f"   Prob Phishing: {prob_phish:.3f}")
            logger.info(f"   Threshold: {threshold:.3f}")
        
        # Calculate risk
        risk_score = calculate_risk_score(confidence, prediction, features)
        
        # Determine risk level
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
        
        # Generate warnings
        warnings = analyze_url_warnings(url, features)
        
        # Processing time
        proc_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Create detailed response
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
                    "very_long_numeric_subdomain": bool(features.get('very_long_numeric_subdomain', 0) > 0.5),
                    "subdomain_length": int(features.get('subdomain_length', 0)),
                    "numeric_ratio": round(features.get('subdomain_numeric_ratio', 0), 2),
                    "brand_spoofing": bool(features.get('brand_spoofing_pattern', 0)),
                    "suspicious_tld": bool(features.get('suspicious_tld', 0)),
                    "has_ip": bool(features.get('is_ip_address', 0)),
                    "is_https": bool(features.get('is_https', 0)),
                    "suspicious_keywords": int(features.get('suspicious_keyword_count', 0))
                },
                "url_structure": {
                    "length": int(features.get('url_length', 0)),
                    "domain_length": int(features.get('domain_length', 0)),
                    "subdomain_count": int(features.get('subdomain_count', 0)),
                    "path_depth": int(features.get('path_depth', 0))
                }
            },
            warnings=warnings
        )
        
        logger.info("=" * 70)
        logger.info("✅ FINAL RESULT")
        logger.info("=" * 70)
        logger.info(f"   Prediction: {prediction.upper()}")
        logger.info(f"   Risk Level: {risk_level}")
        logger.info(f"   Risk Score: {risk_score}/100")
        logger.info(f"   Confidence: {confidence:.3f}")
        if rule_triggered:
            logger.info(f"   Rule Triggered: {rule_triggered}")
        logger.info(f"   Processing Time: {proc_time:.2f}ms")
        logger.info("=" * 70 + "\n")
        
        return result
        
    except Exception as e:
        logger.error("=" * 70)
        logger.error("❌ PREDICTION ERROR")
        logger.error("=" * 70)
        logger.error(f"Error: {str(e)}")
        logger.error("")
        logger.error("Stack trace:")
        logger.error(traceback.format_exc())
        logger.error("=" * 70)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch")
async def predict_batch(urls: List[str], model=Depends(get_model_dependency)):
    """Analyze multiple URLs in batch (max 50)."""
    
    if len(urls) > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 50 URLs per batch request"
        )
    
    if len(urls) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="URL list cannot be empty"
        )
    
    results = []
    phishing_count = 0
    legitimate_count = 0
    high_risk_count = 0
    errors = 0
    
    logger.info(f"📊 Batch analysis started: {len(urls)} URLs")
    
    for url in urls:
        try:
            req = URLRequest(url=url)
            result = await predict_url(req, model)
            results.append(result.dict())
            
            if result.prediction == "phishing":
                phishing_count += 1
            else:
                legitimate_count += 1
                
            if result.risk_level in ["HIGH", "CRITICAL"]:
                high_risk_count += 1
                
        except Exception as e:
            logger.error(f"❌ Error processing URL {url}: {str(e)}")
            results.append({
                "url": url,
                "error": str(e),
                "prediction": None,
                "confidence": None
            })
            errors += 1
    
    logger.info(f"✓ Batch complete: {phishing_count} phishing, {legitimate_count} legitimate, {errors} errors")
    
    return {
        "results": results,
        "summary": {
            "total": len(urls),
            "phishing": phishing_count,
            "legitimate": legitimate_count,
            "high_risk": high_risk_count,
            "errors": errors,
            "success_rate": round((len(urls) - errors) / len(urls) * 100, 2) if len(urls) > 0 else 0
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    uptime = (datetime.now() - startup_time).total_seconds()
    
    return HealthResponse(
        status="healthy" if ml_model is not None else "unhealthy",
        model_loaded=ml_model is not None,
        model_version=model_metadata['version'] if model_metadata else None,
        uptime_seconds=round(uptime, 2),
        timestamp=datetime.now().isoformat()
    )


@app.get("/model/info")
async def model_info(model=Depends(get_model_dependency)):
    """Get detailed model information."""
    return {
        "model_type": model_metadata['type'],
        "version": model_metadata['version'],
        "training_date": model_metadata['training_date'],
        "model_path": MODEL_PATH,
        "features": {
            "count": len(model_metadata['feature_names']),
            "names": model_metadata['feature_names'][:20] + ["..."] if len(model_metadata['feature_names']) > 20 else model_metadata['feature_names']
        },
        "optimal_threshold": model_metadata['optimal_threshold'],
        "performance": model_metadata['performance'],
        "timestamp": datetime.now().isoformat()
    }


# ============================================================
# EXCEPTION HANDLERS
# ============================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with consistent format."""
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
    """Handle unexpected exceptions."""
    logger.error(f"❌ Unexpected error: {str(exc)}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8000))
    
    print("\n" + "="*70)
    print("🚀 PHISHING DETECTION API")
    print("="*70)
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Model Path: {MODEL_PATH}")
    print(f"   Docs: http://localhost:{port}/docs")
    print(f"   Health: http://localhost:{port}/health")
    print("="*70 + "\n")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )