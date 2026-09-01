const express = require('express');
const axios = require('axios');
const Joi = require('joi');
const Prediction = require('../models/Prediction');
const { getRedisClient, isRedisEnabled } = require('../utils/redisClient');

const router = express.Router();

function normalizeUrlForAnalysis(rawUrl) {
  let urlStr = rawUrl.trim();
  // Simply ensure it starts with http:// or https:// without formatting away the user's explicit path
  if (!urlStr.startsWith('http://') && !urlStr.startsWith('https://')) {
    urlStr = 'https://' + urlStr;
  }
  return urlStr;
}

// Validation schema for URL input
const urlSchema = Joi.object({
  url: Joi.string()
    .uri({ scheme: ['http', 'https'] })
    .required()
    .max(2048)
    .messages({
      'string.uri': 'Please provide a valid URL with http:// or https://',
      'string.max': 'URL is too long (maximum 2048 characters)',
      'any.required': 'URL is required'
    })
});

// POST /api/predict - Analyze URL for phishing
router.post('/predict', async (req, res) => {
  const startTime = Date.now();

  try {
    // Validate input
    const { error, value } = urlSchema.validate(req.body);
    if (error) {
      return res.status(400).json({
        error: error.details[0].message,
        success: false
      });
    }

    const { url } = value;
    const originalUrl = url.trim();
    const canonicalUrl = normalizeUrlForAnalysis(originalUrl);

    // Attempt to serve cached result first
    const cacheTtlSeconds = parseInt(process.env.REDIS_CACHE_TTL_SECONDS, 10) || 3600;
    const cacheKey = `prediction:${canonicalUrl.toLowerCase()}`;
    if (isRedisEnabled()) {
      try {
        const redisClient = getRedisClient();
        if (redisClient) {
          const cached = await redisClient.get(cacheKey);
          if (cached) {
            const cachedResponse = JSON.parse(cached);
            return res.json({
              ...cachedResponse,
              processingTime: Date.now() - startTime,
              cache: 'hit',
            });
          }
        } else {
          // Redis enabled in config but client not initialized — skip caching
          console.warn('⚠️ Redis enabled but client unavailable; skipping cache read');
        }
      } catch (cacheReadError) {
        console.error('Redis cache read error:', cacheReadError && cacheReadError.message ? cacheReadError.message : cacheReadError);
      }
    }

    // Call ML API
    const mlApiUrl = `${process.env.ML_API_URL}${process.env.ML_API_ENDPOINT || '/predict'}`;

    let mlResponse;
    try {
      mlResponse = await axios.post(
        mlApiUrl,
        { url: canonicalUrl },
        {
          timeout: 30000,
          headers: {
            'Content-Type': 'application/json'
          }
        }
      );
    } catch (mlError) {
      console.error('ML API Error:', mlError.message);

      // If ML API is down, attempt CLI fallback using the project's Python script
      try {
        const { execFile } = require('child_process');
        const pyPath = require('path').join(__dirname, '..', '..', 'ml-api', 'scripts', 'predict_model.py');

        const cliResult = await new Promise((resolve, reject) => {
          execFile('python', [pyPath, canonicalUrl], { timeout: 15000 }, (err, stdout, stderr) => {
            if (err) return reject(err);
            return resolve(stdout);
          });
        });

        // Parse CLI output to extract prediction and confidence
        const out = String(cliResult);
        const isPhishing = /PHISHING DETECTED!/i.test(out);
        const confMatch = out.match(/Confidence:\s*([0-9]+\.?[0-9]*)%/i);
        const confidence = confMatch ? parseFloat(confMatch[1]) / 100.0 : 0.5;

        mlResponse = {
          data: {
            prediction: isPhishing ? 'phishing' : 'legitimate',
            confidence: confidence,
            details: { source: 'cli-fallback' },
            risk_level: 'UNKNOWN',
            risk_score: 0,
            threshold_used: null,
            warnings: [],
            rule_triggered: null
          }
        };

      } catch (cliError) {
        console.error('CLI fallback failed:', cliError && cliError.message ? cliError.message : cliError);

        if (mlError.code === 'ECONNREFUSED') {
          return res.status(503).json({
            error: 'ML service is currently unavailable. Please try again later.',
            success: false
          });
        }

        return res.status(500).json({
          error: 'Error processing your request. Please try again.',
          success: false
        });
      }
    }

    // Extract prediction and confidence
    const mlData = mlResponse.data;
    const prediction = mlData.prediction || 'legitimate';
    const confidence = mlData.confidence || 0.5;
    const processingTime = Date.now() - startTime;

    // Save prediction to DB
    try {
      const predictionRecord = new Prediction({
        url: originalUrl,
        prediction: prediction === 'phishing',
        confidence: confidence || 0.5,
        userAgent: req.get('User-Agent') || '',
        ipAddress: req.ip || req.connection.remoteAddress || '',
        processingTimeMs: processingTime
      });

      await predictionRecord.save();
    } catch (dbError) {
      console.error('Database save error:', dbError);
      // Continue even if DB save fails
    }

    // Use the ML API response natively without Express-level overrides
    const predictionPayload = {
      url: originalUrl,
      canonical_url: canonicalUrl,
      prediction: prediction === 'phishing', // Use ML model directly
      confidence: confidence || 0.5,
      processingTime,
      success: true,
      details: mlData.details || {},
      risk_level: mlData.risk_level || 'UNKNOWN',
      risk_score: mlData.risk_score || 0,
      threshold_used: mlData.threshold_used,
      warnings: mlData.warnings || [],
      rule_triggered: mlData.rule_triggered,
      cache: 'miss'
    };

    if (isRedisEnabled()) {
      try {
        const redisClient = getRedisClient();
        if (redisClient) {
          await redisClient.set(cacheKey, JSON.stringify(predictionPayload), {
            EX: cacheTtlSeconds,
          });
        } else {
          console.warn('⚠️ Redis enabled but client unavailable; skipping cache write');
        }
      } catch (cacheWriteError) {
        console.error('Redis cache write error:', cacheWriteError && cacheWriteError.message ? cacheWriteError.message : cacheWriteError);
      }
    }

    return res.json(predictionPayload);
  } catch (error) {
    console.error('Prediction endpoint error:', error);
    res.status(500).json({
      error: 'Internal server error',
      success: false
    });
  }
});

// GET /api/history - Recent prediction history
router.get('/history', async (req, res) => {
  try {
    const limit = parseInt(req.query.limit) || 10;
    const maxLimit = 100;
    const finalLimit = Math.min(limit, maxLimit);

    // Skip database operations in development without MongoDB
    if (process.env.NODE_ENV === 'development' && !process.env.MONGODB_URI) {
      return res.json({
        predictions: [],
        count: 0,
        success: true,
        message: 'History not available in development mode without database'
      });
    }

    const predictions = await Prediction.getRecent(finalLimit);

    res.json({
      predictions,
      count: predictions.length,
      success: true
    });
  } catch (error) {
    console.error('History endpoint error:', error);
    res.status(500).json({
      error: 'Failed to retrieve prediction history',
      success: false
    });
  }
});

// GET /api/stats - Prediction statistics
router.get('/stats', async (req, res) => {
  try {
    // Skip database operations in development without MongoDB
    if (process.env.NODE_ENV === 'development' && !process.env.MONGODB_URI) {
      return res.json({
        totalPredictions: 0,
        phishingDetected: 0,
        legitimateDetected: 0,
        averageConfidence: 0,
        success: true,
        message: 'Statistics not available in development mode without database'
      });
    }

    const stats = await Prediction.getStats();

    res.json({
      ...stats,
      success: true
    });
  } catch (error) {
    console.error('Stats endpoint error:', error);
    res.status(500).json({
      error: 'Failed to retrieve statistics',
      success: false
    });
  }
});

// GET /api/health - Health check
router.get('/health', async (req, res) => {
  try {
    // ML API health check
    let mlApiHealthy = false;
    try {
      const mlApiUrl = `${process.env.ML_API_URL}/health`;
      const healthResponse = await axios.get(mlApiUrl, { timeout: 5000 });
      mlApiHealthy = healthResponse.status === 200;
    } catch (error) {
      console.log('ML API health check failed:', error.message);
    }

    // Database health check
    let dbHealthy = false;
    try {
      await Prediction.countDocuments().limit(1);
      dbHealthy = true;
    } catch (error) {
      console.log('Database health check failed:', error.message);
    }

    // Redis cache health check
    const redisHealthy = isRedisEnabled();

    const overall = mlApiHealthy && dbHealthy;

    res.status(overall ? 200 : 503).json({
      status: overall ? 'healthy' : 'unhealthy',
      services: {
        mlApi: mlApiHealthy ? 'healthy' : 'unhealthy',
        database: dbHealthy ? 'healthy' : 'unhealthy',
        redisCache: redisHealthy ? 'healthy' : 'disabled'
      },
      timestamp: new Date().toISOString(),
      success: true
    });
  } catch (error) {
    res.status(500).json({
      status: 'error',
      error: error.message,
      success: false
    });
  }
});

module.exports = router;
