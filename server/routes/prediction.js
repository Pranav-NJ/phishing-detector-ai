const express = require('express');
const axios = require('axios');
const Joi = require('joi');
const Prediction = require('../models/Prediction');

const router = express.Router();

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

    // Call ML API
    const mlApiUrl = `${process.env.ML_API_URL}${process.env.ML_API_ENDPOINT || '/predict'}`;

    let mlResponse;
    try {
      mlResponse = await axios.post(
        mlApiUrl,
        { url },
        {
          timeout: 30000,
          headers: {
            'Content-Type': 'application/json'
          }
        }
      );
    } catch (mlError) {
      console.error('ML API Error:', mlError.message);

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

    // Extract prediction and confidence
    const mlData = mlResponse.data;
    const prediction = mlData.prediction || 'legitimate';
    const confidence = mlData.confidence || 0.5;
    const processingTime = Date.now() - startTime;

    // Save prediction to DB
    try {
      const predictionRecord = new Prediction({
        url,
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

    // Host-based safe overrides
    const hostname = new URL(url).hostname.toLowerCase();
    const SAFE_HOSTS = new Set([
      'chatgpt.com',
      'openai.com',
      'chat.openai.com',
      'platform.openai.com',
      'auth.openai.com'
    ]);

    let predictionStatus = 'safe';
    let isPhishing = false;

    if (SAFE_HOSTS.has(hostname)) {
      predictionStatus = 'safe';
      isPhishing = false;
    } else if (prediction === 'phishing' || prediction === 'dangerous') {
      predictionStatus = 'phishing';
      isPhishing = true;
    } else if (confidence && confidence < 0.6) {
      predictionStatus = 'suspicious';
      isPhishing = false;
    }

    res.json({
      url,
      prediction: isPhishing,
      confidence: confidence || 0.5,
      processingTime,
      success: true
    });
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

    const overall = mlApiHealthy && dbHealthy;

    res.status(overall ? 200 : 503).json({
      status: overall ? 'healthy' : 'unhealthy',
      services: {
        mlApi: mlApiHealthy ? 'healthy' : 'unhealthy',
        database: dbHealthy ? 'healthy' : 'unhealthy'
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
