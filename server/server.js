const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
require('dotenv').config();

const predictionRoutes = require('./routes/prediction');

const app = express();
const PORT = process.env.PORT || 5000;

// CORS configuration
const corsOptions = {
  origin: function (origin, callback) {
    // Allow requests with no origin (like mobile apps or curl requests)
    if (!origin) return callback(null, true);
    
    // Remove trailing slash if present for comparison
    const normalizedOrigin = origin.endsWith('/') ? origin.slice(0, -1) : origin;
    
    // List of allowed origins
    const allowedOrigins = [
      /^https?:\/\/localhost(:[0-9]+)?$/,
      /^https?:\/\/127\.0\.0\.1(:[0-9]+)?$/,
      /^chrome-extension:\/\/[a-z0-9]+$/, // Allow any Chrome extension
    ];

    // Check if the origin matches any of the allowed patterns
    if (allowedOrigins.some(pattern => pattern.test(normalizedOrigin))) {
      return callback(null, true);
    }

    // For development, log the blocked origin
    if (process.env.NODE_ENV === 'development') {
      console.log('Blocked CORS request from origin:', origin);
    }
    
    return callback(new Error('Not allowed by CORS'));
  },
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true,
  optionsSuccessStatus: 200
};

// Apply CORS before other middleware
app.use(cors(corsOptions));

// Security middleware
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      scriptSrc: ["'self'"],
      connectSrc: ["'self'", process.env.ML_API_URL || 'http://localhost:8000']
    }
  }
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS) || 15 * 60 * 1000, // 15 minutes
  max: parseInt(process.env.RATE_LIMIT_MAX_REQUESTS) || 100, // limit each IP to 100 requests per windowMs
  message: {
    error: 'Too many requests from this IP, please try again later.',
    retryAfter: Math.ceil(parseInt(process.env.RATE_LIMIT_WINDOW_MS) / 1000) || 900
  },
  standardHeaders: true,
  legacyHeaders: false
});

app.use('/api/', limiter);


// Body parsing middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Trust proxy for accurate IP addresses
app.set('trust proxy', 1);

// MongoDB connection
const connectDB = async () => {
  try {
    const mongoURI = process.env.MONGODB_URI || 'mongodb://localhost:27017/phishing_detector';
    
    await mongoose.connect(mongoURI, {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    });
    
    console.log('✅ MongoDB connected successfully');
    
    // Handle connection events
    mongoose.connection.on('error', (err) => {
      console.error('❌ MongoDB connection error:', err);
    });
    
    mongoose.connection.on('disconnected', () => {
      console.log('⚠️ MongoDB disconnected');
    });
    
    mongoose.connection.on('reconnected', () => {
      console.log('✅ MongoDB reconnected');
    });
    
  } catch (error) {
    console.error('❌ MongoDB connection failed:', error.message);
    
    if (process.env.NODE_ENV === 'production') {
      process.exit(1);
    } else {
      console.log('⚠️ Continuing without database in development mode');
    }
  }
};

// Connect to database
connectDB();

// Routes
app.use('/api', predictionRoutes);

// Root endpoint
app.get('/', (req, res) => {
  res.json({
    name: 'Phishing Detector API',
    version: '1.0.0',
    description: 'AI-powered phishing website detection system',
    endpoints: {
      predict: 'POST /api/predict',
      history: 'GET /api/history',
      stats: 'GET /api/stats',
      health: 'GET /api/health'
    },
    status: 'running',
    timestamp: new Date().toISOString()
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    error: 'Endpoint not found',
    message: `Cannot ${req.method} ${req.originalUrl}`,
    availableEndpoints: [
      'GET /',
      'POST /api/predict',
      'GET /api/history',
      'GET /api/stats',
      'GET /api/health'
    ]
  });
});

// Global error handler
app.use((error, req, res, next) => {
  console.error('Global error handler:', error);
  
  // Handle specific error types
  if (error.name === 'ValidationError') {
    return res.status(400).json({
      error: 'Validation error',
      details: error.message
    });
  }
  
  if (error.name === 'CastError') {
    return res.status(400).json({
      error: 'Invalid data format',
      details: error.message
    });
  }
  
  // Default error response
  res.status(500).json({
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'development' ? error.message : 'Something went wrong'
  });
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  console.log('🔄 SIGTERM received, shutting down gracefully');
  
  try {
    await mongoose.connection.close();
    console.log('✅ Database connection closed');
  } catch (error) {
    console.error('❌ Error closing database:', error);
  }
  
  process.exit(0);
});

process.on('SIGINT', async () => {
  console.log('🔄 SIGINT received, shutting down gracefully');
  
  try {
    await mongoose.connection.close();
    console.log('✅ Database connection closed');
  } catch (error) {
    console.error('❌ Error closing database:', error);
  }
  
  process.exit(0);
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
  console.log(`📍 Environment: ${process.env.NODE_ENV || 'development'}`);
  console.log(`🔗 CORS origin: ${process.env.CORS_ORIGIN || 'http://localhost:3000'}`);
  console.log(`🤖 ML API URL: ${process.env.ML_API_URL || 'http://localhost:8000'}`);
  console.log(`📊 MongoDB URI: ${process.env.MONGODB_URI ? '[CONFIGURED]' : 'mongodb://localhost:27017/phishing_detector'}`);
});

module.exports = app;