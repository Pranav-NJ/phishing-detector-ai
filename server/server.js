const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
require('dotenv').config();
const { initRedis, closeRedis, isRedisEnabled } = require('./utils/redisClient');

const predictionRoutes = require('./routes/prediction');

const app = express();
const PORT = process.env.PORT || 5000;

const corsOptions = {
  origin: '*',
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true,
  optionsSuccessStatus: 200
};

app.use(cors(corsOptions));
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      connectSrc: ["'self'", process.env.ML_API_URL || 'http://localhost:8000']
    }
  }
}));

const limiter = rateLimit({
  windowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS) || 15 * 60 * 1000,
  max: parseInt(process.env.RATE_LIMIT_MAX_REQUESTS) || 100
});
app.use('/api/', limiter);

app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));
app.set('trust proxy', 1);

let _redisConnected = false;

async function setupAndStart() {
  try {
    const mongoURI = process.env.MONGODB_URI || 'mongodb://localhost:27017/phishing_detector';
    await mongoose.connect(mongoURI);
    console.log('✅ MongoDB connected successfully');
  } catch (error) {
    console.warn('⚠️ MongoDB connection failed:', error.message);
  }

  try {
    _redisConnected = await initRedis();
  } catch (e) {
    console.warn('⚠️ Redis setup error:', e && e.message ? e.message : e);
  }

  app.use('/api', predictionRoutes);

  app.get('/', (req, res) => {
    res.json({ status: 'running', timestamp: new Date().toISOString() });
  });

  app.use('*', (req, res) => {
    res.status(404).json({ error: 'Endpoint not found' });
  });

  app.use((error, req, res, next) => {
    res.status(500).json({ error: 'Internal server error' });
  });

  app.listen(PORT, () => {
    console.log(`🚀 Server running on port ${PORT}`);
    console.log(`🤖 ML API URL: ${process.env.ML_API_URL || 'http://localhost:8000'}`);
  });
}

setupAndStart();

process.on('SIGINT', async () => {
  await closeRedis();
  await mongoose.connection.close();
  process.exit(0);
});
module.exports = app;
