const mongoose = require('mongoose');

const PredictionSchema = new mongoose.Schema({
  url: {
    type: String,
    required: true,
    trim: true,
    maxlength: 2048 // Standard maximum URL length
  },
  prediction: {
    type: Boolean,
    required: true
  },
  confidence: {
    type: Number,
    required: true,
    min: 0,
    max: 1
  },
  timestamp: {
    type: Date,
    default: Date.now
  },
  userAgent: {
    type: String,
    default: ''
  },
  ipAddress: {
    type: String,
    default: ''
  },
  processingTimeMs: {
    type: Number,
    default: 0
  }
}, {
  timestamps: true
});

// Create index for efficient queries
PredictionSchema.index({ url: 1 });
PredictionSchema.index({ timestamp: -1 });
PredictionSchema.index({ prediction: 1 });

// Static method to get prediction statistics
PredictionSchema.statics.getStats = async function() {
  const stats = await this.aggregate([
    {
      $group: {
        _id: null,
        totalPredictions: { $sum: 1 },
        phishingCount: {
          $sum: { $cond: [{ $eq: ['$prediction', true] }, 1, 0] }
        },
        legitimateCount: {
          $sum: { $cond: [{ $eq: ['$prediction', false] }, 1, 0] }
        },
        avgConfidence: { $avg: '$confidence' },
        avgProcessingTime: { $avg: '$processingTimeMs' }
      }
    }
  ]);
  
  return stats.length > 0 ? stats[0] : {
    totalPredictions: 0,
    phishingCount: 0,
    legitimateCount: 0,
    avgConfidence: 0,
    avgProcessingTime: 0
  };
};

// Static method to get recent predictions
PredictionSchema.statics.getRecent = function(limit = 10) {
  return this.find()
    .select('url prediction confidence timestamp')
    .sort({ timestamp: -1 })
    .limit(limit);
};

module.exports = mongoose.model('Prediction', PredictionSchema);