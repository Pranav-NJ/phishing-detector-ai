const express = require('express');
const cors = require('cors');

const app = express();
const PORT = 5000;

app.use(cors());
app.use(express.json());

// Mock prediction function
function predictPhishing(url) {
  // Simple mock logic - in a real app, this would use your ML model
  const hostname = new URL(url).hostname;
  const isDangerous = hostname.includes('phish') || hostname.includes('hack') || hostname.includes('malware');
  const isSuspicious = Math.random() > 0.7; // 30% chance of being suspicious
  
  if (isDangerous) {
    return { prediction: 'dangerous', confidence: 0.9 + Math.random() * 0.1 };
  } else if (isSuspicious) {
    return { prediction: 'suspicious', confidence: 0.4 + Math.random() * 0.3 };
  } else {
    return { prediction: 'safe', confidence: 0.8 + Math.random() * 0.2 };
  }
}

// API endpoint
app.post('/api/predict', (req, res) => {
  try {
    const { url } = req.body;
    if (!url) {
      return res.status(400).json({ error: 'URL is required' });
    }
    
    // Add artificial delay to simulate network
    setTimeout(() => {
      const result = predictPhishing(url);
      res.json(result);
    }, 500);
  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.listen(PORT, () => {
  console.log(`Mock server running on http://localhost:${PORT}`);
  console.log(`Test with: curl -X POST http://localhost:${PORT}/api/predict -H "Content-Type: application/json" -d '{"url":"https://example.com"}'`);
});
