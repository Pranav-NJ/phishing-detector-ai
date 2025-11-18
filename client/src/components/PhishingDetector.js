import React, { useState } from 'react';
import axios from 'axios';

const PhishingDetector = () => {
  const [url, setUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  // URL validation regex
  const isValidUrl = (string) => {
    try {
      new URL(string);
      return true;
    } catch (_) {
      return false;
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Reset states
    setError('');
    setResult(null);
    
    // Validate URL
    if (!url.trim()) {
      setError('Please enter a URL to analyze.');
      return;
    }
    
    if (!isValidUrl(url)) {
      setError('Please enter a valid URL (including http:// or https://).');
      return;
    }
    
    setLoading(true);
    
    try {
      // Call backend API
      const response = await axios.post('/api/predict', {
        url: url.trim()
      });
      
      setResult(response.data);
    } catch (err) {
      console.error('Prediction error:', err);
      if (err.response) {
        // Server responded with error
        setError(err.response.data.error || 'Server error occurred.');
      } else if (err.request) {
        // Network error
        setError('Network error. Please check if the server is running.');
      } else {
        // Other error
        setError('An unexpected error occurred.');
      }
    } finally {
      setLoading(false);
    }
  };

  const renderResult = () => {
    if (!result) return null;
    
    const { prediction, confidence, url: analyzedUrl } = result;
    const isPhishing = prediction === true || prediction === 'phishing';
    
    return (
      <div className={`result-container ${isPhishing ? 'result-dangerous' : 'result-safe'}`}>
        <div className="result-title">
          {isPhishing ? '🚨 DANGEROUS' : '✅ SAFE'}
        </div>
        <div className="result-details">
          {isPhishing 
            ? 'This URL appears to be a phishing site. Avoid entering personal information.'
            : 'This URL appears to be legitimate and safe to visit.'
          }
        </div>
        <div className="confidence-score">
          Confidence: {Math.round(confidence * 100)}%
        </div>
        <div style={{ fontSize: '0.8rem', marginTop: '15px', opacity: 0.8 }}>
          Analyzed URL: {analyzedUrl}
        </div>
      </div>
    );
  };

  return (
    <div className="phishing-detector">
      <h2 style={{ marginBottom: '30px', fontSize: '1.8rem' }}>
        🔍 URL Security Analysis
      </h2>
      
      <form onSubmit={handleSubmit} className="url-form">
        <input
          type="text"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="Enter URL to analyze (e.g., https://example.com)"
          className="url-input"
          disabled={loading}
        />
        <button 
          type="submit" 
          className="analyze-btn"
          disabled={loading}
        >
          {loading ? (
            <>
              <div className="loading-spinner"></div>
              Analyzing...
            </>
          ) : (
            'Analyze URL'
          )}
        </button>
      </form>
      
      {error && (
        <div className="error-message">
          ⚠️ {error}
        </div>
      )}
      
      {renderResult()}
      
      <div style={{ 
        fontSize: '0.8rem', 
        marginTop: '30px', 
        opacity: 0.7,
        lineHeight: '1.4'
      }}>
        <p>💡 <strong>Tips:</strong></p>
        <p>• Always include http:// or https:// in the URL</p>
        <p>• Be cautious with URLs from unknown sources</p>
        <p>• This tool analyzes URL patterns and structure</p>
      </div>
    </div>
  );
};

export default PhishingDetector;