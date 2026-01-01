import React, { useState } from 'react';
import axios from 'axios';

const PhishingDetector = () => {
  // Use explicit API base if provided (set REACT_APP_API_BASE=http://localhost:5000)
  // Fall back to relative path so CRA proxy still works in development.
  const API_BASE = process.env.REACT_APP_API_BASE || '';

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
      // Call backend API (use explicit API base if configured)
      const response = await axios.post(`${API_BASE}/api/predict`, {
        url: url.trim()
      });
      
      console.log('API Response:', response.data);
      console.log('Details:', response.data.details);
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
    
    const { prediction, confidence, url: analyzedUrl, details, risk_level, risk_score } = result;
    const isPhishing = prediction === true || prediction === 'phishing';
    
    // Extract detailed information from API response
    const urlStructure = details?.url_structure || {};
    const domainAnalysis = details?.domain_analysis || {};
    const phishingIndicators = details?.phishing_indicators || [];
    const positiveIndicators = details?.positive_indicators || [];
    const suspiciousKeywords = details?.suspicious_keywords || [];
    const brandKeywords = details?.brand_keywords || [];
    
    return (
      <div className={`result-container ${isPhishing ? 'result-dangerous' : 'result-safe'}`}>
        {/* Main Result */}
        <div className="result-title">
          {isPhishing ? '🚨 PHISHING DETECTED!' : '✅ SAFE / LEGITIMATE'}
        </div>
        <div className="result-details">
          {isPhishing 
            ? 'This URL is likely a phishing attack. Do not enter personal information!'
            : 'This URL appears to be legitimate and safe to visit.'
          }
        </div>
        <div className="confidence-score">
          Confidence: {Math.round(confidence * 100)}%
          {risk_level && <span style={{ marginLeft: '15px' }}>Risk Level: {risk_level}</span>}
        </div>
        
        {/* Detailed Analysis Section */}
        <div style={{ marginTop: '25px', textAlign: 'left' }}>
          <h3 style={{ borderBottom: '2px solid #ddd', paddingBottom: '10px', marginBottom: '15px' }}>
            📊 DETAILED ANALYSIS
          </h3>
          
          {/* URL Structure */}
          {urlStructure && Object.keys(urlStructure).length > 0 && (
            <div style={{ marginBottom: '20px' }}>
              <h4 style={{ color: '#2c3e50', marginBottom: '10px' }}>🔗 URL Structure</h4>
              <div style={{ background: '#f8f9fa', padding: '12px', borderRadius: '6px', fontSize: '0.9rem', color: '#000' }}>
                {urlStructure.protocol && <div style={{ color: '#000' }}>• Protocol: <strong>{urlStructure.protocol}</strong></div>}
                {urlStructure.domain && <div style={{ color: '#000' }}>• Domain: <strong>{urlStructure.domain}</strong></div>}
                {urlStructure.subdomain && <div style={{ color: '#000' }}>• Subdomain: <strong>{urlStructure.subdomain}</strong></div>}
                {urlStructure.path && <div style={{ color: '#000' }}>• Path: <strong>{urlStructure.path}</strong></div>}
                {urlStructure.tld && <div style={{ color: '#000' }}>• TLD: <strong>.{urlStructure.tld}</strong></div>}
              </div>
            </div>
          )}
          
          {/* Domain Analysis */}
          {domainAnalysis && Object.keys(domainAnalysis).length > 0 && (
            <div style={{ marginBottom: '20px' }}>
              <h4 style={{ color: '#2c3e50', marginBottom: '10px' }}>🌐 Domain Analysis</h4>
              <div style={{ background: '#f8f9fa', padding: '12px', borderRadius: '6px', fontSize: '0.9rem', color: '#000' }}>
                {domainAnalysis.domain_length !== undefined && (
                  <div style={{ color: '#000' }}>• Domain length: <strong>{domainAnalysis.domain_length} characters</strong></div>
                )}
                {domainAnalysis.has_digits !== undefined && (
                  <div style={{ color: '#000' }}>• Has digits in domain: <strong>{domainAnalysis.has_digits ? 'Yes' : 'No'}</strong></div>
                )}
                {domainAnalysis.subdomain_length && domainAnalysis.subdomain_length > 0 && (
                  <>
                    <div style={{ color: '#000' }}>• Subdomain length: <strong>{domainAnalysis.subdomain_length} characters</strong></div>
                    {domainAnalysis.subdomain_numeric_ratio && (
                      <div style={{ color: '#000' }}>• Subdomain numeric ratio: <strong>{domainAnalysis.subdomain_numeric_ratio}</strong></div>
                    )}
                  </>
                )}
              </div>
            </div>
          )}
          
          {/* Phishing Indicators */}
          {phishingIndicators && phishingIndicators.length > 0 && (
            <div style={{ marginBottom: '20px' }}>
              <h4 style={{ color: '#e74c3c', marginBottom: '10px' }}>⚠️ PHISHING INDICATORS</h4>
              <div style={{ background: '#fee', padding: '12px', borderRadius: '6px', fontSize: '0.85rem' }}>
                {phishingIndicators.map((indicator, idx) => (
                  <div key={idx} style={{ marginBottom: '8px', padding: '8px', background: 'white', borderRadius: '4px', borderLeft: `3px solid ${indicator.severity === 'critical' ? '#c0392b' : indicator.severity === 'high' ? '#e67e22' : '#f39c12'}`, color: '#000' }}>
                    <strong style={{ color: '#000' }}>❌ {indicator.type}</strong>
                    <div style={{ fontSize: '0.85rem', color: '#000', marginTop: '3px' }}>
                      {indicator.description}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* Suspicious Keywords */}
          {suspiciousKeywords && suspiciousKeywords.length > 0 && (
            <div style={{ marginBottom: '20px' }}>
              <h4 style={{ color: '#e67e22', marginBottom: '10px' }}>🔍 Suspicious Keywords Found</h4>
              <div style={{ background: '#fff3cd', padding: '12px', borderRadius: '6px', fontSize: '0.9rem', color: '#000' }}>
                <strong style={{ color: '#000' }}>{suspiciousKeywords.join(', ')}</strong>
              </div>
            </div>
          )}
          
          {/* Brand Keywords */}
          {brandKeywords && brandKeywords.length > 0 && (
            <div style={{ marginBottom: '20px' }}>
              <h4 style={{ color: '#e67e22', marginBottom: '10px' }}>🏢 Brand Keywords Detected</h4>
              <div style={{ background: '#fff3cd', padding: '12px', borderRadius: '6px', fontSize: '0.9rem', color: '#000' }}>
                <strong style={{ color: '#000' }}>{brandKeywords.join(', ')}</strong>
                <div style={{ fontSize: '0.85rem', color: '#000', marginTop: '5px' }}>
                  Possible brand impersonation attempt
                </div>
              </div>
            </div>
          )}
          
          {/* Positive Indicators */}
          {positiveIndicators && positiveIndicators.length > 0 && (
            <div style={{ marginBottom: '20px' }}>
              <h4 style={{ color: '#27ae60', marginBottom: '10px' }}>✅ POSITIVE INDICATORS</h4>
              <div style={{ background: '#d4edda', padding: '12px', borderRadius: '6px', fontSize: '0.9rem', color: '#000' }}>
                {positiveIndicators.map((indicator, idx) => (
                  <div key={idx} style={{ color: '#000' }}>✓ {indicator}</div>
                ))}
              </div>
            </div>
          )}
        </div>
        
        <div style={{ fontSize: '0.75rem', marginTop: '20px', opacity: 0.7, borderTop: '1px solid #ddd', paddingTop: '10px' }}>
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