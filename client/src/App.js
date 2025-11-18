import React from 'react';
import './App.css';
import PhishingDetector from './components/PhishingDetector';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>🛡️ Phishing Detector AI</h1>
        <p>Advanced AI-powered URL security analysis</p>
      </header>
      <main>
        <PhishingDetector />
      </main>
      <footer className="App-footer">
        <p>Protect yourself from malicious websites with AI technology</p>
      </footer>
    </div>
  );
}

export default App;