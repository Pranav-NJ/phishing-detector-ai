document.addEventListener('DOMContentLoaded', () => {
  const statusElement = document.getElementById('status');
  const urlElement = document.getElementById('url');
  const confidenceElement = document.getElementById('confidence');
  const refreshButton = document.getElementById('refresh');

  // Get current tab and check URL
  async function checkCurrentTab() {
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      console.log('Tab info:', tab);
      
      if (!tab?.url) {
        console.log('No tab URL found');
        updateUI('error', 'No active tab found');
        return;
      }

      console.log('Raw tab URL:', tab.url);

      // Skip extension pages and invalid URLs
      if (!tab.url || 
          tab.url.startsWith('chrome://') || 
          tab.url.startsWith('chrome-extension://') || 
          tab.url.startsWith('edge://') || 
          tab.url.startsWith('about:') ||
          tab.url.startsWith('moz-extension://') ||
          !tab.url.startsWith('http')) {
        console.log('Skipping internal URL:', tab.url);
        updateUI('safe', 'Internal browser page');
        return;
      }

      const url = new URL(tab.url);
      urlElement.textContent = url.hostname;

      // Check if we have cached result
      const result = await chrome.storage.local.get(url.href);
      if (result[url.href]) {
        console.log('Using cached result:', result[url.href]);
        // Backend returns prediction: true for PHISHING, false for SAFE
        const isPhishing = result[url.href].prediction;
        updateUI(isPhishing ? 'dangerous' : 'safe', result[url.href].confidence);
        return;
      }

      // If no cached result, show loading and wait for background script
      console.log('No cached result, waiting for background script...');
      updateUI('safe', 'Loading...');
      
      // Wait for background script to cache result (retry up to 3 times)
      let retries = 0;
      const checkCache = async () => {
        retries++;
        const cachedResult = await chrome.storage.local.get(url.href);
        if (cachedResult[url.href]) {
          console.log('Got cached result after retry:', cachedResult[url.href]);
          // Backend returns prediction: true for PHISHING, false for SAFE
          const isPhishing = cachedResult[url.href].prediction;
          updateUI(isPhishing ? 'dangerous' : 'safe', cachedResult[url.href].confidence);
          return;
        }
        
        if (retries < 3) {
          setTimeout(checkCache, 1000); // Wait 1 second and retry
        } else {
          console.log('No cached result after retries, showing error');
          updateUI('error');
        }
      };
      
      setTimeout(checkCache, 500); // Initial wait for background script

    } catch (error) {
      console.error('Error:', error);
      updateUI('error');
    }
  }

  // Update the UI based on the result
  function updateUI(status, confidence) {
    const statusMap = {
      safe: { text: '✅ Safe Website', class: 'safe' },
      suspicious: { text: '⚠️ Suspicious Website', class: 'suspicious' },
      dangerous: { text: '🚨 Dangerous Website!', class: 'dangerous' },
      error: { text: '❌ Error checking URL', class: 'error' }
    };

    const statusInfo = statusMap[status] || statusMap.error;
    statusElement.textContent = statusInfo.text;
    statusElement.className = statusInfo.class;

    if (confidence !== undefined) {
      confidenceElement.textContent = `Confidence: ${Math.round(confidence * 100)}%`;
    } else {
      confidenceElement.textContent = '';
    }
  }

  // Add click handler for refresh button
  refreshButton.addEventListener('click', checkCurrentTab);

  // Initial check
  checkCurrentTab();
});
