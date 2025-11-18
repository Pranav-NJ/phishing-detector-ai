document.addEventListener('DOMContentLoaded', () => {
  const statusElement = document.getElementById('status');
  const urlElement = document.getElementById('url');
  const confidenceElement = document.getElementById('confidence');
  const refreshButton = document.getElementById('refresh');

  // Get current tab and check URL
  async function checkCurrentTab() {
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tab?.url) {
        updateUI('error', 'No active tab found');
        return;
      }

      const url = new URL(tab.url);
      urlElement.textContent = url.hostname;

      // Check if we have cached result
      const result = await chrome.storage.local.get(url.href);
      if (result[url.href]) {
        updateUI(result[url.href].prediction, result[url.href].confidence);
      }

      // Make a new request
      try {
        const response = await fetch("http://localhost:5000/api/predict", {
          method: "POST",
          headers: { 
            "Content-Type": "application/json",
            "X-Requested-With": "XMLHttpRequest"
          },
          credentials: 'include',
          body: JSON.stringify({ url: url.href })
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        await chrome.storage.local.set({ [url.href]: data });
        updateUI(data.prediction, data.confidence);
      } catch (error) {
        console.error('Error checking URL:', error);
        updateUI('error');
      }

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
