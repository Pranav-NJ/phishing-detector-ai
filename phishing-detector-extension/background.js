// Initialize the extension when installed
chrome.runtime.onInstalled.addListener(() => {
  console.log('Extension installed/updated');
  updateBadge('safe');
  
  // Set up the initial state
  chrome.storage.local.set({ 'lastChecked': Date.now() });
  
  // Simple check for active tab on startup
  checkActiveTab();
  
  // Set up periodic checks using setInterval
  setInterval(checkActiveTab, 60000); // Check every minute
});

// Function to check the currently active tab
async function checkActiveTab() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab && tab.url) {
      checkUrl(tab.url);
    }
  } catch (error) {
    console.error('Error checking active tab:', error);
  }
}

// Listen for tab updates
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  console.log('Tab updated:', { tabId, url: tab.url, status: changeInfo.status });
  
  // Only proceed if URL changed and tab is fully loaded
  if (changeInfo.status === 'complete' && tab.url && tab.url.startsWith('http')) {
    try {
      // Skip chrome://, extension://, and other internal pages
      if (tab.url.startsWith('chrome://') || 
          tab.url.startsWith('edge://') || 
          tab.url.startsWith('about:') ||
          tab.url.startsWith('chrome-extension://')) {
        console.log('Skipping internal page:', tab.url);
        return;
      }
      
      console.log('Checking URL:', tab.url);
      checkUrl(tab.url);
    } catch (error) {
      console.error('Error in tab update listener:', error);
    }
  }
});

// Function to check URL against the API
async function checkUrl(url) {
  console.log('checkUrl called with:', url);
  
  // Don't check empty, invalid, or chrome:// URLs
  if (!url || !url.startsWith('http') || url.startsWith('chrome://') || url.startsWith('edge://') || url.startsWith('about:')) {
    console.log('Skipping URL check for system URL:', url);
    updateBadge('safe');
    return { prediction: 'safe', confidence: 1 };
  }
  
  // Skip checking extension pages and Chrome Web Store
  if (url.startsWith('chrome-extension://') || 
      url.startsWith('chrome://') || 
      url.startsWith('edge://') || 
      url.startsWith('about:') || 
      url.includes('chrome.google.com/webstore')) {
    console.log('Skipping extension/internal page:', url);
    return;
  }
  try {
    const apiUrl = 'http://localhost:5000/api/predict';
    console.log('Sending request to:', apiUrl);
    
    try {
      const response = await fetch(apiUrl, {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "X-Requested-With": "XMLHttpRequest"
        },
        body: JSON.stringify({ url })
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('API Error:', {
          status: response.status,
          statusText: response.statusText,
          error: errorText
        });
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log("Scan result:", data);

      // Store the result for the popup
      await chrome.storage.local.set({ [url]: data });

      // Update badge for the current tab
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (tab && tab.url === url) {
        updateBadge(data.prediction);
      }

      // Log alert for dangerous sites
      if (data.prediction === "dangerous") {
        logAlert(`The site ${new URL(url).hostname} seems dangerous!`);
      }
      
      return data;
    } catch (err) {
      console.error("Error checking URL:", err);
      updateBadge("error");
      throw err; // Re-throw to be caught by the caller
    }
  } catch (err) {
    console.error("Error in checkUrl:", err);
    updateBadge("error");
    throw err;
  }
}

// Update extension badge
function updateBadge(status) {
  const colors = {
    safe: "#4CAF50",
    suspicious: "#FFC107",
    dangerous: "#F44336",
    error: "#9E9E9E"
  };

  const texts = {
    safe: "✓",
    suspicious: "?",
    dangerous: "!",
    error: "!"
  };

  const color = colors[status] || "#9E9E9E";
  const text = texts[status] || "?";

  try {
    chrome.action.setBadgeText({ text });
    chrome.action.setBadgeBackgroundColor({ color });
    chrome.action.setBadgeTextColor({ color: "#FFFFFF" });
  } catch (error) {
    console.error("Error updating badge:", error);
  }
}

// Log alert to console instead of showing notification
function logAlert(message) {
  console.log("ALERT:", message);
}
