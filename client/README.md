# 🎨 Phishing Detector - React Frontend

The React frontend for the Phishing Detector AI system. Provides a modern, responsive web interface for URL analysis.

## ✨ Features

- **Modern UI**: Clean, responsive design with gradient backgrounds
- **Real-time Analysis**: Submit URLs and get instant feedback
- **Visual Feedback**: Color-coded results with confidence scores
- **Loading States**: Smooth animations during analysis
- **Error Handling**: User-friendly error messages
- **Mobile Responsive**: Works great on all devices

## 🚀 Quick Start

### Prerequisites
- Node.js (v16+)
- npm or yarn

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm start
```

The app will open at [http://localhost:3000](http://localhost:3000).

### Available Scripts

- `npm start` - Runs the app in development mode
- `npm run build` - Builds the app for production
- `npm test` - Launches the test runner
- `npm run eject` - **Note: this is a one-way operation. Don't eject unless you know what you're doing!**

## 📁 Project Structure

```
src/
├── components/
│   └── PhishingDetector.js    # Main detection component
├── App.js                     # Root component
├── App.css                    # Global styles
├── index.js                   # React entry point
└── index.css                  # Base styles
public/
├── index.html                 # HTML template
└── ...
```

## 🎨 Styling

The app uses modern CSS with:
- **CSS Variables** for consistent theming
- **Flexbox/Grid** for layouts
- **CSS Animations** for smooth interactions
- **Media queries** for responsive design
- **Backdrop filters** for modern glass effects

### Color Scheme
- **Primary Gradient**: Purple to blue (`#667eea` to `#764ba2`)
- **Success**: Green gradient (`#2ed573` to `#1e90ff`)
- **Danger**: Red gradient (`#ff3742` to `#ff6b6b`)
- **Warning**: Orange gradient (`#ffa726` to `#ff9800`)

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the client directory:

```env
# API Configuration
REACT_APP_API_BASE_URL=http://localhost:5000
REACT_APP_ML_API_URL=http://localhost:8000

# Optional: Google Analytics
REACT_APP_GA_TRACKING_ID=your-ga-id
```

### Proxy Configuration

The `package.json` includes a proxy setting for development:

```json
{
  "proxy": "http://localhost:5000"
}
```

This allows the frontend to make API calls to `/api/predict` without CORS issues during development.

## 🧩 Components

### PhishingDetector Component

The main component that handles:
- URL input validation
- API communication
- Result display
- Error handling
- Loading states

**Props**: None (self-contained)

**State**:
- `url`: Current URL input
- `loading`: Loading state
- `result`: API response
- `error`: Error messages

### Key Features:

```javascript
// URL validation
const isValidUrl = (string) => {
  try {
    new URL(string);
    return true;
  } catch (_) {
    return false;
  }
};

// API call with axios
const response = await axios.post('/api/predict', {
  url: url.trim()
});
```

## 🔌 API Integration

### Backend API Endpoints

The frontend communicates with:

```javascript
// Main prediction endpoint
POST /api/predict
{
  "url": "https://example.com"
}

// Response format
{
  "url": "https://example.com",
  "prediction": false, // true for phishing
  "confidence": 0.95,
  "processingTime": 150,
  "success": true
}
```

### Error Handling

The app handles various error types:
- **Network errors**: Backend not running
- **Validation errors**: Invalid URL format
- **Server errors**: Internal server issues
- **Timeout errors**: Slow responses

## 🎯 User Experience

### Flow
1. User enters URL in input field
2. Click "Analyze URL" button
3. Loading spinner shows during processing
4. Results display with:
   - Safety status (Safe/Dangerous)
   - Confidence percentage
   - Visual indicators (colors, icons)
   - Additional tips

### Validation
- Empty URL check
- URL format validation
- Real-time feedback
- Helpful error messages

### Visual Feedback
- **Green**: Safe URLs
- **Red**: Dangerous URLs
- **Loading**: Animated spinner
- **Smooth transitions**: CSS animations

## 📱 Responsive Design

The app adapts to different screen sizes:

```css
@media (max-width: 768px) {
  .App-header h1 {
    font-size: 2rem;
  }
  
  .phishing-detector {
    padding: 25px;
    margin: 0 10px;
  }
}
```

## 🧪 Testing

### Manual Testing

Test with these URLs:
- **Safe**: `https://www.google.com`
- **Suspicious**: `http://paypal-verify.tk/login`
- **Invalid**: `not-a-url`
- **Empty**: (leave blank)

### Automated Testing

```bash
# Run tests
npm test

# Run tests in watch mode
npm test -- --watch

# Run tests with coverage
npm test -- --coverage
```

## 🚀 Deployment

### Development Build
```bash
npm run build
```

### Production Deployment

#### Netlify
1. Connect your GitHub repository
2. Build command: `npm run build`
3. Publish directory: `build`
4. Set environment variables in Netlify dashboard

#### Vercel
1. Install Vercel CLI: `npm i -g vercel`
2. Run `vercel` in project directory
3. Follow the prompts

#### Traditional Web Server
```bash
npm run build
# Copy the 'build' folder to your web server
```

### Environment Variables for Production

```env
REACT_APP_API_BASE_URL=https://your-backend.herokuapp.com
REACT_APP_ML_API_URL=https://your-ml-api.herokuapp.com
```

## 🔧 Development Tips

### Hot Reloading
The development server supports hot reloading. Changes to components will automatically refresh the browser.

### Debugging
- Use React Developer Tools browser extension
- Console.log in components for debugging
- Network tab to inspect API calls

### Code Organization
- Keep components small and focused
- Use custom hooks for complex logic
- Separate concerns (UI vs. logic)

## 🎨 Customization

### Changing Colors
Edit the CSS variables in `App.css`:

```css
:root {
  --primary-color: #667eea;
  --secondary-color: #764ba2;
  --success-color: #2ed573;
  --danger-color: #ff3742;
}
```

### Adding New Features
1. Create new components in `src/components/`
2. Import and use in `App.js`
3. Add styles to component-specific CSS files

### Performance Optimization
- Use React.memo for expensive components
- Implement code splitting with React.lazy
- Optimize images and assets

## 🐛 Common Issues

**White screen on production**
- Check console for errors
- Verify environment variables
- Check build process

**API calls failing**
- Verify backend is running
- Check CORS configuration
- Verify API endpoints

**Styling issues**
- Clear browser cache
- Check CSS syntax
- Verify responsive breakpoints

## 📚 Dependencies

### Main Dependencies
- **react**: UI library
- **axios**: HTTP client
- **react-scripts**: Build tools

### Development Dependencies
- **@testing-library/react**: Testing utilities
- **@testing-library/jest-dom**: Jest matchers

## 🔄 Updates

To update dependencies:

```bash
# Check for updates
npm outdated

# Update all dependencies
npm update

# Update specific dependency
npm install package-name@latest
```

## 📞 Support

For frontend-specific issues:
1. Check browser console for errors
2. Verify API endpoints are accessible
3. Test with different browsers
4. Check responsive design on mobile

---

**Built with ❤️ using React**