// Configuration for different environments
const config = {
    development: {
        API_BASE: "https://localhost:8443/api"
    },
    production: {
        // Use HTTPS with ngrok - browsers will show warning but allow override
        API_BASE: "https://0c9d97e9dff0.ngrok-free.app/api"
        // Fallback HTTP (will cause mixed content warnings):
        // API_BASE: "http://YOUR_NEW_NGROK_URL.ngrok-free.app/api"
    }
};

// Auto-detect environment
const isDevelopment = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
const currentConfig = isDevelopment ? config.development : config.production;

// Clean API base without cache busting (cache busting will be added per request)
const API_BASE = currentConfig.API_BASE;

// Debug logging
console.log('[CONFIG] isDevelopment:', isDevelopment);
console.log('[CONFIG] currentConfig:', currentConfig);
console.log('[CONFIG] API_BASE:', API_BASE);
