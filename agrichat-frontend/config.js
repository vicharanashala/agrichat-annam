// Configuration for different environments
const config = {
    development: {
        API_BASE: "https://localhost:8443/api"
    },
    production: {
        // Use HTTPS with ngrok - browsers will show warning but allow override
        API_BASE: "https://4020811e2f92.ngrok-free.app/api"
        // Fallback HTTP (will cause mixed content warnings):
        // API_BASE: "http://4020811e2f92.ngrok-free.app/api"
    }
};

// Auto-detect environment
const isDevelopment = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
const currentConfig = isDevelopment ? config.development : config.production;

// Add cache busting to prevent cached API calls
const API_BASE = currentConfig.API_BASE + (currentConfig.API_BASE.includes('?') ? '&' : '?') + 't=' + Date.now();
