// Configuration for different environments
const config = {
    development: {
        API_BASE: "https://localhost:8443/api"
    },
    production: {
        // Force HTTP to avoid SSL certificate issues - with cache busting
        API_BASE: "http://4020811e2f92.ngrok-free.app/api?v=" + Date.now()
        // HTTPS URL with SSL issues:
        // API_BASE: "https://4020811e2f92.ngrok-free.app/api"
    }
};

// Auto-detect environment
const isDevelopment = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
const currentConfig = isDevelopment ? config.development : config.production;

const API_BASE = currentConfig.API_BASE;
