// Configuration for different environments
const config = {
    development: {
        API_BASE: "https://localhost:8443/api"
    },
    production: {
        // ngrok HTTPS URL for your local backend (forwarding to HTTP backend)
        API_BASE: "https://4020811e2f92.ngrok-free.app/api"
        // Previous URLs:
        // API_BASE: "https://05d5a0c84535.ngrok-free.app/api"
    }
};

// Auto-detect environment
const isDevelopment = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
const currentConfig = isDevelopment ? config.development : config.production;

const API_BASE = currentConfig.API_BASE;
