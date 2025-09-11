const config = {
    development: {
        API_BASE: "http://localhost:8000/api"
    },
    production: {
        API_BASE: "https://443c63b57774.ngrok-free.app/api"

    }
};

const isDevelopment = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
const currentConfig = isDevelopment ? config.development : config.production;

const API_BASE = currentConfig.API_BASE;

console.log("Using API_BASE:", API_BASE);

console.log('[CONFIG] isDevelopment:', isDevelopment);
console.log('[CONFIG] currentConfig:', currentConfig);
console.log('[CONFIG] API_BASE:', API_BASE);
