// Clear browser debugging script
// This will help identify why HTTPS URLs are still being used

console.log("=== AgriChat Debug Information ===");
console.log("Current API_BASE:", typeof API_BASE !== 'undefined' ? API_BASE : 'API_BASE not loaded');
console.log("Window location:", window.location.href);
console.log("Environment detected:",
    (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
        ? 'development' : 'production'
);

// Check if config.js was loaded properly
if (typeof config !== 'undefined') {
    console.log("Config object loaded:", config);
    console.log("Production API_BASE:", config.production.API_BASE);
} else {
    console.error("Config object not loaded! Check if config.js is included.");
}

// Check for cached values
console.log("Local storage items:");
for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (key.includes('api') || key.includes('url') || key.includes('base')) {
        console.log(`  ${key}: ${localStorage.getItem(key)}`);
    }
}

// Force clear any cached API URLs
console.log("Clearing potential cached API URLs...");
localStorage.removeItem('api_base');
localStorage.removeItem('API_BASE');
localStorage.removeItem('api_url');

console.log("=== End Debug Information ===");
