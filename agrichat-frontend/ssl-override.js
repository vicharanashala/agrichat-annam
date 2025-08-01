// SSL Certificate Override Helper
// This script helps users bypass ngrok's SSL certificate issues

function handleSSLError() {
    const modal = document.createElement('div');
    modal.innerHTML = `
        <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
                    background: rgba(0,0,0,0.8); z-index: 10000; display: flex; 
                    align-items: center; justify-content: center;">
            <div style="background: white; padding: 30px; border-radius: 10px; 
                       max-width: 500px; text-align: center;">
                <h3>SSL Certificate Issue</h3>
                <p>The backend server is using a self-signed certificate. 
                   You need to accept it to continue.</p>
                <p><strong>Steps:</strong></p>
                <ol style="text-align: left;">
                    <li>Click the button below</li>
                    <li>Click "Advanced" on the warning page</li>
                    <li>Click "Proceed to site (unsafe)"</li>
                    <li>Return to this page and refresh</li>
                </ol>
                <button onclick="window.open('${currentConfig.API_BASE.replace('/api', '')}/docs', '_blank')" 
                        style="background: #007cba; color: white; padding: 10px 20px; 
                               border: none; border-radius: 5px; margin: 10px;">
                    Accept SSL Certificate
                </button>
                <button onclick="this.parentElement.parentElement.remove()" 
                        style="background: #ccc; color: black; padding: 10px 20px; 
                               border: none; border-radius: 5px; margin: 10px;">
                    Close
                </button>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
}

// Auto-detect SSL errors and show helper
window.addEventListener('load', () => {
    setTimeout(() => {
        // Check if API calls are failing
        fetch(currentConfig.API_BASE.replace('/api', '/health'))
            .catch(error => {
                if (error.message.includes('certificate') ||
                    error.message.includes('SSL') ||
                    error.message.includes('ERR_CERT')) {
                    handleSSLError();
                }
            });
    }, 2000);
});
