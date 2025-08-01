# Stop any existing ngrok processes
Write-Host "Stopping existing ngrok processes..." -ForegroundColor Yellow
Get-Process ngrok -ErrorAction SilentlyContinue | Stop-Process -Force

# Wait a moment for processes to stop
Start-Sleep 2

# Start ngrok with HTTPS forwarding
Write-Host "Starting ngrok with HTTPS forwarding..." -ForegroundColor Green
Write-Host "Forwarding: https://yoururl.ngrok.io -> https://localhost:8443" -ForegroundColor Cyan

# Start ngrok in background
Start-Process -NoNewWindow "ngrok" -ArgumentList "http", "https://localhost:8443"

# Wait for ngrok to start
Start-Sleep 5

# Get the new ngrok URL
try {
    $tunnels = Invoke-WebRequest -Uri "http://127.0.0.1:4040/api/tunnels" | ConvertFrom-Json
    $publicUrl = $tunnels.tunnels | Where-Object { $_.proto -eq "https" } | Select-Object -ExpandProperty public_url
    Write-Host "New ngrok URL: $publicUrl" -ForegroundColor Green
    Write-Host "Update your config.js with: API_BASE: '$publicUrl/api'" -ForegroundColor Cyan
} catch {
    Write-Host "Could not get ngrok URL automatically. Check http://127.0.0.1:4040" -ForegroundColor Red
}
