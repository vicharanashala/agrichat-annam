# Setup ngrok with proper authentication for SSL certificates
# This script will help you set up ngrok authentication to get trusted SSL certificates

Write-Host "=== ngrok SSL Certificate Fix ===" -ForegroundColor Green
Write-Host ""

# Check if ngrok is running and stop it
$ngrokProcess = Get-Process -Name "ngrok" -ErrorAction SilentlyContinue
if ($ngrokProcess) {
    Write-Host "Stopping existing ngrok processes..." -ForegroundColor Yellow
    taskkill /f /im ngrok.exe
    Start-Sleep -Seconds 2
}

Write-Host "Step 1: Sign up for ngrok free account" -ForegroundColor Cyan
Write-Host "1. Go to: https://ngrok.com"
Write-Host "2. Sign up for a free account"
Write-Host "3. Go to 'Getting Started' > 'Your Authtoken'"
Write-Host "4. Copy your authtoken"
Write-Host ""

$authToken = Read-Host "Enter your ngrok authtoken (or press Enter to skip)"

if ($authToken -and $authToken.Trim() -ne "") {
    Write-Host "Setting up ngrok authentication..." -ForegroundColor Yellow
    & ngrok config add-authtoken $authToken
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ ngrok authentication successful!" -ForegroundColor Green
        
        # Start ngrok with authentication (this will provide trusted SSL certificates)
        Write-Host "Starting authenticated ngrok tunnel..." -ForegroundColor Yellow
        Start-Process -FilePath "ngrok" -ArgumentList "http", "8000", "--host-header=rewrite" -WindowStyle Minimized
        
        Start-Sleep -Seconds 5
        
        # Get the new ngrok URL
        try {
            $ngrokStatus = Invoke-RestMethod -Uri "http://localhost:4040/api/tunnels" -ErrorAction Stop
            $httpsUrl = ($ngrokStatus.tunnels | Where-Object { $_.proto -eq "https" }).public_url
            $httpUrl = ($ngrokStatus.tunnels | Where-Object { $_.proto -eq "http" }).public_url
            
            Write-Host "✅ New ngrok URLs:" -ForegroundColor Green
            Write-Host "HTTPS (with trusted SSL): $httpsUrl" -ForegroundColor Green
            Write-Host "HTTP (fallback): $httpUrl" -ForegroundColor Yellow
            Write-Host ""
            
            # Extract the subdomain for updating config
            if ($httpsUrl -match "https://([^.]+)\.ngrok-free\.app") {
                $newSubdomain = $matches[1]
                Write-Host "New subdomain: $newSubdomain" -ForegroundColor Cyan
                Write-Host ""
                Write-Host "Now updating your config.js to use the new HTTPS URL..." -ForegroundColor Yellow
                
                # Read current config.js
                $configPath = "c:\Users\dledlab\agrichat-annam\agrichat-frontend\config.js"
                $configContent = Get-Content $configPath -Raw
                
                # Update with new HTTPS URL (authenticated ngrok should have trusted SSL)
                $newConfigContent = $configContent -replace 'API_BASE: "http://[^"]+\.ngrok-free\.app/api"', "API_BASE: `"$httpsUrl/api`""
                $newConfigContent = $newConfigContent -replace 'API_BASE: "https://[^"]+\.ngrok-free\.app/api"', "API_BASE: `"$httpsUrl/api`""
                
                Set-Content -Path $configPath -Value $newConfigContent
                
                Write-Host "✅ config.js updated with authenticated HTTPS URL" -ForegroundColor Green
                Write-Host ""
                Write-Host "Deploying to Vercel..." -ForegroundColor Yellow
                
                Set-Location "c:\Users\dledlab\agrichat-annam"
                git add .
                git commit -m "Update to authenticated ngrok HTTPS URL with trusted SSL certificates"
                git push
                
                Write-Host "✅ Deployed to Vercel!" -ForegroundColor Green
                Write-Host ""
                Write-Host "🎉 SSL certificate issues should now be resolved!" -ForegroundColor Green
                Write-Host "Wait 2-3 minutes for Vercel deployment, then test your app." -ForegroundColor Cyan
            }
        }
        catch {
            Write-Host "❌ Could not get ngrok status. Please check if ngrok started correctly." -ForegroundColor Red
        }
    }
    else {
        Write-Host "❌ Failed to authenticate ngrok. Please check your token." -ForegroundColor Red
    }
}
else {
    Write-Host "Skipping authentication setup." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Alternative: Using HTTP endpoint (may cause mixed content warnings)" -ForegroundColor Yellow
    
    # Start ngrok without auth
    Start-Process -FilePath "ngrok" -ArgumentList "http", "8000", "--host-header=rewrite" -WindowStyle Minimized
    Start-Sleep -Seconds 5
    
    try {
        $ngrokStatus = Invoke-RestMethod -Uri "http://localhost:4040/api/tunnels" -ErrorAction Stop
        $httpUrl = ($ngrokStatus.tunnels | Where-Object { $_.proto -eq "http" }).public_url
        
        Write-Host "HTTP URL: $httpUrl" -ForegroundColor Yellow
        
        if ($httpUrl -match "http://([^.]+)\.ngrok-free\.app") {
            $newSubdomain = $matches[1]
            
            # Update config.js with HTTP URL
            $configPath = "c:\Users\dledlab\agrichat-annam\agrichat-frontend\config.js"
            $configContent = Get-Content $configPath -Raw
            $newConfigContent = $configContent -replace 'API_BASE: "https?://[^"]+\.ngrok-free\.app/api"', "API_BASE: `"$httpUrl/api`""
            Set-Content -Path $configPath -Value $newConfigContent
            
            Write-Host "✅ config.js updated with HTTP URL" -ForegroundColor Green
            
            Set-Location "c:\Users\dledlab\agrichat-annam"
            git add .
            git commit -m "Update to HTTP ngrok URL to avoid SSL certificate issues"
            git push
            
            Write-Host "✅ Deployed! Note: You may see mixed content warnings." -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "❌ Could not get ngrok status." -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "=== Summary ===" -ForegroundColor Green
Write-Host "• Recommended: Use ngrok authentication for trusted SSL certificates"
Write-Host "• Alternative: HTTP URLs may work but cause mixed content warnings"
Write-Host "• Fallback: Use Cloudflare Tunnel for always-trusted SSL"
