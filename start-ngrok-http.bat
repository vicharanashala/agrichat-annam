@echo off
echo Stopping any existing ngrok processes...
taskkill /f /im ngrok.exe 2>nul

echo.
echo Starting ngrok tunnel for HTTP backend (port 8000)...
echo This avoids SSL certificate issues with ngrok free tier
echo.
echo Make sure your backend is running on http://localhost:8000
echo.
pause

echo Starting ngrok...
ngrok http 8000
