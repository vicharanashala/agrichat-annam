@echo off
echo Starting ngrok tunnel for AgriChat backend...
echo This will create a public HTTPS URL for your local backend
echo.
echo Make sure your backend is running on https://localhost:8443
echo.
pause
ngrok http https://localhost:8443
