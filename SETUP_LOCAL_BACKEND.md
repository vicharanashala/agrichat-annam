# Local Backend + Vercel Frontend Setup Guide

## Overview
- **Backend**: Running locally on Docker (your computer)
- **Frontend**: Deployed on Vercel
- **Database**: Local MongoDB in Docker

## Setup Steps

### 1. Backend Setup (Local Docker)

1. **Configure Environment Variables**
   ```bash
   # Edit .env file and add your API keys
   GOOGLE_API_KEY=your_actual_google_api_key
   MONGO_URI=mongodb://mongodb:27017/agrichat
   ```

2. **Start Backend Services**
   ```bash
   # Build and start backend + database
   docker-compose up -d backend mongodb
   
   # Check if services are running
   docker-compose ps
   ```

3. **Make Backend Publicly Accessible**
   
   **Option A: Use ngrok (Recommended for testing)**
   ```bash
   # Install ngrok from https://ngrok.com/
   ngrok http 8000
   # Copy the https URL (e.g., https://abc123.ngrok.io)
   ```
   
   **Option B: Use your public IP (if you have static IP)**
   ```bash
   # Find your public IP
   curl ifconfig.me
   # Use http://YOUR_PUBLIC_IP:8000/api
   # Make sure port 8000 is open in your router/firewall
   ```

4. **Update Frontend Configuration**
   Edit `agrichat-frontend/config.js`:
   ```javascript
   production: {
       API_BASE: "https://your-ngrok-url.ngrok.io/api"
       // or API_BASE: "http://YOUR_PUBLIC_IP:8000/api"
   }
   ```

### 2. Frontend Setup (Vercel)

1. **Deploy to Vercel**
   ```bash
   # If you haven't already, install Vercel CLI
   npm i -g vercel
   
   # In the agrichat-frontend directory
   cd agrichat-frontend
   vercel --prod
   ```

2. **Alternative: GitHub Integration**
   - Push your code to GitHub
   - Connect your repo to Vercel
   - Set build settings:
     - Build Command: (leave empty)
     - Output Directory: (leave empty)
     - Install Command: (leave empty)

### 3. Testing

1. **Test Backend Locally**
   ```bash
   curl http://localhost:8000/
   # Should return: {"message": "AgriChat backend is running."}
   ```

2. **Test Backend Publicly**
   ```bash
   curl https://your-ngrok-url.ngrok.io/
   # Should return the same message
   ```

3. **Test Frontend**
   - Visit your Vercel URL
   - Try creating a new chat session
   - Check browser console for any CORS errors

## Troubleshooting

### CORS Issues
If you see CORS errors:
1. Make sure your Vercel domain is added to the CORS origins in `app.py`
2. Restart the backend after making changes

### ngrok Session Expired
- Free ngrok URLs expire after 8 hours
- Restart ngrok and update the frontend config
- Consider ngrok paid plans for persistent URLs

### Backend Not Accessible
1. Check if Docker containers are running: `docker-compose ps`
2. Check firewall settings for port 8000
3. Verify your router port forwarding (if using public IP)

### Frontend Build Issues
1. Make sure `config.js` is in the same directory as `index.html`
2. Verify the script tags are in the correct order in `index.html`

## Production Considerations

### Security
- Use HTTPS for production (ngrok provides this automatically)
- Restrict CORS origins to your specific Vercel domain
- Use environment variables for sensitive data

### Reliability
- Consider using a cloud database (MongoDB Atlas) instead of local MongoDB
- Set up automatic Docker container restart policies
- Monitor your backend uptime

### Scaling
- If you need better uptime, consider deploying backend to cloud services
- Use a reverse proxy (nginx) for better performance
- Consider load balancing for high traffic
