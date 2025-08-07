FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y build-essential openssl && apt-get clean

# Copy requirements file first for better layer caching
COPY backendRequirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r backendRequirements.txt

# Copy the entire backend application
COPY . .

# Create SSL certificates if they don't exist
RUN mkdir -p ssl && \
    if [ ! -f ssl/certificate.crt ]; then \
    openssl genrsa -out ssl/private.key 2048 && \
    openssl req -new -x509 -key ssl/private.key -out ssl/certificate.crt -days 365 \
    -subj "/C=IN/ST=Delhi/L=Delhi/O=AgriChat/CN=192.168.1.17"; \
    fi

# Expose the ports
EXPOSE 8000 8443

# Start the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
