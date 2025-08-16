FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    openssl \
    ffmpeg \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY backendRequirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r backendRequirements.txt

# Copy application code
COPY . .

# Create necessary directories and set permissions
RUN mkdir -p ssl logs chromaDb && \
    chmod 755 ssl logs chromaDb

# Generate SSL certificates if they don't exist
RUN if [ ! -f ssl/certificate.crt ]; then \
    openssl genrsa -out ssl/private.key 2048 && \
    openssl req -new -x509 -key ssl/private.key -out ssl/certificate.crt -days 365 \
    -subj "/C=IN/ST=Delhi/L=Delhi/O=AgriChat/CN=localhost"; \
    fi

# Create non-root user for security
RUN useradd -m -u 1000 agrichat

# Set proper permissions for the agrichat user
RUN chown -R agrichat:agrichat /app

USER agrichat

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000 8443

# Use uvicorn with optimized settings
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]
