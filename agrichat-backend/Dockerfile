FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    openssl \
    ffmpeg \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY backendRequirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r backendRequirements.txt

COPY . .

RUN mkdir -p ssl logs chromaDb .cache/whisper && \
    chmod 755 ssl logs chromaDb .cache/whisper

# Set environment variable for Whisper cache directory
ENV WHISPER_CACHE_DIR=/app/.cache/whisper
ENV XDG_CACHE_HOME=/app/.cache

# Create whisper cache directory and set up conditional download
RUN mkdir -p /app/.cache/whisper

# Only download Whisper model if not already cached
RUN if [ ! -f /app/.cache/whisper/base.pt ]; then \
        echo "Downloading Whisper base model..."; \
        python -c "import whisper; whisper.load_model('base')"; \
    else \
        echo "Whisper base model already cached, skipping download"; \
    fi

RUN if [ ! -f ssl/certificate.crt ]; then \
    openssl genrsa -out ssl/private.key 2048 && \
    openssl req -new -x509 -key ssl/private.key -out ssl/certificate.crt -days 365 \
    -subj "/C=IN/ST=Delhi/L=Delhi/O=AgriChat/CN=localhost"; \
    fi

RUN useradd -m -u 1000 agrichat

RUN chown -R agrichat:agrichat /app

# Set environment variable for Whisper cache directory
ENV XDG_CACHE_HOME=/app/.cache

USER agrichat

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000 8443

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]
