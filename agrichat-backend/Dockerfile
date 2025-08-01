FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential openssl && apt-get clean

COPY backendRequirements.txt .
RUN pip install --upgrade pip && pip install -r backendRequirements.txt

COPY . .

# Create SSL directory and generate certificates if they don't exist
RUN mkdir -p ssl && \
    if [ ! -f ssl/certificate.crt ]; then \
    openssl genrsa -out ssl/private.key 2048 && \
    openssl req -new -key ssl/private.key -out ssl/certificate.csr -subj "/C=IN/ST=Delhi/L=Delhi/O=AgriChat/CN=192.168.1.17" && \
    openssl x509 -req -days 365 -in ssl/certificate.csr -signkey ssl/private.key -out ssl/certificate.crt; \
    fi

EXPOSE 8000 8443

# Default to HTTP, can be overridden for HTTPS
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]