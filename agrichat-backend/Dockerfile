FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential openssl && apt-get clean

COPY backendRequirements.txt .

RUN pip install --upgrade pip && pip install -r backendRequirements.txt

COPY . .

RUN mkdir -p ssl && \
    if [ ! -f ssl/certificate.crt ]; then \
    openssl genrsa -out ssl/private.key 2048 && \
    openssl req -new -x509 -key ssl/private.key -out ssl/certificate.crt -days 365 \
    -subj "/C=IN/ST=Delhi/L=Delhi/O=AgriChat/CN=192.168.1.17"; \
    fi

EXPOSE 8000 8443

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
