#!/bin/bash

# Create SSL directory
mkdir -p ssl

# Generate private key
openssl genrsa -out ssl/private.key 2048

# Generate certificate signing request
openssl req -new -key ssl/private.key -out ssl/certificate.csr -subj "/C=IN/ST=Delhi/L=Delhi/O=AgriChat/CN=192.168.1.17"

# Generate self-signed certificate
openssl x509 -req -days 365 -in ssl/certificate.csr -signkey ssl/private.key -out ssl/certificate.crt

# Create combined certificate file
cat ssl/certificate.crt ssl/private.key > ssl/combined.pem

echo "SSL certificates generated successfully!"
echo "Certificate: ssl/certificate.crt"
echo "Private Key: ssl/private.key"
echo "Combined: ssl/combined.pem"
