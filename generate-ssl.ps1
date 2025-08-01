# PowerShell script to generate SSL certificates for local development

# Create SSL directory
New-Item -ItemType Directory -Force -Path "ssl"

# Check if OpenSSL is available
try {
    openssl version
    Write-Host "OpenSSL found, generating certificates..."
    
    # Generate private key
    openssl genrsa -out ssl/private.key 2048
    
    # Generate certificate signing request
    openssl req -new -key ssl/private.key -out ssl/certificate.csr -subj "/C=IN/ST=Delhi/L=Delhi/O=AgriChat/CN=192.168.1.17"
    
    # Generate self-signed certificate
    openssl x509 -req -days 365 -in ssl/certificate.csr -signkey ssl/private.key -out ssl/certificate.crt
    
    Write-Host "SSL certificates generated successfully!"
    Write-Host "Certificate: ssl/certificate.crt"
    Write-Host "Private Key: ssl/private.key"
    
} catch {
    Write-Host "OpenSSL not found. Please install OpenSSL or use the manual approach."
    Write-Host "Alternative: Download Git for Windows which includes OpenSSL"
    Write-Host "Or use: choco install openssl (if you have Chocolatey)"
}
