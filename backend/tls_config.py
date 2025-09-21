import ssl
import os
from typing import Dict, Any

class TLSConfiguration:
    """TLS/SSL configuration for secure HTTPS communication."""

    @staticmethod
    def get_development_ssl_config() -> Dict[str, Any]:
        """Get SSL configuration for development environment."""
        return {
            "ssl_keyfile": None,
            "ssl_certfile": None,
            "ssl_ca_certs": None,
            "ssl_cert_reqs": ssl.CERT_NONE,
            "use_ssl": False
        }

    @staticmethod
    def get_production_ssl_config() -> Dict[str, Any]:
        """Get SSL configuration for production environment."""
        cert_dir = os.getenv("SSL_CERT_DIR", "/etc/ssl/certs")
        key_dir = os.getenv("SSL_KEY_DIR", "/etc/ssl/private")

        return {
            "ssl_keyfile": os.path.join(key_dir, "nightingale.key"),
            "ssl_certfile": os.path.join(cert_dir, "nightingale.crt"),
            "ssl_ca_certs": os.path.join(cert_dir, "ca-certificates.crt"),
            "ssl_cert_reqs": ssl.CERT_NONE,  # Change to CERT_REQUIRED for client cert auth
            "ssl_version": ssl.PROTOCOL_TLS_SERVER,
            "ssl_ciphers": "ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS",
            "use_ssl": True
        }

    @staticmethod
    def create_ssl_context() -> ssl.SSLContext:
        """Create SSL context with secure settings."""
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)

        # Load certificate and private key
        cert_file = os.getenv("SSL_CERT_FILE", "/etc/ssl/certs/nightingale.crt")
        key_file = os.getenv("SSL_KEY_FILE", "/etc/ssl/private/nightingale.key")

        if os.path.exists(cert_file) and os.path.exists(key_file):
            context.load_cert_chain(cert_file, key_file)

        # Security settings
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.maximum_version = ssl.TLSVersion.TLSv1_3

        # Strong cipher suites
        context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')

        # Security options
        context.options |= ssl.OP_NO_SSLv2
        context.options |= ssl.OP_NO_SSLv3
        context.options |= ssl.OP_NO_TLSv1
        context.options |= ssl.OP_NO_TLSv1_1
        context.options |= ssl.OP_SINGLE_DH_USE
        context.options |= ssl.OP_SINGLE_ECDH_USE
        context.options |= ssl.OP_NO_COMPRESSION

        return context

    @staticmethod
    def get_uvicorn_ssl_config() -> Dict[str, Any]:
        """Get SSL configuration for Uvicorn ASGI server."""
        env = os.getenv("ENVIRONMENT", "development")

        if env == "production":
            return TLSConfiguration.get_production_ssl_config()
        else:
            return TLSConfiguration.get_development_ssl_config()

    @staticmethod
    def validate_ssl_certificates() -> bool:
        """Validate SSL certificate files exist and are readable."""
        cert_file = os.getenv("SSL_CERT_FILE", "/etc/ssl/certs/nightingale.crt")
        key_file = os.getenv("SSL_KEY_FILE", "/etc/ssl/private/nightingale.key")

        if not os.path.exists(cert_file):
            print(f"SSL certificate file not found: {cert_file}")
            return False

        if not os.path.exists(key_file):
            print(f"SSL private key file not found: {key_file}")
            return False

        try:
            # Test loading the certificate and key
            context = ssl.create_default_context()
            context.load_cert_chain(cert_file, key_file)
            return True
        except Exception as e:
            print(f"SSL certificate validation failed: {e}")
            return False

# Self-signed certificate generation for development
def generate_self_signed_cert():
    """Generate self-signed certificate for development."""
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from datetime import datetime, timedelta

        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )

        # Create certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Nightingale AI"),
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ])

        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.DNSName("127.0.0.1"),
                x509.DNSName("0.0.0.0"),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256())

        # Write certificate and key files
        cert_dir = "ssl"
        os.makedirs(cert_dir, exist_ok=True)

        cert_file = os.path.join(cert_dir, "cert.pem")
        key_file = os.path.join(cert_dir, "key.pem")

        with open(cert_file, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        with open(key_file, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))

        print(f"Self-signed certificate generated:")
        print(f"Certificate: {cert_file}")
        print(f"Private key: {key_file}")

        return cert_file, key_file

    except ImportError:
        print("cryptography package required for self-signed certificate generation")
        print("Install with: pip install cryptography")
        return None, None
    except Exception as e:
        print(f"Failed to generate self-signed certificate: {e}")
        return None, None

# Production deployment configurations
class ProductionTLSConfig:
    """Production-ready TLS configurations."""

    @staticmethod
    def get_nginx_config() -> str:
        """Get Nginx configuration for TLS termination."""
        return """
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL Configuration
    ssl_certificate /etc/ssl/certs/nightingale.crt;
    ssl_certificate_key /etc/ssl/private/nightingale.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    # HSTS
    add_header Strict-Transport-Security "max-age=63072000" always;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";

    # Proxy to FastAPI
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}
"""

    @staticmethod
    def get_docker_compose_ssl() -> str:
        """Get Docker Compose configuration with SSL."""
        return """
version: '3.8'

services:
  nightingale-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./ssl:/etc/ssl:ro
      - ./data:/app/data
    environment:
      - ENVIRONMENT=production
      - SSL_CERT_FILE=/etc/ssl/cert.pem
      - SSL_KEY_FILE=/etc/ssl/key.pem
      - DB_ENCRYPTION_KEY=${DB_ENCRYPTION_KEY}
    command: uvicorn backend.main:app --host 0.0.0.0 --port 8000 --ssl-keyfile /etc/ssl/key.pem --ssl-certfile /etc/ssl/cert.pem

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/ssl:ro
    depends_on:
      - nightingale-api
"""

    @staticmethod
    def get_systemd_service() -> str:
        """Get systemd service configuration."""
        return """
[Unit]
Description=Nightingale AI Medical Assistant
After=network.target

[Service]
Type=exec
User=nightingale
Group=nightingale
WorkingDirectory=/opt/nightingale
Environment=ENVIRONMENT=production
Environment=SSL_CERT_FILE=/etc/ssl/certs/nightingale.crt
Environment=SSL_KEY_FILE=/etc/ssl/private/nightingale.key
ExecStart=/opt/nightingale/venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000 --ssl-keyfile /etc/ssl/private/nightingale.key --ssl-certfile /etc/ssl/certs/nightingale.crt
Restart=always
RestartSec=10

# Security settings
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/opt/nightingale/data

[Install]
WantedBy=multi-user.target
"""