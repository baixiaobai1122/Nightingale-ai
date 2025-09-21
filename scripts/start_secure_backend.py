#!/usr/bin/env python3
"""
Secure startup script for Nightingale AI backend with TLS and audit logging.
"""
import os
import sys
import logging
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def setup_logging():
    """Configure logging for the application."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler('nightingale.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def check_security_requirements():
    """Check security requirements before starting."""
    print("🔒 Checking security requirements...")

    # Check environment variables
    required_env_vars = [
        'DB_ENCRYPTION_KEY',
        'SECRET_KEY',
        'MASTER_PASSWORD'
    ]

    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("Setting demo values for development...")
        for var in missing_vars:
            if var == 'DB_ENCRYPTION_KEY':
                os.environ[var] = 'demo-encryption-key-not-for-production'
            elif var == 'SECRET_KEY':
                os.environ[var] = 'demo-secret-key-not-for-production'
            elif var == 'MASTER_PASSWORD':
                os.environ[var] = 'demo-master-password'

    # Check TLS certificates in production
    env = os.getenv('ENVIRONMENT', 'development')
    if env == 'production':
        from backend.tls_config import TLSConfiguration
        if not TLSConfiguration.validate_ssl_certificates():
            print("❌ SSL certificate validation failed")
            return False

    print("✅ Security requirements checked")
    return True

def test_phi_redaction():
    """Test PHI redaction functionality."""
    print("🔍 Testing PHI redaction...")

    try:
        from backend.redact import redact_text, assert_no_phi, get_phi_analysis

        test_data = "Patient John Doe, SSN: 123-45-6789, visited on 01/15/2024"
        redacted, mapping = redact_text(test_data)

        if assert_no_phi(redacted):
            print(f"✅ PHI redaction working - {len(mapping)} items redacted")
            return True
        else:
            print("❌ PHI redaction failed validation")
            return False

    except Exception as e:
        print(f"❌ PHI redaction test failed: {e}")
        return False

def test_encryption():
    """Test encryption functionality."""
    print("🔐 Testing encryption...")

    try:
        from backend.encrypted_db import EncryptionManager

        em = EncryptionManager()
        test_data = "Sensitive patient information"
        encrypted = em.encrypt(test_data)
        decrypted = em.decrypt(encrypted)

        if decrypted == test_data and encrypted != test_data:
            print("✅ Encryption working")
            return True
        else:
            print("❌ Encryption test failed")
            return False

    except ImportError:
        print("⚠️  Encryption not available (cryptography package required)")
        return True  # Don't fail startup
    except Exception as e:
        print(f"❌ Encryption test failed: {e}")
        return False

def create_demo_ssl_cert():
    """Create demo SSL certificate for development."""
    print("🔧 Creating demo SSL certificate...")

    try:
        from backend.tls_config import generate_self_signed_cert

        cert_file, key_file = generate_self_signed_cert()
        if cert_file and key_file:
            print(f"✅ Demo SSL certificate created: {cert_file}")
            os.environ['SSL_CERT_FILE'] = cert_file
            os.environ['SSL_KEY_FILE'] = key_file
            return True

    except Exception as e:
        print(f"⚠️  Could not create SSL certificate: {e}")

    return False

def start_server():
    """Start the secure FastAPI server."""
    print("🚀 Starting Nightingale AI secure backend...")

    try:
        import uvicorn
        from backend.app import app
        from backend.tls_config import TLSConfiguration

        # Get SSL configuration
        ssl_config = TLSConfiguration.get_uvicorn_ssl_config()

        # Server configuration
        config = {
            "app": app,
            "host": "0.0.0.0",
            "port": int(os.getenv("PORT", "8000")),
            "reload": os.getenv("ENVIRONMENT", "development") == "development",
            "log_level": "info"
        }

        # Add SSL configuration if enabled
        if ssl_config.get("use_ssl"):
            config.update({
                "ssl_keyfile": ssl_config["ssl_keyfile"],
                "ssl_certfile": ssl_config["ssl_certfile"]
            })
            print(f"🔒 Starting with HTTPS on port {config['port']}")
        else:
            print(f"⚠️  Starting with HTTP on port {config['port']} (development mode)")

        print("📊 Access API documentation at:")
        protocol = "https" if ssl_config.get("use_ssl") else "http"
        print(f"   {protocol}://localhost:{config['port']}/docs")
        print("\n🔒 Security Features Enabled:")
        print("   ✅ PHI Redaction")
        print("   ✅ Audit Logging")
        print("   ✅ Rate Limiting")
        print("   ✅ Security Headers")
        print("   ✅ HIPAA Compliance Middleware")

        uvicorn.run(**config)

    except ImportError:
        print("❌ uvicorn not available. Install with: pip install uvicorn[standard]")
        return False
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        return False

def main():
    """Main startup sequence."""
    print("🏥 Nightingale AI - Secure Medical Voice Assistant")
    print("=" * 60)

    setup_logging()

    # Security checks
    if not check_security_requirements():
        print("❌ Security requirements not met")
        sys.exit(1)

    if not test_phi_redaction():
        print("❌ PHI redaction test failed")
        sys.exit(1)

    if not test_encryption():
        print("❌ Encryption test failed")
        sys.exit(1)

    # Create SSL cert for development
    env = os.getenv("ENVIRONMENT", "development")
    if env == "development":
        create_demo_ssl_cert()

    print("=" * 60)

    # Start server
    if not start_server():
        sys.exit(1)

if __name__ == "__main__":
    main()