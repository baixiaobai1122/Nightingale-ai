import time
import secrets
import hashlib
from typing import Dict, Optional, Tuple
from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.security.utils import get_authorization_scheme_param
from sqlalchemy.orm import Session
from starlette.middleware.base import BaseHTTPMiddleware as StarletteBaseHTTPMiddleware

from .audit import AuditLogger, get_audit_logger
from .db import get_db

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    def __init__(self, app: FastAPI):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Add security headers
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "camera=(), microphone=(), geolocation=()"
        }

        for header, value in security_headers.items():
            response.headers[header] = value

        return response

class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Rate limiting to prevent abuse and DoS attacks."""

    def __init__(self, app: FastAPI, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.client_requests: Dict[str, list] = {}

    async def dispatch(self, request: Request, call_next):
        client_ip = self._get_client_ip(request)
        current_time = time.time()

        # Clean old requests
        if client_ip in self.client_requests:
            self.client_requests[client_ip] = [
                req_time for req_time in self.client_requests[client_ip]
                if current_time - req_time < 60  # Keep last minute
            ]
        else:
            self.client_requests[client_ip] = []

        # Check rate limit
        if len(self.client_requests[client_ip]) >= self.requests_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests. Please try again later."
            )

        # Record this request
        self.client_requests[client_ip].append(current_time)

        return await call_next(request)

    def _get_client_ip(self, request: Request) -> str:
        # Check for forwarded headers (e.g., from load balancer)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

class AuditMiddleware(BaseHTTPMiddleware):
    """Comprehensive audit logging middleware."""

    def __init__(self, app: FastAPI):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Get client info
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")

        # Process request
        response = await call_next(request)

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Log important endpoints
        if self._should_audit_endpoint(request.url.path):
            try:
                # Get database session for audit logging
                db = next(get_db())
                audit_logger = get_audit_logger(db)

                # Extract user info if available (would need to decode JWT)
                user_id = self._extract_user_id(request)

                # Log the request
                audit_logger.log_event(
                    event_type="API_REQUEST",
                    action=request.method,
                    user_id=user_id,
                    resource=request.url.path,
                    request=request,
                    processing_time_ms=processing_time_ms,
                    result="SUCCESS" if response.status_code < 400 else "FAILURE",
                    details={
                        "status_code": response.status_code,
                        "client_ip": client_ip,
                        "user_agent": user_agent
                    }
                )

                db.close()
            except Exception as e:
                # Don't let audit logging break the request
                print(f"Audit logging error: {e}")

        return response

    def _get_client_ip(self, request: Request) -> str:
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _should_audit_endpoint(self, path: str) -> bool:
        """Determine if endpoint should be audited."""
        audit_patterns = [
            "/auth/",
            "/session/",
            "/asr/",
            "/summarize",
            "/review",
            "/patient/",
            "/consent/"
        ]
        return any(pattern in path for pattern in audit_patterns)

    def _extract_user_id(self, request: Request) -> Optional[int]:
        """Extract user ID from JWT token if present."""
        try:
            authorization = request.headers.get("Authorization")
            if authorization:
                scheme, token = get_authorization_scheme_param(authorization)
                if scheme.lower() == "bearer":
                    # In a real implementation, decode JWT here
                    # For now, return None
                    return None
        except Exception:
            pass
        return None

class HIPAAComplianceMiddleware(BaseHTTPMiddleware):
    """HIPAA compliance checks and PHI protection."""

    def __init__(self, app: FastAPI):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        # Pre-request checks
        if self._is_phi_endpoint(request.url.path):
            # Verify HTTPS in production
            if not self._is_secure_connection(request):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="HTTPS required for PHI endpoints"
                )

        response = await call_next(request)

        # Post-request checks
        if hasattr(response, 'body'):
            # In a real implementation, scan response for PHI leakage
            pass

        return response

    def _is_phi_endpoint(self, path: str) -> bool:
        """Check if endpoint handles PHI data."""
        phi_patterns = [
            "/patient/",
            "/session/",
            "/asr/",
            "/summarize",
            "/segments"
        ]
        return any(pattern in path for pattern in phi_patterns)

    def _is_secure_connection(self, request: Request) -> bool:
        """Check if connection is secure (HTTPS)."""
        # In development, allow HTTP
        if hasattr(request.app.state, 'development') and request.app.state.development:
            return True

        # Check various indicators of HTTPS
        return (
            request.url.scheme == "https" or
            request.headers.get("X-Forwarded-Proto") == "https" or
            request.headers.get("X-Forwarded-SSL") == "on"
        )

def add_security_middleware(app: FastAPI):
    """Add all security middleware to the FastAPI app."""

    # Add in reverse order (they wrap around each other)
    app.add_middleware(HIPAAComplianceMiddleware)
    app.add_middleware(AuditMiddleware)
    app.add_middleware(RateLimitingMiddleware, requests_per_minute=100)
    app.add_middleware(SecurityHeadersMiddleware)

# TLS Configuration for production
class TLSConfig:
    """TLS configuration for secure communications."""

    @staticmethod
    def get_ssl_context():
        """Get SSL context for production deployment."""
        import ssl

        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)

        # Load certificate and key
        context.load_cert_chain(
            certfile="/etc/ssl/certs/nightingale.crt",
            keyfile="/etc/ssl/private/nightingale.key"
        )

        # Security settings
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')

        # Optional: Client certificate verification
        # context.verify_mode = ssl.CERT_REQUIRED
        # context.load_verify_locations("/etc/ssl/certs/ca-certificates.crt")

        return context

    @staticmethod
    def get_uvicorn_ssl_config():
        """Get SSL configuration for Uvicorn server."""
        return {
            "ssl_keyfile": "/etc/ssl/private/nightingale.key",
            "ssl_certfile": "/etc/ssl/certs/nightingale.crt",
            "ssl_version": ssl.PROTOCOL_TLSv1_2,
            "ssl_cert_reqs": ssl.CERT_NONE,  # Change to ssl.CERT_REQUIRED for client certs
            "ssl_ca_certs": "/etc/ssl/certs/ca-certificates.crt"
        }

# Database encryption utilities
class DatabaseEncryption:
    """Database encryption utilities for data at rest."""

    @staticmethod
    def encrypt_sensitive_field(data: str, key: Optional[str] = None) -> str:
        """Encrypt sensitive data before storing in database."""
        from cryptography.fernet import Fernet
        import os

        # Use environment variable or generate key
        if key is None:
            key = os.getenv("DB_ENCRYPTION_KEY")
            if not key:
                # In production, this should come from a secure key management system
                key = Fernet.generate_key()

        f = Fernet(key.encode() if isinstance(key, str) else key)
        return f.encrypt(data.encode()).decode()

    @staticmethod
    def decrypt_sensitive_field(encrypted_data: str, key: Optional[str] = None) -> str:
        """Decrypt sensitive data when reading from database."""
        from cryptography.fernet import Fernet
        import os

        if key is None:
            key = os.getenv("DB_ENCRYPTION_KEY")
            if not key:
                raise ValueError("Encryption key not found")

        f = Fernet(key.encode() if isinstance(key, str) else key)
        return f.decrypt(encrypted_data.encode()).decode()

    @staticmethod
    def hash_for_indexing(data: str) -> str:
        """Create searchable hash of sensitive data."""
        # Use consistent salt for searchable hashes
        salt = "nightingale_search_salt"  # In production, use secure random salt
        return hashlib.sha256((data + salt).encode()).hexdigest()

# Session security
class SessionSecurity:
    """Secure session management utilities."""

    @staticmethod
    def generate_session_token() -> str:
        """Generate cryptographically secure session token."""
        return secrets.token_urlsafe(32)

    @staticmethod
    def generate_csrf_token() -> str:
        """Generate CSRF protection token."""
        return secrets.token_urlsafe(32)

    @staticmethod
    def validate_session_integrity(session_data: dict) -> bool:
        """Validate session data integrity."""
        # Implement session validation logic
        required_fields = ["user_id", "created_at", "last_activity"]
        return all(field in session_data for field in required_fields)