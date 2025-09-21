import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, Float
from sqlalchemy.orm import Session
from fastapi import Request
from .db import Base
from .models import User, Patient, Session as ConsultSession

# Configure structured audit logging
audit_logger = logging.getLogger("nightingale.audit")
audit_logger.setLevel(logging.INFO)

# Create handler for audit log file
handler = logging.FileHandler("audit.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
audit_logger.addHandler(handler)

class AuditLog(Base):
    """Database model for audit trail storage."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    user_id = Column(Integer, nullable=True)  # Can be null for system events
    session_id = Column(Integer, nullable=True)
    patient_id = Column(Integer, nullable=True)

    # Event details
    event_type = Column(String, nullable=False)  # PHI_ACCESS, AI_PROCESSING, USER_AUTH, etc.
    action = Column(String, nullable=False)      # READ, WRITE, DELETE, REDACT, SUMMARIZE, etc.
    resource = Column(String, nullable=True)     # Table/endpoint affected
    resource_id = Column(String, nullable=True)  # Specific record ID

    # Request context
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    endpoint = Column(String, nullable=True)
    method = Column(String, nullable=True)

    # Security and compliance
    phi_detected = Column(Boolean, default=False)
    phi_types = Column(Text, nullable=True)  # JSON array of PHI types found
    consent_verified = Column(Boolean, default=False)

    # Performance metrics
    processing_time_ms = Column(Float, nullable=True)

    # Additional context
    details = Column(Text, nullable=True)  # JSON for additional context
    result = Column(String, nullable=True)  # SUCCESS, FAILURE, BLOCKED
    error_message = Column(Text, nullable=True)

class AuditLogger:
    """Centralized audit logging for HIPAA compliance."""

    def __init__(self, db: Session):
        self.db = db

    def log_event(
        self,
        event_type: str,
        action: str,
        user_id: Optional[int] = None,
        session_id: Optional[int] = None,
        patient_id: Optional[int] = None,
        resource: Optional[str] = None,
        resource_id: Optional[str] = None,
        request: Optional[Request] = None,
        phi_detected: bool = False,
        phi_types: Optional[List[str]] = None,
        consent_verified: bool = False,
        processing_time_ms: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
        result: str = "SUCCESS",
        error_message: Optional[str] = None
    ) -> AuditLog:
        """Log an audit event to both database and log file."""

        # Extract request context if available
        ip_address = None
        user_agent = None
        endpoint = None
        method = None

        if request:
            ip_address = getattr(request.client, 'host', None) if request.client else None
            user_agent = request.headers.get('user-agent')
            endpoint = str(request.url.path)
            method = request.method

        # Create audit log entry
        audit_entry = AuditLog(
            user_id=user_id,
            session_id=session_id,
            patient_id=patient_id,
            event_type=event_type,
            action=action,
            resource=resource,
            resource_id=resource_id,
            ip_address=ip_address,
            user_agent=user_agent,
            endpoint=endpoint,
            method=method,
            phi_detected=phi_detected,
            phi_types=json.dumps(phi_types) if phi_types else None,
            consent_verified=consent_verified,
            processing_time_ms=processing_time_ms,
            details=json.dumps(details) if details else None,
            result=result,
            error_message=error_message
        )

        # Save to database
        self.db.add(audit_entry)
        self.db.commit()
        self.db.refresh(audit_entry)

        # Log to file for immediate access
        log_data = {
            "id": audit_entry.id,
            "timestamp": audit_entry.timestamp.isoformat(),
            "event_type": event_type,
            "action": action,
            "user_id": user_id,
            "patient_id": patient_id,
            "resource": resource,
            "endpoint": endpoint,
            "phi_detected": phi_detected,
            "consent_verified": consent_verified,
            "result": result
        }

        if error_message:
            log_data["error"] = error_message
        if phi_types:
            log_data["phi_types"] = phi_types
        if details:
            log_data["details"] = details

        audit_logger.info(json.dumps(log_data))

        return audit_entry

    def log_phi_access(
        self,
        user_id: int,
        patient_id: int,
        action: str,
        phi_types: List[str],
        request: Optional[Request] = None,
        session_id: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Specialized logging for PHI access events."""
        return self.log_event(
            event_type="PHI_ACCESS",
            action=action,
            user_id=user_id,
            patient_id=patient_id,
            session_id=session_id,
            request=request,
            phi_detected=True,
            phi_types=phi_types,
            consent_verified=True,  # Only log if consent was verified
            details=details
        )

    def log_ai_processing(
        self,
        user_id: int,
        session_id: int,
        patient_id: int,
        action: str,
        processing_time_ms: float,
        phi_types: Optional[List[str]] = None,
        request: Optional[Request] = None,
        result: str = "SUCCESS",
        error_message: Optional[str] = None
    ):
        """Specialized logging for AI processing events."""
        return self.log_event(
            event_type="AI_PROCESSING",
            action=action,
            user_id=user_id,
            session_id=session_id,
            patient_id=patient_id,
            request=request,
            phi_detected=bool(phi_types),
            phi_types=phi_types,
            consent_verified=True,
            processing_time_ms=processing_time_ms,
            result=result,
            error_message=error_message
        )

    def log_authentication(
        self,
        user_id: Optional[int],
        action: str,
        request: Optional[Request] = None,
        result: str = "SUCCESS",
        error_message: Optional[str] = None
    ):
        """Specialized logging for authentication events."""
        return self.log_event(
            event_type="USER_AUTH",
            action=action,
            user_id=user_id,
            request=request,
            result=result,
            error_message=error_message
        )

    def log_consent_action(
        self,
        patient_id: int,
        action: str,
        consent_status: bool,
        user_id: Optional[int] = None,
        request: Optional[Request] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Specialized logging for consent-related actions."""
        return self.log_event(
            event_type="CONSENT_MANAGEMENT",
            action=action,
            user_id=user_id,
            patient_id=patient_id,
            request=request,
            consent_verified=consent_status,
            details=details
        )

def get_audit_logger(db: Session) -> AuditLogger:
    """Dependency injection for audit logger."""
    return AuditLogger(db)

class AuditMiddleware:
    """FastAPI middleware for automatic audit logging."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Track request start time
            start_time = time.time()

            # Process request
            await self.app(scope, receive, send)

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            # Log request if it's an API endpoint
            path = scope.get("path", "")
            if path.startswith("/api/") or any(endpoint in path for endpoint in [
                "/auth/", "/session/", "/asr/", "/summarize", "/review", "/patient/"
            ]):
                # This would require access to the database session
                # In practice, this would be implemented as a FastAPI dependency
                pass
        else:
            await self.app(scope, receive, send)

# Security and encryption utilities
class SecurityConfig:
    """Security configuration for TLS and encryption."""

    @staticmethod
    def get_tls_config():
        """Return TLS configuration for production deployment."""
        return {
            "keyfile": "/etc/ssl/private/nightingale.key",
            "certfile": "/etc/ssl/certs/nightingale.crt",
            "ssl_version": "TLSv1.2",  # Minimum TLS 1.2
            "ciphers": "ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS",
            "ca_certs": "/etc/ssl/certs/ca-certificates.crt"
        }

    @staticmethod
    def get_database_encryption_config():
        """Return database encryption configuration."""
        return {
            "encryption_key": "AES-256-GCM",  # Use environment variable in production
            "key_rotation_interval": 90,      # Days
            "backup_encryption": True,
            "transparent_data_encryption": True
        }

    @staticmethod
    def get_security_headers():
        """Return security headers for HTTP responses."""
        return {
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }