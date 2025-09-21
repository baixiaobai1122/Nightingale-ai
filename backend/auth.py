from fastapi import HTTPException, Depends, Request, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from jose import jwt, JWTError
from typing import Optional

from .models import User, Patient, Session as ConsultSession
from .db import get_db
from .security import SECRET_KEY, ALGORITHM

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# Global token store (demo only - use Redis/database in production)
TOKENS = {}

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """Extract and validate current user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # Check if token exists in our store (demo only)
        if token not in TOKENS:
            raise credentials_exception

        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.id == int(user_id)).first()
    if user is None:
        raise credentials_exception
    return user

async def verify_patient_consent(
    patient_id: int,
    db: Session = Depends(get_db)
) -> Patient:
    """Verify that patient has given consent for AI processing."""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )

    if not patient.consent_status:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Patient consent required for AI processing"
        )

    return patient

async def verify_session_access(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> ConsultSession:
    """Verify user has access to the specified session."""
    session = db.query(ConsultSession).filter(ConsultSession.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    # Check if user is authorized to access this session
    # For now, allow if user is clinician role or if patient owns the session
    if current_user.role == "clinician":
        return session
    elif current_user.role == "patient":
        # In a real system, you'd link users to patients properly
        # For demo, assume patient can access if session exists
        return session
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to access session"
        )

class ConsentGate:
    """Middleware-like dependency to ensure consent before AI processing."""

    def __init__(self, require_consent: bool = True):
        self.require_consent = require_consent

    async def __call__(
        self,
        request: Request,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
        """Check consent requirements for AI processing endpoints."""
        if not self.require_consent:
            return current_user

        # Extract patient_id or session_id from request to check consent
        path = request.url.path

        # For AI processing endpoints, ensure consent is verified
        if any(endpoint in path for endpoint in ["/asr/ingest", "/summarize", "/review"]):
            # Try to extract session_id from path or body
            if "summarize" in path or "review" in path:
                # Extract session_id from path
                try:
                    session_id = int(path.split("/")[-1]) if path.split("/")[-1].isdigit() else None
                    if session_id:
                        session = await verify_session_access(session_id, current_user, db)
                        await verify_patient_consent(session.patient_id, db)
                except (ValueError, IndexError):
                    pass  # Will be handled by individual endpoints

        return current_user