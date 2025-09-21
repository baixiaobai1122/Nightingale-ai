import time
from fastapi import FastAPI, Depends, HTTPException, status, Body, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from datetime import timedelta
from pydantic import BaseModel

from .db import Base, engine, get_db
from .models import User, Patient, Session as ConsultSession, Segment, Summary
from .security import verify_password, get_password_hash, create_access_token
from .redact import redact_text, assert_no_phi, get_phi_analysis
from .summarize import make_dual_summaries
from .auth import get_current_user, verify_patient_consent, verify_session_access, ConsentGate, TOKENS
from .audit import AuditLog, AuditLogger, get_audit_logger
from .security_middleware import add_security_middleware
from .persistent_memory import (
    PersistentMemoryManager, ConversationSegment, SessionSnapshot,
    SegmentStatus, SessionState
)

# Create all tables including audit logs
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Nightingale MVP",
    version="0.3.0",
    description="HIPAA-compliant medical AI assistant with comprehensive security",
    docs_url="/docs",  # Keep docs for development
    redoc_url="/redoc"
)

# Add security middleware
add_security_middleware(app)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# --------------------
# Auth
# --------------------
@app.post("/auth/signup")
def signup(
    name: str = Body(...),
    password: str = Body(...),
    db: Session = Depends(get_db)
):
    if db.query(User).filter(User.name == name).first():
        raise HTTPException(400, "user exists")
    u = User(name=name, pwd_hash=get_password_hash(password), role="clinician")
    db.add(u)
    db.commit()
    db.refresh(u)
    return {"id": u.id, "name": u.name}


@app.post("/auth/login")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    request: Request,
    db: Session = Depends(get_db)
):
    audit_logger = get_audit_logger(db)
    start_time = time.time()

    try:
        u = db.query(User).filter(User.name == form_data.username).first()
        if not u or not verify_password(form_data.password, u.pwd_hash):
            # Log failed authentication
            audit_logger.log_authentication(
                user_id=None,
                action="LOGIN_FAILED",
                request=request,
                result="FAILURE",
                error_message="Invalid credentials"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="bad creds"
            )

        token = create_access_token({"sub": str(u.id)})
        TOKENS[token] = u.id

        # Log successful authentication
        audit_logger.log_authentication(
            user_id=u.id,
            action="LOGIN_SUCCESS",
            request=request,
            result="SUCCESS"
        )

        return {"access_token": token, "token_type": "bearer"}

    except HTTPException:
        raise
    except Exception as e:
        # Log system error
        audit_logger.log_authentication(
            user_id=None,
            action="LOGIN_ERROR",
            request=request,
            result="FAILURE",
            error_message=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication system error"
        )


# --------------------
# Consent
# --------------------
@app.post("/consent/record")
def record_consent(
    patient_name: str = Body(...),
    consent: bool = Body(...),
    request: Request,
    db: Session = Depends(get_db)
):
    audit_logger = get_audit_logger(db)

    try:
        # Redact patient name before storage
        redacted_name, mapping = redact_text(patient_name)

        # Create patient record
        p = Patient(name_redacted=redacted_name, consent_status=consent)
        db.add(p)
        db.commit()
        db.refresh(p)

        # Log consent action
        audit_logger.log_consent_action(
            patient_id=p.id,
            action="CONSENT_RECORDED",
            consent_status=consent,
            request=request,
            details={
                "phi_redacted": bool(mapping),
                "consent_version": "v1"
            }
        )

        return {"patient_id": p.id, "consent": consent}

    except Exception as e:
        audit_logger.log_event(
            event_type="CONSENT_MANAGEMENT",
            action="CONSENT_ERROR",
            request=request,
            result="FAILURE",
            error_message=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Consent recording error"
        )


# --------------------
# Session (Pydantic JSON body)
# --------------------
class SessionStartReq(BaseModel):
    patient_id: int
    title: Optional[str] = None
    description: Optional[str] = None
    auto_summarize: bool = True

class SegmentAppendReq(BaseModel):
    text: str
    speaker: str = "patient"  # patient|doctor|system
    start_ms: int = 0
    end_ms: int = 0
    edit_reason: Optional[str] = None

class SegmentEditReq(BaseModel):
    new_text: str
    edit_reason: str
    edit_note: Optional[str] = None

class SessionUpdateReq(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    auto_summarize: Optional[bool] = None

class SnapshotCreateReq(BaseModel):
    name: str


@app.post("/session/start")
def start_session(
    req: SessionStartReq,
    current_user: User = Depends(get_current_user),
    request: Request,
    db: Session = Depends(get_db)
):
    audit_logger = get_audit_logger(db)

    try:
        p = db.query(Patient).get(req.patient_id)
        if not p or not p.consent_status:
            raise HTTPException(403, "consent required")

        # Create enhanced session
        s = ConsultSession(
            patient_id=req.patient_id,
            title=req.title,
            description=req.description,
            auto_summarize=req.auto_summarize,
            status="active"
        )
        db.add(s)
        db.commit()
        db.refresh(s)

        # Audit log
        audit_logger.log_event(
            event_type="SESSION_MANAGEMENT",
            action="SESSION_START",
            user_id=current_user.id,
            session_id=s.id,
            patient_id=req.patient_id,
            request=request,
            details={
                "title": req.title,
                "auto_summarize": req.auto_summarize
            }
        )

        return {
            "session_id": s.id,
            "title": s.title,
            "status": s.status,
            "auto_summarize": s.auto_summarize
        }

    except Exception as e:
        audit_logger.log_event(
            event_type="SESSION_MANAGEMENT",
            action="SESSION_START_ERROR",
            user_id=current_user.id,
            request=request,
            result="FAILURE",
            error_message=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Session creation error"
        )


# --------------------
# ASR ingest (text-based stub)
# --------------------
@app.post("/asr/ingest")
def asr_ingest(
    session_id: int = Body(...),
    segments: List[dict] = Body(...),
    current_user: User = Depends(ConsentGate()),
    request: Request,
    db: Session = Depends(get_db)
):
    audit_logger = get_audit_logger(db)
    start_time = time.time()

    try:
        s = db.query(ConsultSession).get(session_id)
        if not s:
            raise HTTPException(404, "session not found")

        created: List[int] = []
        phi_types_detected = set()

        # 连续递增 span_idx，避免重复编号
        current_max = (
            db.query(func.max(Segment.span_idx))
            .filter(Segment.session_id == session_id)
            .scalar()
            or 0
        )

        for i, seg in enumerate(segments, start=1):
            original_text = seg.get("text", "")

            # Analyze PHI before redaction
            phi_analysis = get_phi_analysis(original_text)
            phi_types_detected.update(phi_analysis.keys())

            # Redact PHI
            red_txt, mapping = redact_text(original_text)
            if not assert_no_phi(red_txt):
                audit_logger.log_event(
                    event_type="PHI_ACCESS",
                    action="REDACTION_FAILED",
                    user_id=current_user.id,
                    session_id=session_id,
                    patient_id=s.patient_id,
                    request=request,
                    result="FAILURE",
                    error_message="PHI redaction validation failed"
                )
                raise HTTPException(400, "redaction failed")

            new_idx = current_max + i
            rec = Segment(
                session_id=session_id,
                start_ms=seg.get("start_ms", 0),
                end_ms=seg.get("end_ms", 0),
                text_redacted=red_txt,
                span_idx=new_idx,
            )
            db.add(rec)
            created.append(new_idx)

        db.commit()

        # Log successful PHI processing
        processing_time_ms = (time.time() - start_time) * 1000
        audit_logger.log_phi_access(
            user_id=current_user.id,
            patient_id=s.patient_id,
            action="ASR_INGEST",
            phi_types=list(phi_types_detected),
            request=request,
            session_id=session_id,
            details={
                "segments_processed": len(segments),
                "segments_created": len(created),
                "processing_time_ms": processing_time_ms
            }
        )

        return {"ingested_spans": created}

    except HTTPException:
        raise
    except Exception as e:
        audit_logger.log_event(
            event_type="ASR_PROCESSING",
            action="INGEST_ERROR",
            user_id=current_user.id,
            session_id=session_id,
            request=request,
            result="FAILURE",
            error_message=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ASR ingestion error"
        )


# --------------------
# Summarize
# --------------------
@app.post("/summarize/{session_id}")
def summarize(
    session_id: int,
    current_user: User = Depends(ConsentGate()),
    request: Request,
    db: Session = Depends(get_db)
):
    audit_logger = get_audit_logger(db)
    start_time = time.time()

    try:
        # Get session for patient_id
        session = db.query(ConsultSession).filter(ConsultSession.id == session_id).first()
        if not session:
            raise HTTPException(404, "session not found")

        segs = (
            db.query(Segment)
            .filter(Segment.session_id == session_id)
            .order_by(Segment.span_idx)
            .all()
        )
        if not segs:
            raise HTTPException(404, "no segments")

        spans = [(s.span_idx, s.text_redacted) for s in segs]

        # Generate AI summaries
        clin, pat = make_dual_summaries(spans)

        # 去重但保持顺序的 provenance_ids
        seen = set()
        ordered_ids: List[str] = []
        for s in segs:
            sid = str(s.span_idx)
            if sid not in seen:
                seen.add(sid)
                ordered_ids.append(sid)
        span_ids = ",".join(ordered_ids)

        s1 = Summary(
            session_id=session_id,
            type="clinician",
            body=clin,
            provenance_ids=span_ids,
        )
        s2 = Summary(
            session_id=session_id,
            type="patient",
            body=pat,
            provenance_ids=span_ids,
        )
        db.add_all([s1, s2])
        db.commit()
        db.refresh(s1)
        db.refresh(s2)

        # Log AI processing
        processing_time_ms = (time.time() - start_time) * 1000
        audit_logger.log_ai_processing(
            user_id=current_user.id,
            session_id=session_id,
            patient_id=session.patient_id,
            action="SUMMARIZE",
            processing_time_ms=processing_time_ms,
            request=request,
            result="SUCCESS"
        )

        return {
            "clinician_summary_id": s1.id,
            "patient_summary_id": s2.id,
        }

    except HTTPException:
        raise
    except Exception as e:
        audit_logger.log_ai_processing(
            user_id=current_user.id,
            session_id=session_id,
            patient_id=None,
            action="SUMMARIZE_ERROR",
            processing_time_ms=(time.time() - start_time) * 1000,
            request=request,
            result="FAILURE",
            error_message=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Summarization error"
        )


# --------------------
# Review (HITL)
# --------------------
@app.post("/review/{summary_id}/approve")
def approve(
    summary_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    sm = db.query(Summary).get(summary_id)
    if not sm:
        raise HTTPException(404, "summary not found")
    sm.status = "approved"
    db.commit()
    db.refresh(sm)
    return {"id": sm.id, "status": sm.status}


# --------------------
# Patient QA (simple search)
# --------------------
@app.get("/patient/{patient_id}/qa")
def patient_qa(
    patient_id: int,
    q: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    sessions = (
        db.query(ConsultSession).filter(ConsultSession.patient_id == patient_id).all()
    )
    sids = [s.id for s in sessions]
    res = (
        db.query(Summary)
        .filter(Summary.session_id.in_(sids), Summary.status == "approved")
        .all()
    )
    hits = []
    for sm in res:
        if q.lower() in sm.body.lower():
            # 去重 citations（保持顺序）
            seen = set()
            cset = []
            for i in sm.provenance_ids.split(","):
                if i and i not in seen:
                    seen.add(i)
                    cset.append(f"[S{i}]")
            hits.append(
                {
                    "summary_id": sm.id,
                    "type": sm.type,
                    "snippet": sm.body[:500],  # 由 200 -> 500，避免看不到占位符
                    "citations": cset,
                }
            )
    return {"results": hits}


# --------------------
# Persistent Memory Management
# --------------------
@app.post("/session/{session_id}/segments/append")
def append_segment(
    session_id: int,
    req: SegmentAppendReq,
    current_user: User = Depends(get_current_user),
    request: Request,
    db: Session = Depends(get_db)
):
    """Append a new conversation segment to the session."""
    audit_logger = get_audit_logger(db)
    memory_manager = PersistentMemoryManager(db, audit_logger)

    try:
        segment = memory_manager.append_segment(
            session_id=session_id,
            text=req.text,
            user_id=current_user.id,
            speaker=req.speaker,
            start_ms=req.start_ms,
            end_ms=req.end_ms,
            edit_reason=req.edit_reason
        )

        return {
            "segment_id": segment.id,
            "span_idx": segment.span_idx,
            "speaker": segment.speaker,
            "phi_detected": segment.phi_detected,
            "phi_types": segment.phi_types_list,
            "status": segment.status
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@app.put("/session/{session_id}/segments/{segment_id}/edit")
def edit_segment(
    session_id: int,
    segment_id: int,
    req: SegmentEditReq,
    current_user: User = Depends(get_current_user),
    request: Request,
    db: Session = Depends(get_db)
):
    """Edit an existing conversation segment."""
    audit_logger = get_audit_logger(db)
    memory_manager = PersistentMemoryManager(db, audit_logger)

    try:
        new_segment = memory_manager.edit_segment(
            segment_id=segment_id,
            new_text=req.new_text,
            user_id=current_user.id,
            edit_reason=req.edit_reason,
            edit_note=req.edit_note
        )

        return {
            "new_segment_id": new_segment.id,
            "span_idx": new_segment.span_idx,
            "version": new_segment.version,
            "phi_detected": new_segment.phi_detected,
            "phi_types": new_segment.phi_types_list,
            "status": new_segment.status
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@app.get("/session/{session_id}/conversation")
def get_conversation_history(
    session_id: int,
    include_deleted: bool = False,
    version_history: bool = False,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get conversation history with optional filters."""
    audit_logger = get_audit_logger(db)
    memory_manager = PersistentMemoryManager(db, audit_logger)

    try:
        segments = memory_manager.get_conversation_history(
            session_id=session_id,
            include_deleted=include_deleted,
            version_history=version_history
        )

        return {
            "session_id": session_id,
            "segments": [
                {
                    "id": s.id,
                    "span_idx": s.span_idx,
                    "version": s.version,
                    "text_redacted": s.text_redacted,
                    "speaker": s.speaker,
                    "start_ms": s.start_ms,
                    "end_ms": s.end_ms,
                    "status": s.status,
                    "created_at": s.created_at.isoformat(),
                    "modified_at": s.modified_at.isoformat(),
                    "phi_detected": s.phi_detected,
                    "phi_types": s.phi_types_list,
                    "edit_reason": s.edit_reason,
                    "edit_note": s.edit_note
                }
                for s in segments
            ]
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@app.put("/session/{session_id}")
def update_session(
    session_id: int,
    req: SessionUpdateReq,
    current_user: User = Depends(get_current_user),
    request: Request,
    db: Session = Depends(get_db)
):
    """Update session metadata and state."""
    from datetime import datetime, timezone
    audit_logger = get_audit_logger(db)

    try:
        session = db.query(ConsultSession).get(session_id)
        if not session:
            raise HTTPException(404, "session not found")

        # Update fields if provided
        updates = {}
        if req.title is not None:
            session.title = req.title
            updates["title"] = req.title

        if req.description is not None:
            session.description = req.description
            updates["description"] = req.description

        if req.auto_summarize is not None:
            session.auto_summarize = req.auto_summarize
            updates["auto_summarize"] = req.auto_summarize

        if req.status is not None:
            old_status = session.status
            session.status = req.status
            updates["status"] = {"old": old_status, "new": req.status}

            # Update timestamps based on status
            if req.status == "paused":
                session.paused_at = datetime.now(timezone.utc)
            elif req.status == "completed":
                session.completed_at = datetime.now(timezone.utc)

        session.last_activity = datetime.now(timezone.utc)
        db.commit()

        # Audit log
        audit_logger.log_event(
            event_type="SESSION_MANAGEMENT",
            action="SESSION_UPDATE",
            user_id=current_user.id,
            session_id=session_id,
            request=request,
            details=updates
        )

        return {
            "session_id": session.id,
            "title": session.title,
            "description": session.description,
            "status": session.status,
            "auto_summarize": session.auto_summarize,
            "last_activity": session.last_activity.isoformat() if session.last_activity else None
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

# --------------------
# Get full summary (debug/preview)
# --------------------
@app.get("/summary/{summary_id}")
def get_summary(summary_id: int, db: Session = Depends(get_db)):
    sm = db.query(Summary).get(summary_id)
    if not sm:
        raise HTTPException(404, "summary not found")
    return {
        "id": sm.id,
        "type": sm.type,
        "status": sm.status,
        "body": sm.body,
        "provenance": [i for i in sm.provenance_ids.split(",") if i],
    }
