from fastapi import FastAPI, Depends, HTTPException, status, Body
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List
from datetime import timedelta
from pydantic import BaseModel

from .db import Base, engine, get_db
from .models import User, Patient, Session as ConsultSession, Segment, Summary
from .security import verify_password, get_password_hash, create_access_token
from .redact import redact_text, assert_no_phi
from .summarize import make_dual_summaries

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Nightingale MVP", version="0.3.0")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# --- In-memory token store demo (not for prod)
TOKENS = {}

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
    db: Session = Depends(get_db)
):
    u = db.query(User).filter(User.name == form_data.username).first()
    if not u or not verify_password(form_data.password, u.pwd_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="bad creds"
        )
    token = create_access_token({"sub": str(u.id)})
    TOKENS[token] = u.id
    return {"access_token": token, "token_type": "bearer"}


# --------------------
# Consent
# --------------------
@app.post("/consent/record")
def record_consent(
    patient_name: str = Body(...),
    consent: bool = Body(...),
    db: Session = Depends(get_db)
):
    # redacted storage for demo
    p = Patient(name_redacted="<NAME>", consent_status=consent)
    db.add(p)
    db.commit()
    db.refresh(p)
    return {"patient_id": p.id, "consent": consent}


# --------------------
# Session (Pydantic JSON body)
# --------------------
class SessionStartReq(BaseModel):
    patient_id: int

@app.post("/session/start")
def start_session(
    req: SessionStartReq,
    db: Session = Depends(get_db)
):
    p = db.query(Patient).get(req.patient_id)
    if not p or not p.consent_status:
        raise HTTPException(403, "consent required")
    s = ConsultSession(patient_id=req.patient_id)
    db.add(s)
    db.commit()
    db.refresh(s)
    return {"session_id": s.id}


# --------------------
# ASR ingest (text-based stub)
# --------------------
@app.post("/asr/ingest")
def asr_ingest(
    session_id: int = Body(...),
    segments: List[dict] = Body(...),
    db: Session = Depends(get_db)
):
    s = db.query(ConsultSession).get(session_id)
    if not s:
        raise HTTPException(404, "session not found")

    created: List[int] = []
    # 连续递增 span_idx，避免重复编号
    current_max = (
        db.query(func.max(Segment.span_idx))
        .filter(Segment.session_id == session_id)
        .scalar()
        or 0
    )

    for i, seg in enumerate(segments, start=1):
        red_txt, _ = redact_text(seg.get("text", ""))
        if not assert_no_phi(red_txt):
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
    return {"ingested_spans": created}


# --------------------
# Summarize
# --------------------
@app.post("/summarize/{session_id}")
def summarize(session_id: int, db: Session = Depends(get_db)):
    segs = (
        db.query(Segment)
        .filter(Segment.session_id == session_id)
        .order_by(Segment.span_idx)
        .all()
    )
    if not segs:
        raise HTTPException(404, "no segments")
    spans = [(s.span_idx, s.text_redacted) for s in segs]
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
    return {
        "clinician_summary_id": s1.id,
        "patient_summary_id": s2.id,
    }


# --------------------
# Review (HITL)
# --------------------
@app.post("/review/{summary_id}/approve")
def approve(summary_id: int, db: Session = Depends(get_db)):
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
def patient_qa(patient_id: int, q: str, db: Session = Depends(get_db)):
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
