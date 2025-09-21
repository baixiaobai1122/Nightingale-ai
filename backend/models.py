from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Text, DateTime, JSON, Float
from sqlalchemy.orm import relationship
from datetime import datetime
from .db import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    pwd_hash = Column(String)
    role = Column(String, default="clinician")  # patient|clinician|admin

class Patient(Base):
    __tablename__ = "patients"
    id = Column(Integer, primary_key=True, index=True)
    name_redacted = Column(String, default="<NAME>")
    consent_status = Column(Boolean, default=False)

class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    started_at = Column(DateTime, default=datetime.utcnow)

    # Enhanced session state management
    status = Column(String, default="active")  # active|paused|completed|archived
    consent_version = Column(String, default="v1")

    # Session metadata
    title = Column(String, nullable=True)           # User-defined session title
    description = Column(Text, nullable=True)       # Session description

    # State tracking
    last_activity = Column(DateTime, default=datetime.utcnow)
    paused_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Session configuration
    auto_summarize = Column(Boolean, default=True)  # Auto-generate summaries
    edit_mode = Column(String, default="append")    # append|edit|locked

    patient = relationship("Patient")
    segments = relationship("Segment", back_populates="session")

class Segment(Base):
    __tablename__ = "segments"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), index=True)
    start_ms = Column(Integer)
    end_ms = Column(Integer)
    text_redacted = Column(Text)
    span_idx = Column(Integer)

    # Enhanced for backward compatibility
    session = relationship("Session", back_populates="segments")

class Summary(Base):
    __tablename__ = "summaries"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), index=True)
    type = Column(String)  # clinician|patient
    body = Column(Text)
    provenance_ids = Column(String)  # comma-separated S# indices e.g., "1,2,3"
    status = Column(String, default="draft")  # draft|approved

    # Enhanced summary metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    approved_at = Column(DateTime, nullable=True)
    approved_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    version = Column(Integer, default=1)

    # Summary editing
    edit_history = Column(Text, nullable=True)  # JSON of edit history

    approver = relationship("User")
