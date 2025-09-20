from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Text, DateTime
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
    status = Column(String, default="active")
    consent_version = Column(String, default="v1")
    patient = relationship("Patient")

class Segment(Base):
    __tablename__ = "segments"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), index=True)
    start_ms = Column(Integer)
    end_ms = Column(Integer)
    text_redacted = Column(Text)
    span_idx = Column(Integer)

class Summary(Base):
    __tablename__ = "summaries"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), index=True)
    type = Column(String)  # clinician|patient
    body = Column(Text)
    provenance_ids = Column(String)  # comma-separated S# indices e.g., "1,2,3"
    status = Column(String, default="draft")  # draft|approved
