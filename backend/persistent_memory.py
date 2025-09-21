from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Text, DateTime, JSON, Float
from sqlalchemy.orm import relationship, Session
from sqlalchemy.ext.hybrid import hybrid_property
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import json

from .db import Base
from .models import User, Patient, Session as ConsultSession, Segment, Summary
from .redact import redact_text, assert_no_phi, get_phi_analysis
from .audit import AuditLogger

class SegmentStatus(str, Enum):
    """Status of conversation segments."""
    ACTIVE = "active"           # Current active segment
    EDITED = "edited"           # Has been edited
    DELETED = "deleted"         # Soft deleted
    SUPERSEDED = "superseded"   # Replaced by newer version

class SessionState(str, Enum):
    """Session states for conversation management."""
    ACTIVE = "active"           # Currently ongoing
    PAUSED = "paused"           # Temporarily paused
    COMPLETED = "completed"     # Session finished
    ARCHIVED = "archived"       # Archived for storage

class ConversationSegment(Base):
    """Enhanced segment model with versioning and editing capabilities."""
    __tablename__ = "conversation_segments"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), index=True)

    # Versioning and hierarchy
    span_idx = Column(Integer, index=True)  # Position in conversation
    version = Column(Integer, default=1)    # Version number
    parent_segment_id = Column(Integer, ForeignKey("conversation_segments.id"), nullable=True)

    # Content
    original_text = Column(Text)            # Original unredacted text (encrypted in production)
    text_redacted = Column(Text)            # Redacted text for processing

    # Metadata
    start_ms = Column(Integer)
    end_ms = Column(Integer)
    speaker = Column(String, default="patient")  # patient|doctor|system

    # Status and tracking
    status = Column(String, default=SegmentStatus.ACTIVE)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    modified_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    modified_by = Column(Integer, ForeignKey("users.id"), nullable=True)

    # PHI and compliance
    phi_detected = Column(Boolean, default=False)
    phi_types = Column(Text)  # JSON array of detected PHI types
    redaction_verified = Column(Boolean, default=False)

    # Edit metadata
    edit_reason = Column(String, nullable=True)  # correction|clarification|privacy|other
    edit_note = Column(Text, nullable=True)      # Human-readable edit explanation

    # Relationships
    session = relationship("Session", back_populates="segments")
    modified_by_user = relationship("User")
    child_segments = relationship("ConversationSegment",
                                backref="parent_segment",
                                remote_side=[id])

    @hybrid_property
    def phi_types_list(self) -> List[str]:
        """Get PHI types as a list."""
        if self.phi_types:
            try:
                return json.loads(self.phi_types)
            except json.JSONDecodeError:
                return []
        return []

    @phi_types_list.setter
    def phi_types_list(self, value: List[str]):
        """Set PHI types from a list."""
        self.phi_types = json.dumps(value) if value else None

    def is_current_version(self) -> bool:
        """Check if this is the current/active version of the segment."""
        return self.status == SegmentStatus.ACTIVE

class SessionSnapshot(Base):
    """Snapshots of session state for rollback and history."""
    __tablename__ = "session_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), index=True)

    # Snapshot metadata
    snapshot_name = Column(String)                    # User-defined name
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    created_by = Column(Integer, ForeignKey("users.id"))

    # State capture
    segment_count = Column(Integer)
    last_segment_idx = Column(Integer)
    session_state = Column(String)

    # Snapshot data (references to segments at time of snapshot)
    segment_ids = Column(Text)  # JSON array of segment IDs included in snapshot

    # Summary state at time of snapshot
    summary_data = Column(Text)  # JSON of summary state

    # Relationships
    session = relationship("Session")
    created_by_user = relationship("User")

class PersistentMemoryManager:
    """Manages persistent conversation memory with editing capabilities."""

    def __init__(self, db: Session, audit_logger: AuditLogger):
        self.db = db
        self.audit_logger = audit_logger

    def append_segment(
        self,
        session_id: int,
        text: str,
        user_id: int,
        speaker: str = "patient",
        start_ms: int = 0,
        end_ms: int = 0,
        edit_reason: Optional[str] = None
    ) -> ConversationSegment:
        """Append a new segment to the conversation."""

        # Get next span index
        max_span = (
            self.db.query(ConversationSegment.span_idx)
            .filter(ConversationSegment.session_id == session_id)
            .order_by(ConversationSegment.span_idx.desc())
            .first()
        )
        next_span_idx = (max_span[0] + 1) if max_span else 1

        # Analyze and redact PHI
        phi_analysis = get_phi_analysis(text)
        redacted_text, redaction_mapping = redact_text(text)
        redaction_valid = assert_no_phi(redacted_text)

        if not redaction_valid:
            raise ValueError("PHI redaction validation failed")

        # Create new segment
        segment = ConversationSegment(
            session_id=session_id,
            span_idx=next_span_idx,
            version=1,
            original_text=text,
            text_redacted=redacted_text,
            start_ms=start_ms,
            end_ms=end_ms,
            speaker=speaker,
            status=SegmentStatus.ACTIVE,
            modified_by=user_id,
            phi_detected=bool(phi_analysis),
            phi_types_list=list(phi_analysis.keys()),
            redaction_verified=redaction_valid,
            edit_reason=edit_reason
        )

        self.db.add(segment)
        self.db.commit()
        self.db.refresh(segment)

        # Audit log
        self.audit_logger.log_event(
            event_type="CONVERSATION_MANAGEMENT",
            action="SEGMENT_APPEND",
            user_id=user_id,
            session_id=session_id,
            resource="conversation_segment",
            resource_id=str(segment.id),
            phi_detected=bool(phi_analysis),
            phi_types=list(phi_analysis.keys()),
            details={
                "span_idx": next_span_idx,
                "speaker": speaker,
                "phi_items_redacted": len(redaction_mapping)
            }
        )

        return segment

    def edit_segment(
        self,
        segment_id: int,
        new_text: str,
        user_id: int,
        edit_reason: str,
        edit_note: Optional[str] = None
    ) -> ConversationSegment:
        """Edit an existing segment, creating a new version."""

        # Get original segment
        original_segment = self.db.query(ConversationSegment).get(segment_id)
        if not original_segment:
            raise ValueError("Segment not found")

        if original_segment.status != SegmentStatus.ACTIVE:
            raise ValueError("Can only edit active segments")

        # Mark original as superseded
        original_segment.status = SegmentStatus.SUPERSEDED
        original_segment.modified_at = datetime.now(timezone.utc)

        # Analyze and redact new content
        phi_analysis = get_phi_analysis(new_text)
        redacted_text, redaction_mapping = redact_text(new_text)
        redaction_valid = assert_no_phi(redacted_text)

        if not redaction_valid:
            raise ValueError("PHI redaction validation failed")

        # Create new version
        new_segment = ConversationSegment(
            session_id=original_segment.session_id,
            span_idx=original_segment.span_idx,
            version=original_segment.version + 1,
            parent_segment_id=original_segment.id,
            original_text=new_text,
            text_redacted=redacted_text,
            start_ms=original_segment.start_ms,
            end_ms=original_segment.end_ms,
            speaker=original_segment.speaker,
            status=SegmentStatus.ACTIVE,
            modified_by=user_id,
            phi_detected=bool(phi_analysis),
            phi_types_list=list(phi_analysis.keys()),
            redaction_verified=redaction_valid,
            edit_reason=edit_reason,
            edit_note=edit_note
        )

        self.db.add(new_segment)
        self.db.commit()
        self.db.refresh(new_segment)

        # Audit log
        self.audit_logger.log_event(
            event_type="CONVERSATION_MANAGEMENT",
            action="SEGMENT_EDIT",
            user_id=user_id,
            session_id=original_segment.session_id,
            resource="conversation_segment",
            resource_id=str(new_segment.id),
            phi_detected=bool(phi_analysis),
            phi_types=list(phi_analysis.keys()),
            details={
                "original_segment_id": segment_id,
                "span_idx": original_segment.span_idx,
                "version": new_segment.version,
                "edit_reason": edit_reason,
                "edit_note": edit_note
            }
        )

        return new_segment

    def delete_segment(
        self,
        segment_id: int,
        user_id: int,
        delete_reason: str
    ) -> bool:
        """Soft delete a segment."""

        segment = self.db.query(ConversationSegment).get(segment_id)
        if not segment:
            raise ValueError("Segment not found")

        segment.status = SegmentStatus.DELETED
        segment.modified_at = datetime.now(timezone.utc)
        segment.modified_by = user_id
        segment.edit_reason = delete_reason

        self.db.commit()

        # Audit log
        self.audit_logger.log_event(
            event_type="CONVERSATION_MANAGEMENT",
            action="SEGMENT_DELETE",
            user_id=user_id,
            session_id=segment.session_id,
            resource="conversation_segment",
            resource_id=str(segment_id),
            details={
                "span_idx": segment.span_idx,
                "delete_reason": delete_reason
            }
        )

        return True

    def get_conversation_history(
        self,
        session_id: int,
        include_deleted: bool = False,
        version_history: bool = False
    ) -> List[ConversationSegment]:
        """Get conversation history with various filtering options."""

        query = self.db.query(ConversationSegment).filter(
            ConversationSegment.session_id == session_id
        )

        if not include_deleted:
            query = query.filter(ConversationSegment.status != SegmentStatus.DELETED)

        if not version_history:
            # Only current versions
            query = query.filter(ConversationSegment.status == SegmentStatus.ACTIVE)

        return query.order_by(
            ConversationSegment.span_idx,
            ConversationSegment.version.desc()
        ).all()

    def create_session_snapshot(
        self,
        session_id: int,
        user_id: int,
        snapshot_name: str
    ) -> SessionSnapshot:
        """Create a snapshot of current session state."""

        # Get current active segments
        active_segments = self.get_conversation_history(session_id)

        # Get session info
        session = self.db.query(ConsultSession).get(session_id)
        if not session:
            raise ValueError("Session not found")

        # Create snapshot
        snapshot = SessionSnapshot(
            session_id=session_id,
            snapshot_name=snapshot_name,
            created_by=user_id,
            segment_count=len(active_segments),
            last_segment_idx=max([s.span_idx for s in active_segments]) if active_segments else 0,
            session_state=getattr(session, 'status', 'active'),
            segment_ids=json.dumps([s.id for s in active_segments])
        )

        self.db.add(snapshot)
        self.db.commit()
        self.db.refresh(snapshot)

        # Audit log
        self.audit_logger.log_event(
            event_type="CONVERSATION_MANAGEMENT",
            action="SNAPSHOT_CREATE",
            user_id=user_id,
            session_id=session_id,
            resource="session_snapshot",
            resource_id=str(snapshot.id),
            details={
                "snapshot_name": snapshot_name,
                "segments_count": len(active_segments)
            }
        )

        return snapshot

    def rollback_to_snapshot(
        self,
        snapshot_id: int,
        user_id: int
    ) -> bool:
        """Rollback session to a previous snapshot state."""

        snapshot = self.db.query(SessionSnapshot).get(snapshot_id)
        if not snapshot:
            raise ValueError("Snapshot not found")

        # Get snapshot segment IDs
        try:
            snapshot_segment_ids = json.loads(snapshot.segment_ids)
        except json.JSONDecodeError:
            raise ValueError("Invalid snapshot data")

        # Mark all segments after snapshot as superseded
        segments_to_supersede = self.db.query(ConversationSegment).filter(
            ConversationSegment.session_id == snapshot.session_id,
            ConversationSegment.status == SegmentStatus.ACTIVE,
            ~ConversationSegment.id.in_(snapshot_segment_ids)
        ).all()

        for segment in segments_to_supersede:
            segment.status = SegmentStatus.SUPERSEDED
            segment.modified_at = datetime.now(timezone.utc)
            segment.modified_by = user_id

        self.db.commit()

        # Audit log
        self.audit_logger.log_event(
            event_type="CONVERSATION_MANAGEMENT",
            action="SNAPSHOT_ROLLBACK",
            user_id=user_id,
            session_id=snapshot.session_id,
            resource="session_snapshot",
            resource_id=str(snapshot_id),
            details={
                "segments_superseded": len(segments_to_supersede),
                "rollback_to": snapshot.snapshot_name
            }
        )

        return True

    def get_segment_history(self, span_idx: int, session_id: int) -> List[ConversationSegment]:
        """Get full version history for a specific segment position."""

        return self.db.query(ConversationSegment).filter(
            ConversationSegment.session_id == session_id,
            ConversationSegment.span_idx == span_idx
        ).order_by(ConversationSegment.version.desc()).all()

    def get_session_statistics(self, session_id: int) -> Dict[str, Any]:
        """Get comprehensive session statistics."""

        segments = self.db.query(ConversationSegment).filter(
            ConversationSegment.session_id == session_id
        ).all()

        active_segments = [s for s in segments if s.status == SegmentStatus.ACTIVE]

        total_phi_types = set()
        for segment in active_segments:
            total_phi_types.update(segment.phi_types_list)

        stats = {
            "total_segments": len(segments),
            "active_segments": len(active_segments),
            "deleted_segments": len([s for s in segments if s.status == SegmentStatus.DELETED]),
            "edited_segments": len([s for s in segments if s.version > 1]),
            "phi_types_detected": list(total_phi_types),
            "phi_segments_count": len([s for s in active_segments if s.phi_detected]),
            "conversation_duration_ms": (
                max([s.end_ms for s in active_segments if s.end_ms]) -
                min([s.start_ms for s in active_segments if s.start_ms])
            ) if active_segments else 0,
            "last_activity": max([s.modified_at for s in segments]) if segments else None
        }

        return stats

    def update_session_state(
        self,
        session_id: int,
        new_state: SessionState,
        user_id: int
    ) -> bool:
        """Update session state with audit logging."""

        session = self.db.query(ConsultSession).get(session_id)
        if not session:
            raise ValueError("Session not found")

        old_state = getattr(session, 'status', 'unknown')
        session.status = new_state.value

        self.db.commit()

        # Audit log
        self.audit_logger.log_event(
            event_type="SESSION_MANAGEMENT",
            action="STATE_CHANGE",
            user_id=user_id,
            session_id=session_id,
            details={
                "old_state": old_state,
                "new_state": new_state.value
            }
        )

        return True