import os
import base64
from typing import Optional, Any
from sqlalchemy import Column, Integer, String, Boolean, Text, DateTime, LargeBinary, event
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.types import TypeDecorator, String as SQLString
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from .db import Base

class EncryptionManager:
    """Manages encryption keys and operations for sensitive data."""

    _instance = None
    _encryption_key = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._encryption_key is None:
            self._initialize_encryption()

    def _initialize_encryption(self):
        """Initialize encryption key from environment or generate new one."""
        key_b64 = os.getenv("DB_ENCRYPTION_KEY")

        if key_b64:
            try:
                self._encryption_key = base64.urlsafe_b64decode(key_b64.encode())
            except Exception:
                # If key is invalid, generate new one
                self._generate_new_key()
        else:
            self._generate_new_key()

    def _generate_new_key(self):
        """Generate a new encryption key."""
        # In production, derive from a master key with proper key management
        password = os.getenv("MASTER_PASSWORD", "nightingale-demo-key").encode()
        salt = os.getenv("ENCRYPTION_SALT", "nightingale-salt").encode()

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        self._encryption_key = base64.urlsafe_b64decode(
            base64.urlsafe_b64encode(kdf.derive(password))
        )

    def get_fernet(self) -> Fernet:
        """Get Fernet encryption instance."""
        key_b64 = base64.urlsafe_b64encode(self._encryption_key)
        return Fernet(key_b64)

    def encrypt(self, data: str) -> str:
        """Encrypt string data."""
        if not data:
            return data
        f = self.get_fernet()
        encrypted = f.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data."""
        if not encrypted_data:
            return encrypted_data
        try:
            f = self.get_fernet()
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = f.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception:
            # Return as-is if decryption fails (for backward compatibility)
            return encrypted_data

class EncryptedType(TypeDecorator):
    """SQLAlchemy type for automatic encryption/decryption."""

    impl = Text
    cache_ok = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encryption_manager = EncryptionManager()

    def process_bind_param(self, value: Optional[str], dialect) -> Optional[str]:
        """Encrypt value before storing in database."""
        if value is not None:
            return self.encryption_manager.encrypt(value)
        return value

    def process_result_value(self, value: Optional[str], dialect) -> Optional[str]:
        """Decrypt value when retrieving from database."""
        if value is not None:
            return self.encryption_manager.decrypt(value)
        return value

class HashableEncryptedType(TypeDecorator):
    """SQLAlchemy type for searchable encrypted fields."""

    impl = Text
    cache_ok = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encryption_manager = EncryptionManager()

    def process_bind_param(self, value: Optional[str], dialect) -> Optional[str]:
        """Encrypt value before storing in database."""
        if value is not None:
            return self.encryption_manager.encrypt(value)
        return value

    def process_result_value(self, value: Optional[str], dialect) -> Optional[str]:
        """Decrypt value when retrieving from database."""
        if value is not None:
            return self.encryption_manager.decrypt(value)
        return value

# Enhanced models with encryption
class EncryptedPatient(Base):
    """Patient model with encrypted sensitive fields."""
    __tablename__ = "encrypted_patients"

    id = Column(Integer, primary_key=True, index=True)

    # Encrypted fields
    _name_encrypted = Column("name_encrypted", EncryptedType)
    _email_encrypted = Column("email_encrypted", EncryptedType, nullable=True)
    _phone_encrypted = Column("phone_encrypted", EncryptedType, nullable=True)
    _address_encrypted = Column("address_encrypted", EncryptedType, nullable=True)
    _ssn_encrypted = Column("ssn_encrypted", EncryptedType, nullable=True)
    _medical_record_encrypted = Column("medical_record_encrypted", EncryptedType, nullable=True)

    # Searchable hashes (for finding records without decrypting)
    name_hash = Column(String, index=True, nullable=True)
    email_hash = Column(String, index=True, nullable=True)
    phone_hash = Column(String, index=True, nullable=True)

    # Non-sensitive fields
    consent_status = Column(Boolean, default=False)
    created_at = Column(DateTime)
    consent_version = Column(String, default="v1")

    # Hybrid properties for transparent access
    @hybrid_property
    def name(self):
        return self._name_encrypted

    @name.setter
    def name(self, value):
        self._name_encrypted = value
        if value:
            self.name_hash = self._generate_search_hash(value)

    @hybrid_property
    def email(self):
        return self._email_encrypted

    @email.setter
    def email(self, value):
        self._email_encrypted = value
        if value:
            self.email_hash = self._generate_search_hash(value)

    @hybrid_property
    def phone(self):
        return self._phone_encrypted

    @phone.setter
    def phone(self, value):
        self._phone_encrypted = value
        if value:
            self.phone_hash = self._generate_search_hash(value)

    @hybrid_property
    def address(self):
        return self._address_encrypted

    @address.setter
    def address(self, value):
        self._address_encrypted = value

    @hybrid_property
    def ssn(self):
        return self._ssn_encrypted

    @ssn.setter
    def ssn(self, value):
        self._ssn_encrypted = value

    @hybrid_property
    def medical_record_number(self):
        return self._medical_record_encrypted

    @medical_record_number.setter
    def medical_record_number(self, value):
        self._medical_record_encrypted = value

    @staticmethod
    def _generate_search_hash(value: str) -> str:
        """Generate searchable hash for encrypted fields."""
        import hashlib
        salt = os.getenv("SEARCH_SALT", "nightingale-search").encode()
        return hashlib.sha256(value.lower().encode() + salt).hexdigest()

    @classmethod
    def find_by_name_hash(cls, session, name: str):
        """Find patient by name without decrypting all records."""
        name_hash = cls._generate_search_hash(name)
        return session.query(cls).filter(cls.name_hash == name_hash).first()

    @classmethod
    def find_by_email_hash(cls, session, email: str):
        """Find patient by email without decrypting all records."""
        email_hash = cls._generate_search_hash(email)
        return session.query(cls).filter(cls.email_hash == email_hash).first()

class EncryptedSegment(Base):
    """Segment model with encrypted text content."""
    __tablename__ = "encrypted_segments"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, index=True)
    start_ms = Column(Integer)
    end_ms = Column(Integer)

    # Store original text encrypted, redacted text plain for processing
    _original_text_encrypted = Column("original_text_encrypted", EncryptedType)
    text_redacted = Column(Text)  # Already redacted, safe to store plain

    span_idx = Column(Integer)
    created_at = Column(DateTime)

    @hybrid_property
    def original_text(self):
        """Access to original (unredacted) text - highly controlled."""
        return self._original_text_encrypted

    @original_text.setter
    def original_text(self, value):
        self._original_text_encrypted = value

class EncryptedSummary(Base):
    """Summary model with encrypted content."""
    __tablename__ = "encrypted_summaries"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, index=True)
    type = Column(String)  # clinician|patient

    # Encrypted summary content
    _body_encrypted = Column("body_encrypted", EncryptedType)
    provenance_ids = Column(String)  # Can remain plain as it's just references

    status = Column(String, default="draft")
    created_at = Column(DateTime)
    approved_at = Column(DateTime, nullable=True)
    approved_by = Column(Integer, nullable=True)  # User ID

    @hybrid_property
    def body(self):
        return self._body_encrypted

    @body.setter
    def body(self, value):
        self._body_encrypted = value

# Database backup encryption
class BackupManager:
    """Manages encrypted database backups."""

    def __init__(self):
        self.encryption_manager = EncryptionManager()

    def create_encrypted_backup(self, db_path: str, backup_path: str):
        """Create encrypted backup of database."""
        import sqlite3
        import gzip

        # Read database
        with open(db_path, 'rb') as f:
            db_data = f.read()

        # Compress and encrypt
        compressed_data = gzip.compress(db_data)
        encrypted_data = self.encryption_manager.get_fernet().encrypt(compressed_data)

        # Write encrypted backup
        with open(backup_path, 'wb') as f:
            f.write(encrypted_data)

    def restore_encrypted_backup(self, backup_path: str, restore_path: str):
        """Restore database from encrypted backup."""
        import gzip

        # Read encrypted backup
        with open(backup_path, 'rb') as f:
            encrypted_data = f.read()

        # Decrypt and decompress
        compressed_data = self.encryption_manager.get_fernet().decrypt(encrypted_data)
        db_data = gzip.decompress(compressed_data)

        # Write restored database
        with open(restore_path, 'wb') as f:
            f.write(db_data)

# Configuration for production encrypted database
class EncryptedDatabaseConfig:
    """Configuration for encrypted database setup."""

    @staticmethod
    def get_production_db_url():
        """Get production database URL with encryption."""
        # For PostgreSQL with TDE (Transparent Data Encryption)
        return {
            "url": os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/nightingale"),
            "connect_args": {
                "sslmode": "require",
                "sslcert": "/etc/ssl/certs/client-cert.pem",
                "sslkey": "/etc/ssl/private/client-key.pem",
                "sslrootcert": "/etc/ssl/certs/ca-cert.pem"
            },
            "pool_pre_ping": True,
            "pool_recycle": 3600,
            "echo": False  # Don't log SQL in production
        }

    @staticmethod
    def get_sqlite_encrypted_config():
        """Get SQLite configuration with file encryption."""
        return {
            "url": "sqlite:///./nightingale_encrypted.db",
            "connect_args": {
                "check_same_thread": False,
                # SQLCipher configuration
                "init_command": f"PRAGMA key = '{os.getenv('SQLITE_KEY', 'demo-key')}'"
            }
        }