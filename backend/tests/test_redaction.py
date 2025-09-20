# test_redaction.py
"""
Test to ensure no PHI (Personal Health Information) leaks to outputs or logs.
This is critical for HIPAA compliance and patient privacy.
"""
from backend.redact import redact_text, assert_no_phi
import re

def test_no_phi_after_redaction():
    """Test that all PHI is properly redacted from text."""
    src = "Patient John Doe, DOB 1990-01-01, email john@example.com, phone +65 9123 4567, MRN A1234567"
    red, mapping = redact_text(src)
    
    assert assert_no_phi(red), "PHI detected in redacted text"
    
    assert "<NAME_" in red, "Name placeholder missing"
    assert "<DOB_" in red, "DOB placeholder missing" 
    assert "<EMAIL_" in red, "Email placeholder missing"
    assert "<PHONE_" in red, "Phone placeholder missing"
    assert "<ID_" in red, "ID placeholder missing"

def test_comprehensive_phi_patterns():
    """Test redaction of various PHI patterns."""
    test_cases = [
        ("SSN: 123-45-6789", r"\d{3}-\d{2}-\d{4}"),
        ("Credit card: 4532-1234-5678-9012", r"\d{4}-\d{4}-\d{4}-\d{4}"),
        ("Address: 123 Main St, Anytown, CA 90210", r"\d{5}"),
        ("IP: 192.168.1.1", r"\d+\.\d+\.\d+\.\d+"),
    ]
    
    for original, pattern in test_cases:
        redacted, _ = redact_text(original)
        assert not re.search(pattern, redacted), f"Pattern {pattern} not redacted in: {original}"

def test_redaction_reversibility():
    """Test that redaction mapping allows for proper restoration."""
    src = "Patient Jane Smith called about her appointment"
    red, mapping = redact_text(src)
    
    assert len(mapping) > 0, "Redaction mapping is empty"
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()), "Invalid mapping format"
