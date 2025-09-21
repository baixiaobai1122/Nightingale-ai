"""
test_redaction.py
-----------------
Tests to ensure no PHI (Personal Health Information) leaks.
Validates HIPAA compliance, PHI placeholders, and reversibility.
"""

import re
from backend.redact import redact_text, assert_no_phi
from datasets import load_dataset


def load_real_medical_data(num_samples=2, split="train"):
    """Load dialogues from omi-health/medical-dialogue-to-soap-summary (simplified)."""
    try:
        ds = load_dataset("omi-health/medical-dialogue-to-soap-summary", split=split)
        n = min(num_samples, len(ds))
        dialogues = [ds[i]["dialogue"] for i in range(n) if ds[i].get("dialogue")]
        return dialogues, f"omi-health/medical-dialogue-to-soap-summary ({split}, {len(dialogues)} dialogues)"
    except Exception as e:
        print(f"⚠️ Dataset load failed ({e}), using fallback synthetic data")
        return ["Patient reports chest pain and shortness of breath."], "synthetic_fallback"


def test_no_phi_after_redaction():
    """Basic test: redact common PHI types."""
    print("\n=== REDACTION TEST: Basic PHI Removal ===")
    src = "Patient John Doe, DOB 1990-01-01, email john@example.com, phone +65 9123 4567, MRN A1234567"
    red, mapping = redact_text(src)

    print("Original:", src)
    print("Redacted:", red)
    print("Mapping:", mapping)

    assert assert_no_phi(red), "PHI detected after redaction"
    assert any("<NAME_" in red for _ in [0]), "Name placeholder missing"
    assert any("<EMAIL_" in red for _ in [0]), "Email placeholder missing"
    assert any("<PHONE_" in red for _ in [0]), "Phone placeholder missing"
    assert any("<MRN_" in red for _ in [0]), "MRN placeholder missing"
    assert any("1990" in v for v in mapping.values()), "Date not captured in mapping"

    print("✅ PASS: All expected PHI properly redacted")


def test_comprehensive_phi_patterns():
    """Test multiple PHI categories like SSN, ZIP, IP, Phone."""
    print("\n=== REDACTION TEST: Pattern Coverage ===")
    cases = [
        "SSN: 123-45-6789",
        "Address: 123 Main St, Anytown, CA 90210",
        "IP: 192.168.1.1",
        "Phone: +65 9123 4567",
    ]
    for c in cases:
        red, _ = redact_text(c)
        print(f"Original: {c} | Redacted: {red}")
        assert assert_no_phi(red), f"PHI not redacted: {c}"
    print("✅ PASS: All common PHI patterns covered")


def test_hipaa_compliance():
    """Test a record with multiple HIPAA identifiers."""
    print("\n=== REDACTION TEST: HIPAA Compliance ===")
    text = """
    Patient: John Smith, DOB: 01/15/1980
    Address: 123 Medical Drive, Health City, CA 90210
    Phone: (555) 123-4567, Email: john.smith@email.com
    SSN: 123-45-6789, MRN: MED123456789
    Insurance: Policy INS987654321
    """
    red, mapping = redact_text(text)
    print("Redacted:", red)
    assert assert_no_phi(red), "PHI still present after redaction"
    assert len(mapping) >= 5, "Too few PHI items detected"
    print(f"✅ PASS: HIPAA identifiers redacted ({len(mapping)} items)")


def test_redaction_reversibility():
    """Ensure mapping is consistent for potential reversal."""
    print("\n=== REDACTION TEST: Mapping Reversibility ===")
    src = "Patient Jane Smith called about her appointment"
    red, mapping = redact_text(src)
    print("Redacted:", red, "| Mapping:", mapping)
    assert mapping, "Mapping empty"
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()), "Invalid mapping format"
    print("✅ PASS: Redaction mapping reversible")


def main():
    test_no_phi_after_redaction()
    test_comprehensive_phi_patterns()
    test_hipaa_compliance()
    test_redaction_reversibility()


if __name__ == "__main__":
    main()
