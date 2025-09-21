import re
from typing import Tuple, Dict
from datetime import datetime

# Comprehensive HIPAA PHI redaction patterns - 18 HIPAA identifiers
HIPAA_PATTERNS = {
    # 1. Names (full names, last names with initials)
    "NAME": re.compile(r"\b([A-Z][a-z]{1,}\s+[A-Z][a-z]{1,}|\b[A-Z][a-z]+,\s*[A-Z]\.?|\bDr\.?\s+[A-Z][a-z]+|\bMr\.?\s+[A-Z][a-z]+|\bMs\.?\s+[A-Z][a-z]+|\bMrs\.?\s+[A-Z][a-z]+)"),

    # 2. Geographic subdivisions smaller than a state (addresses, zip codes)
    "ADDRESS": re.compile(r"\b\d+\s+[\w\s]{3,40}\s+(Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Boulevard|Blvd\.?|Drive|Dr\.?|Lane|Ln\.?|Way|Court|Ct\.?|Place|Pl\.?)\b"),
    "ZIP_CODE": re.compile(r"\b\d{5}(?:-\d{4})?\b"),
    "CITY_STATE": re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2}\b"),

    # 3. Dates (birth dates, admission dates, discharge dates, death dates)
    "DATE": re.compile(r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4}|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{2,4})\b"),

    # 4. Telephone numbers
    "PHONE": re.compile(r"\b\+?[\d\s\-\(\)]{7,}\b"),

    # 5. Fax numbers
    "FAX": re.compile(r"\b(?:fax|fax[:\s]*|f[:\s]*)\s*(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b", re.IGNORECASE),

    # 6. Email addresses
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),

    # 7. Social Security numbers
    "SSN": re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),

    # 8. Medical record numbers
    "MRN": re.compile(r"\b(?:MRN|MR|Medical\s+Record|Patient\s+ID)[:\s#]*([A-Z0-9]{6,})\b", re.IGNORECASE),

    # 9. Health plan beneficiary numbers
    "HEALTH_PLAN_ID": re.compile(r"\b(?:Plan\s+ID|Beneficiary|Member\s+ID|Policy)[:\s#]*([A-Z0-9]{6,})\b", re.IGNORECASE),

    # 10. Account numbers
    "ACCOUNT_NUMBER": re.compile(r"\b(?:Account|Acct)[:\s#]*([A-Z0-9]{6,})\b", re.IGNORECASE),

    # 11. Certificate/license numbers
    "LICENSE_NUMBER": re.compile(r"\b(?:License|Cert|Certificate)[:\s#]*([A-Z0-9]{6,})\b", re.IGNORECASE),

    # 12. Vehicle identifiers and serial numbers
    "VIN": re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b"),
    "SERIAL_NUMBER": re.compile(r"\b(?:Serial|SN)[:\s#]*([A-Z0-9]{6,})\b", re.IGNORECASE),

    # 13. Device identifiers and serial numbers
    "DEVICE_ID": re.compile(r"\b(?:Device\s+ID|Serial)[:\s#]*([A-Z0-9]{6,})\b", re.IGNORECASE),

    # 14. Web URLs
    "URL": re.compile(r"\bhttps?://[^\s]+\b"),

    # 15. IP addresses
    "IP_ADDRESS": re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),

    # 16. Biometric identifiers (fingerprints, voiceprints, etc.)
    "BIOMETRIC_ID": re.compile(r"\b(?:Fingerprint|Biometric|Voice\s+ID)[:\s#]*([A-Z0-9]{6,})\b", re.IGNORECASE),

    # 17. Full face photographic images
    "PHOTO_ID": re.compile(r"\b(?:Photo\s+ID|Image\s+ID)[:\s#]*([A-Z0-9]{6,})\b", re.IGNORECASE),

    # 18. Any other unique identifying number, characteristic, or code
    "UNIQUE_ID": re.compile(r"\b[A-Z]{2,}\d{6,}\b"),  # Generic pattern for IDs

    # Additional medical-specific patterns
    "PRESCRIPTION_NUMBER": re.compile(r"\b(?:Rx|Prescription)[:\s#]*([A-Z0-9]{6,})\b", re.IGNORECASE),
    "LAB_RESULT_ID": re.compile(r"\b(?:Lab\s+ID|Test\s+ID)[:\s#]*([A-Z0-9]{6,})\b", re.IGNORECASE),
    "INSURANCE_ID": re.compile(r"\b(?:Insurance|Policy)[:\s#]*([A-Z0-9]{6,})\b", re.IGNORECASE),
}

# Legacy patterns for backward compatibility
PATTERNS = HIPAA_PATTERNS

def redact_text(text: str) -> Tuple[str, dict]:
    """
    Redact PHI from text using comprehensive HIPAA patterns.

    Args:
        text: Input text containing potential PHI

    Returns:
        Tuple of (redacted_text, mapping_dict)
        mapping_dict maps redaction tags back to original values
    """
    mapping = {}
    out = text
    counter = 1

    # Process patterns in order of specificity to avoid conflicts
    pattern_order = [
        "SSN", "MRN", "HEALTH_PLAN_ID", "ACCOUNT_NUMBER", "LICENSE_NUMBER",
        "EMAIL", "URL", "IP_ADDRESS", "PHONE", "FAX",
        "DATE", "ZIP_CODE", "VIN", "BIOMETRIC_ID", "PHOTO_ID",
        "PRESCRIPTION_NUMBER", "LAB_RESULT_ID", "INSURANCE_ID",
        "DEVICE_ID", "SERIAL_NUMBER", "UNIQUE_ID",
        "ADDRESS", "CITY_STATE", "NAME"  # Process names last to avoid conflicts
    ]

    for label in pattern_order:
        if label in PATTERNS:
            rx = PATTERNS[label]
            def repl(m):
                nonlocal counter
                tag = f"<{label}_{counter}>"
                mapping[tag] = m.group(0)
                counter += 1
                return tag
            out = rx.sub(repl, out)

    return out, mapping

def assert_no_phi(text: str) -> bool:
    """
    Validate that no PHI patterns remain in the text.

    Args:
        text: Text to validate

    Returns:
        True if no PHI detected, False otherwise
    """
    for pattern_name, pattern in PATTERNS.items():
        if pattern.search(text):
            # For debugging purposes, you could log which pattern matched
            return False
    return True

def get_phi_analysis(text: str) -> Dict[str, int]:
    """
    Analyze text and return count of each PHI type found.

    Args:
        text: Text to analyze

    Returns:
        Dictionary mapping PHI type names to counts
    """
    analysis = {}
    for pattern_name, pattern in PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            analysis[pattern_name] = len(matches)
    return analysis

def redact_with_context(text: str, context_window: int = 5) -> Tuple[str, dict, dict]:
    """
    Redact PHI while preserving context information for audit purposes.

    Args:
        text: Input text
        context_window: Number of words before/after to include in context

    Returns:
        Tuple of (redacted_text, mapping_dict, context_dict)
    """
    words = text.split()
    redacted_text, mapping = redact_text(text)
    context_info = {}

    # Find positions of redacted items for context
    for tag, original in mapping.items():
        # Simple context extraction - in production, use more sophisticated methods
        for i, word in enumerate(words):
            if original in word:
                start = max(0, i - context_window)
                end = min(len(words), i + context_window + 1)
                context_info[tag] = {
                    "before": " ".join(words[start:i]),
                    "after": " ".join(words[i+1:end]),
                    "position": i
                }
                break

    return redacted_text, mapping, context_info
