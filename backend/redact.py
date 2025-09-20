import re
from typing import Tuple

# Very small redaction utility for demo purposes
PATTERNS = {
    "EMAIL": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "PHONE": re.compile(r"\b\+?\d[\d\s\-]{7,}\b"),
    "DOB": re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})\b"),
    "ID": re.compile(r"\b([A-Z]{1,2}\d{5,}|\d{8,})\b"),
    # naive name pattern for demo: Capitalized First Last
    "NAME": re.compile(r"\b([A-Z][a-z]+\s[A-Z][a-z]+)\b"),
}

def redact_text(text: str) -> Tuple[str, dict]:
    mapping = {}
    out = text
    counter = 1
    for label, rx in PATTERNS.items():
        def repl(m):
            nonlocal counter
            tag = f"<{label}_{counter}>"
            mapping[tag] = m.group(0)
            counter += 1
            return tag
        out = rx.sub(repl, out)
    return out, mapping

def assert_no_phi(text: str) -> bool:
    return not any(rx.search(text) for rx in PATTERNS.values())
