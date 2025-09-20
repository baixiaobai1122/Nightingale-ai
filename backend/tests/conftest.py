"""
Pytest configuration and shared fixtures for the test suite.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture
def sample_medical_text():
    """Fixture providing sample medical consultation text."""
    return """
    Patient John Doe presented with chief complaint of persistent cough for 3 days.
    No fever reported. Patient took panadol for symptom relief.
    Physical examination revealed clear lung sounds.
    Diagnosed with viral upper respiratory infection.
    Advised rest and hydration. Follow-up if symptoms persist beyond 7 days.
    """

@pytest.fixture
def sample_phi_text():
    """Fixture providing text with various PHI patterns."""
    return """
    Patient: Jane Smith
    DOB: 1985-03-15
    Phone: +1-555-123-4567
    Email: jane.smith@email.com
    SSN: 123-45-6789
    Address: 123 Oak Street, Springfield, IL 62701
    """
