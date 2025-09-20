# Nightingale AI Test Suite

This directory contains the four micro-tests that validate the quality and user-centric design of the Nightingale AI system.

## Test Overview

### 1. test_grounding.py
**Purpose**: Validate that every summary bullet has a source anchor [S#]
- Ensures traceability of all generated content
- Validates proper source attribution format
- Critical for medical accuracy and accountability

### 2. test_redaction.py  
**Purpose**: Prove no PHI leaks to outputs or logs on synthetic PHI samples
- Tests comprehensive PHI pattern detection
- Validates redaction reversibility
- Essential for HIPAA compliance and patient privacy

### 3. test_latency.py
**Purpose**: Profile redaction and provenance pipeline performance
- Reports P50/P95 latencies for system performance
- Tests scalability with varying input sizes
- Ensures system meets real-time requirements

### 4. test_summary.py
**Purpose**: Validate dual summary templates (clinician vs patient)
- Tests appropriate tone and terminology for each audience
- Ensures content completeness in both formats
- Validates design choices for user-centric summaries

## Running Tests

```bash
# Run all tests
cd backend
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_grounding.py -v
python -m pytest tests/test_redaction.py -v
python -m pytest tests/test_latency.py -v
python -m pytest tests/test_summary.py -v

# Run with coverage
python -m pytest tests/ --cov=backend --cov-report=html
```

## Design Philosophy

These tests embody our commitment to:
- **Quality**: Rigorous validation of core functionality
- **User-Centric Thinking**: Separate templates for different user needs
- **Privacy**: Zero-tolerance for PHI leaks
- **Performance**: Real-time response requirements
- **Traceability**: Every claim must be grounded in source material
```

```typescriptreact file="app/layout.tsx" isDeleted="true"
...deleted...
