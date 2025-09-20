# test_grounding.py
"""
Test to validate that every summary bullet has a source anchor [S#].
This ensures traceability and grounding of all generated content.
"""
from backend.summarize import make_dual_summaries

def test_every_bullet_has_anchor():
    """Test that every line in both summaries contains source anchors."""
    spans = [(1, "cough for 3 days"), (2, "no fever"), (3, "took panadol")]
    clin, pat = make_dual_summaries(spans)
    
    clin_lines = [line.strip() for line in clin.split('\n') if line.strip() and not line.startswith('#')]
    for line in clin_lines:
        if line and '•' in line or line.startswith('-'):  # Only check bullet points
            assert '[S' in line and ']' in line, f"Missing source anchor in clinician line: {line}"
    
    pat_lines = [line.strip() for line in pat.split('\n') if line.strip() and not line.startswith('#')]
    for line in pat_lines:
        if line and '•' in line or line.startswith('-'):  # Only check bullet points
            assert '[S' in line and ']' in line, f"Missing source anchor in patient line: {line}"

def test_anchor_format():
    """Test that anchors follow the correct [S#] format."""
    spans = [(1, "headache improved"), (2, "no allergies reported")]
    clin, pat = make_dual_summaries(spans)
    
    import re
    anchor_pattern = r'\[S\d+\]'
    
    # Check that anchors match the expected pattern
    clin_anchors = re.findall(anchor_pattern, clin)
    pat_anchors = re.findall(anchor_pattern, pat)
    
    assert len(clin_anchors) > 0, "No valid anchors found in clinician summary"
    assert len(pat_anchors) > 0, "No valid anchors found in patient summary"
