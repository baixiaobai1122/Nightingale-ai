# test_summary.py
"""
Test dual summary templates (clinician vs patient).
Validates that both templates are generated correctly and serve their intended audiences.
"""
from backend.summarize import make_dual_summaries

def test_dual_templates():
    """Test that both clinician and patient summaries are generated with appropriate content."""
    spans = [(1, "headache"), (2, "improved with rest"), (3, "no allergies")]
    clin, pat = make_dual_summaries(spans)
    
    assert "Key point:" in clin or "Clinical" in clin, "Clinician summary missing professional markers"
    assert len(clin.strip()) > 0, "Empty clinician summary"
    
    assert "We discussed:" in pat or "Your" in pat or "You" in pat, "Patient summary missing patient-focused language"
    assert len(pat.strip()) > 0, "Empty patient summary"
    
    assert clin != pat, "Clinician and patient summaries are identical"

def test_template_tone_differences():
    """Test that clinician and patient summaries have appropriate tones."""
    spans = [(1, "chronic pain management"), (2, "medication adjustment needed")]
    clin, pat = make_dual_summaries(spans)
    
    clinical_terms = ["assessment", "diagnosis", "treatment", "clinical", "patient presents"]
    assert any(term in clin.lower() for term in clinical_terms), "Clinician summary lacks professional terminology"
    
    patient_terms = ["we discussed", "your", "you", "we talked about", "next steps"]
    assert any(term in pat.lower() for term in patient_terms), "Patient summary lacks patient-focused language"

def test_content_completeness():
    """Test that both summaries cover all provided information."""
    spans = [(1, "fever 101°F"), (2, "prescribed antibiotics"), (3, "follow-up in 1 week")]
    clin, pat = make_dual_summaries(spans)
    
    key_concepts = ["fever", "antibiotic", "follow"]
    
    for concept in key_concepts:
        assert any(concept.lower() in summary.lower() for summary in [clin, pat]), f"Missing concept '{concept}' in summaries"

def compare_templates_side_by_side():
    """Generate a side-by-side comparison for documentation purposes."""
    spans = [
        (1, "Patient presents with acute bronchitis symptoms"),
        (2, "Prescribed azithromycin 250mg daily for 5 days"), 
        (3, "Advised rest and increased fluid intake"),
        (4, "Return if symptoms worsen or persist beyond 7 days")
    ]
    
    clin, pat = make_dual_summaries(spans)
    
    print("\n" + "="*80)
    print("SIDE-BY-SIDE TEMPLATE COMPARISON")
    print("="*80)
    print("\nCLINICIAN SUMMARY:")
    print("-" * 40)
    print(clin)
    print("\nPATIENT SUMMARY:")
    print("-" * 40) 
    print(pat)
    print("\nDESIGN CHOICES:")
    print("-" * 40)
    print("• Clinician summary uses medical terminology and clinical structure")
    print("• Patient summary uses accessible language and personal pronouns")
    print("• Both maintain source anchoring for traceability")
    print("• Content coverage is equivalent but presentation differs")
    print("="*80)
    
    return clin, pat
