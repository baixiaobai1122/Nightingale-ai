# test_summary.py
"""
Test dual summary templates (clinician vs patient).
Validates that both templates are generated correctly and serve their intended audiences.
"""
from backend.summarize import make_dual_summaries
from datasets import load_dataset


def load_real_medical_data(num_samples=5, split="test"):
    """Load real medical dialogue data from Hugging Face dataset."""
    ds = load_dataset("omi-health/medical-dialogue-to-soap-summary", split=split)
    conversations = []
    for i in range(min(num_samples, len(ds))):
        dialogue = ds[i]["dialogue"]
        if dialogue and isinstance(dialogue, str):
            conversations.append([(1, dialogue)])
    return conversations, f"omi-health/medical-dialogue-to-soap-summary ({split}, {len(conversations)} samples)"


def test_dual_templates():
    """Test that both clinician and patient summaries are generated with appropriate content."""
    print("\n=== SUMMARY TEST: Dual Template Validation ===")
    conversations, dataset_info = load_real_medical_data(1)
    spans = conversations[0]

    clin, pat = make_dual_summaries(spans)

    print(f"Generated dual summaries:")
    print(f"  - Clinician summary: {len(clin)} characters")
    print(f"  - Patient summary: {len(pat)} characters")

    assert len(clin.strip()) > 0, "Empty clinician summary"
    assert len(pat.strip()) > 0, "Empty patient summary"
    assert clin != pat, "Clinician and patient summaries should differ"

    print("✅ PASS: Dual summaries generated successfully")


def test_template_tone_differences():
    """Test that clinician and patient summaries have appropriate tones."""
    print("\n=== SUMMARY TEST: Tone Differentiation ===")
    conversations, dataset_info = load_real_medical_data(1)
    spans = conversations[0]

    clin, pat = make_dual_summaries(spans)

    # Clinical terms in clinician summary
    clinical_terms = ["assessment", "clinical", "symptoms", "evaluation", "diagnostic"]
    clinical_found = [t for t in clinical_terms if t in clin.lower()]

    # Patient-friendly terms in patient summary
    patient_terms = ["you mentioned", "your", "we discussed", "next steps", "questions"]
    patient_found = [t for t in patient_terms if t in pat.lower()]

    assert clinical_found, "Clinician summary lacks professional terminology"
    assert patient_found, "Patient summary lacks patient-focused language"

    print("✅ PASS: Tone differentiation detected")


def compare_templates_side_by_side():
    """Generate a side-by-side comparison for documentation purposes."""
    print("\n=== SUMMARY TEST: Side-by-Side Template Comparison ===")

    conversations, dataset_info = load_real_medical_data(1)
    spans = conversations[0]

    clin, pat = make_dual_summaries(spans)

    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE TEMPLATE COMPARISON")
    print("=" * 80)
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
    print("• Both maintain source anchoring for traceability (if included in summarizer)")
    print("• Content coverage is equivalent but presentation differs")
    print("=" * 80)


if __name__ == "__main__":
    test_dual_templates()
    test_template_tone_differences()
    compare_templates_side_by_side()
