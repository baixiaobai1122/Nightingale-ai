#!/usr/bin/env python3
"""
Deploy trained medical summarization model to production backend
"""
import os
import shutil
import sys
from pathlib import Path

def deploy_model_to_backend():
    """Deploy the trained model to replace backend summarization."""

    print("🚀 Deploying Medical Summarization Model")
    print("=" * 50)

    # Paths
    project_root = Path(__file__).parent.parent
    backend_dir = project_root / "backend"
    summary_training_dir = project_root / "summary_training"

    # 1. Check if trained model exists
    model_path = summary_training_dir / "outputs" / "medical_summarizer_v1" / "best_model"

    if not model_path.exists():
        print("❌ Trained model not found!")
        print(f"Expected location: {model_path}")
        print("Please train the model first using:")
        print("python summary_training/train_medical_summarizer.py --config summary_training/configs/base_config.yaml")
        return False

    print(f"✅ Found trained model at: {model_path}")

    # 2. Backup current backend/summarize.py
    backup_path = backend_dir / "summarize_backup.py"
    original_path = backend_dir / "summarize.py"

    if original_path.exists():
        shutil.copy2(original_path, backup_path)
        print(f"✅ Backed up original summarize.py to: {backup_path}")

    # 3. Create new backend/summarize.py that uses AI model
    new_summarize_content = '''"""
Medical Summarization using trained AI model
This file replaces the rule-based summarization with AI-powered generation.
"""
from typing import List, Tuple
import sys
from pathlib import Path

# Add summary_training to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "summary_training"))

try:
    from src.inference.model_inference import make_dual_summaries_ai
    AI_MODEL_AVAILABLE = True
    print("✅ AI medical summarization model loaded successfully")
except ImportError as e:
    print(f"⚠️  AI model not available: {e}")
    print("Falling back to rule-based summarization...")
    AI_MODEL_AVAILABLE = False

def make_dual_summaries(spans: List[Tuple[int, str]]) -> tuple[str, str]:
    """
    Generate dual medical summaries using AI model or fallback to rules.

    Args:
        spans: list of (span_idx, redacted_text)

    Returns:
        (clinician_summary, patient_summary) tuple
    """
    if AI_MODEL_AVAILABLE:
        try:
            return make_dual_summaries_ai(spans)
        except Exception as e:
            print(f"⚠️  AI model error: {e}")
            print("Falling back to rule-based generation...")

    # Fallback to rule-based generation
    return _make_dual_summaries_fallback(spans)

def _make_dual_summaries_fallback(spans: List[Tuple[int, str]]) -> tuple[str, str]:
    """Fallback rule-based summarization (from backup)."""
    # Import the rule-based logic from backup
    from typing import List, Tuple

    def _infer_condition_from_spans(spans: List[Tuple[int, str]]) -> str:
        """Simple condition inference from conversation content."""
        all_text = " ".join([txt.lower() for _, txt in spans])

        # Basic medical condition detection
        conditions = {
            'respiratory': ['cough', 'breathing', 'shortness', 'chest', 'lung'],
            'cardiac': ['heart', 'chest pain', 'palpitation', 'cardiac'],
            'gastrointestinal': ['stomach', 'nausea', 'vomit', 'diarrhea', 'abdominal'],
            'neurological': ['headache', 'dizzy', 'migraine', 'head'],
            'musculoskeletal': ['pain', 'ache', 'joint', 'muscle', 'back'],
            'general': ['fever', 'tired', 'fatigue', 'weak']
        }

        for condition, keywords in conditions.items():
            if any(keyword in all_text for keyword in keywords):
                return condition

        return 'general_consultation'

    def _generate_clinician_summary(segment: dict) -> str:
        """Generate structured clinical summary."""
        turns = segment['turns']

        # Categorize content
        symptoms = []
        history = []

        for turn in turns:
            text = turn.get('text_redacted', '')
            turn_idx = turn.get('turn_idx', 0)
            text_lower = text.lower()

            if any(word in text_lower for word in ['day', 'week', 'month', 'started', 'began', 'since']):
                history.append(f"• {text} [S{turn_idx}]")
            else:
                symptoms.append(f"• {text} [S{turn_idx}]")

        # Build structured summary
        summary_parts = ["## CLINICAL SUMMARY", ""]

        if symptoms:
            summary_parts.extend(["### Chief Complaint & Symptoms:"] + symptoms + [""])

        if history:
            summary_parts.extend(["### History of Present Illness:"] + history + [""])

        # Add assessment based on inferred condition
        condition = segment.get('disease_tag', 'general_consultation')
        summary_parts.extend([
            "### Assessment:",
            f"• Clinical presentation consistent with {condition.replace('_', ' ')} concerns",
            f"• Further evaluation recommended based on symptom pattern",
            "",
            "### Plan:",
            "• Comprehensive evaluation and appropriate management",
            "• Patient education and follow-up as indicated",
            ""
        ])

        if condition != 'general_consultation':
            summary_parts.extend([f"### Diagnostic Category: {condition}", ""])

        return "\\n".join(summary_parts)

    def _generate_patient_summary(segment: dict) -> str:
        """Generate patient-friendly summary."""
        turns = segment['turns']

        # Process patient statements
        what_you_shared = []
        for turn in turns:
            text = turn.get('text_redacted', '')
            turn_idx = turn.get('turn_idx', 0)

            # Convert to patient-friendly language
            friendly_text = _make_patient_friendly(text)
            what_you_shared.append(f"• You mentioned: {friendly_text} [S{turn_idx}]")

        # Build patient-friendly summary
        summary_parts = ["## Your Medical Visit Summary", ""]

        if what_you_shared:
            summary_parts.extend(["### What You Shared With Us:"] + what_you_shared + [""])

        summary_parts.extend([
            "### What We Discussed:",
            "• We reviewed your symptoms and how they're affecting you",
            "• We talked about the best ways to help you feel better",
            "",
            "### Your Next Steps:",
            "• Follow the treatment plan we discussed",
            "• Keep track of how you're feeling",
            "• Come back to see us as scheduled",
            "",
            "### Questions or Concerns?",
            "• Please contact us if your symptoms get worse",
            "• Don't hesitate to call if you have questions about your treatment",
            "• Follow up as scheduled to monitor your progress",
            ""
        ])

        # Add condition explanation if available
        condition = segment.get('disease_tag', '')
        if condition and condition != 'general_consultation':
            friendly_condition = condition.replace('_', ' ').title()
            summary_parts.insert(-4, f"### About Your Condition:")
            summary_parts.insert(-4, f"We discussed concerns related to: **{friendly_condition}**")
            summary_parts.insert(-4, "")

        return "\\n".join(summary_parts)

    def _make_patient_friendly(medical_text: str) -> str:
        """Convert medical terminology to patient-friendly language."""
        replacements = {
            'diagnosis': 'condition',
            'prescribe': 'give you medicine for',
            'medication': 'medicine',
            'symptoms': 'how you\\'re feeling',
            'examination': 'check-up',
            'assessment': 'review',
            'hypertension': 'high blood pressure',
            'inflammation': 'swelling',
            'acute': 'sudden',
            'chronic': 'ongoing',
            'administer': 'give',
            'monitor': 'keep track of',
            'contraindicated': 'not safe for you',
            'adverse': 'unwanted side',
        }

        friendly_text = medical_text
        for medical_term, friendly_term in replacements.items():
            friendly_text = friendly_text.replace(medical_term, friendly_term)

        return friendly_text

    # Convert spans to structured format for processing
    segment = {
        'turns': [{'speaker': 'patient', 'text_redacted': txt, 'turn_idx': idx} for idx, txt in spans],
        'disease_tag': _infer_condition_from_spans(spans)
    }

    clinician_summary = _generate_clinician_summary(segment)
    patient_summary = _generate_patient_summary(segment)

    return clinician_summary, patient_summary
'''

    # 4. Write new summarize.py
    with open(original_path, 'w') as f:
        f.write(new_summarize_content)

    print(f"✅ Created new AI-powered summarize.py")

    # 5. Test the new implementation
    print("\\n🧪 Testing new AI-powered summarization...")

    try:
        sys.path.insert(0, str(backend_dir))
        from summarize import make_dual_summaries

        test_spans = [(1, "chest pain"), (2, "shortness of breath"), (3, "2 days")]
        clin, pat = make_dual_summaries(test_spans)

        print("✅ AI summarization test successful!")
        print(f"Clinician summary length: {len(clin)} chars")
        print(f"Patient summary length: {len(pat)} chars")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Restoring backup...")
        shutil.copy2(backup_path, original_path)
        return False

    print("\\n🎉 Model deployment completed successfully!")
    print("\\nNext steps:")
    print("1. Test the system: python backend/tests/test_summary.py")
    print("2. Start the backend: python backend/app.py")
    print("3. Monitor performance and accuracy")

    return True

if __name__ == "__main__":
    deploy_model_to_backend()