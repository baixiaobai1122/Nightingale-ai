"""
test_grounding.py
-----------------
Validation tests for summarization output.
Ensures clinician and patient summaries include [S#] source anchors
for every content bullet.
"""

import re
from datasets import load_dataset
from backend.summarize import make_dual_summaries


def load_real_medical_data(num_samples=2, split="train"):
    """
    Load dialogues from omi-health/medical-dialogue-to-soap-summary.
    Returns: list of dialogues (each as str)
    """
    ds = load_dataset("omi-health/medical-dialogue-to-soap-summary", split=split)
    n = min(num_samples, len(ds))
    dialogues = [ds[i]["dialogue"] for i in range(n) if ds[i].get("dialogue")]
    return dialogues


def _check_anchors(summary: str, summary_name: str):
    """Check that each bullet has a valid [S#] anchor."""
    lines = [line.strip() for line in summary.split("\n") if line.strip()]
    bullet_lines = [line for line in lines if line.startswith("•")]

    if not bullet_lines:
        raise AssertionError(f"{summary_name} has no bullet points")

    for line in bullet_lines:
        if "[S" not in line or "]" not in line:
            raise AssertionError(f"Missing [S#] anchor in {summary_name}: {line}")
        if not re.search(r"\[S\d+\]", line):
            raise AssertionError(f"Invalid anchor format in {summary_name}: {line}")

    print(f"✅ PASS: {summary_name} has {len(bullet_lines)} bullets, all with valid anchors.")


def test_every_bullet_has_anchor():
    """Validate clinician and patient summaries both contain [S#] anchors."""
    print("\n=== GROUNDING TEST: Source Anchor Validation ===")

    dialogues = load_real_medical_data(1)
    dialogue = dialogues[0]

    spans = [(i+1, line) for i, line in enumerate(dialogue.split("\n")) if line.strip()]
    clin, pat = make_dual_summaries(spans)

    _check_anchors(clin, "Clinician summary")
    _check_anchors(pat, "Patient summary")


if __name__ == "__main__":
    try:
        test_every_bullet_has_anchor()
        print(" All grounding tests passed successfully.")
    except AssertionError as e:
        print(f" Grounding test failed: {e}")
