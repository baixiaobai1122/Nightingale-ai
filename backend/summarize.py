"""
summarize.py
------------
Summarization pipeline using a finetuned medical dialogue → SOAP model.
Model: omi-health/sum-small (Phi-3-mini finetuned on medical-dialogue-to-soap-summary)

每个摘要条目必须包含源锚点 [S#]，以保证可追溯性。
"""

from typing import List, Tuple
from transformers import pipeline

# === Model setup ===
MODEL_NAME = "omi-health/sum-small"
summarizer = pipeline("text-generation", model=MODEL_NAME, tokenizer=MODEL_NAME)


def make_dual_summaries(spans: List[Tuple[int, str]]) -> tuple[str, str]:
    """
    Generate both clinician-facing (SOAP note) and patient-facing summaries with source anchors.
    
    Args:
        spans: list of (span_idx, redacted_text)
    Returns:
        (clinician_summary, patient_summary)
    """

    # --- 1. 拼接带 [S#] 的对话 ---
    dialogue_text = "\n".join([f"[S{idx}] {txt}" for idx, txt in spans])

    # --- 2. Clinician summary (SOAP style) ---
    clinician_summary = _generate_summary(
        "Create a concise SOAP medical note from this dialogue. "
        "Each bullet point must explicitly cite its source anchor [S#].\n\n"
        f"Dialogue:\n{dialogue_text}"
    )

    # --- 3. Patient summary (friendly version) ---
    patient_summary = _generate_summary(
        "Rewrite the following SOAP note into simple patient-friendly language. "
        "Keep the [S#] source anchors in each bullet point.\n\n"
        f"{clinician_summary}"
    )

    return clinician_summary, patient_summary


def _generate_summary(prompt: str) -> str:
    """
    Generate a summary using the finetuned summarization model.
    
    Args:
        prompt: input instruction + dialogue
    Returns:
        generated summary text
    """
    result = summarizer(
        prompt,
        max_new_tokens=512,
        do_sample=False,
        truncation=True
    )

    # Hugging Face `text-generation` pipeline usually returns `generated_text`
    return result[0].get("generated_text", result[0].get("text", ""))
