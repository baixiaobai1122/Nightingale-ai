from typing import List, Tuple

def make_dual_summaries(spans: List[Tuple[int, str]]) -> tuple[str, str]:
    """Deterministic tiny summarizer for demo.
    spans: list of (span_idx, redacted_text)
    Returns: (clinician, patient) summaries where every bullet cites [S#].
    """
    bullets_clin = []
    bullets_pat = []
    for idx, txt in spans:
        anchor = f"[S{idx}]"
        bullets_clin.append(f"• Key point: {txt} {anchor}")
        bullets_pat.append(f"• We discussed: {txt} {anchor}")
    return "\n".join(bullets_clin), "\n".join(bullets_pat)
