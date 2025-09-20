from typing import List, Dict
from .models import Segment

def index_provenance(segments_text: List[str]) -> List[Dict]:
    """Create span indices with [S#] mapping. Demo: 1 span per segment."""
    spans = []
    for i, txt in enumerate(segments_text, start=1):
        spans.append({"span_idx": i, "anchor": f"[S{i}]", "text": txt})
    return spans
