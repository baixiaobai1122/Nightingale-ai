"""
test_latency.py
---------------
Profile the summarization pipeline for performance.
Reports P50/P95 latencies to ensure system meets performance requirements.
"""

import time
import statistics
from datasets import load_dataset
from backend.summarize import make_dual_summaries


def load_real_medical_data(num_samples=5, split="train"):
    """
    Load dialogues from omi-health/medical-dialogue-to-soap-summary.
    Returns: list[str] dialogues
    """
    ds = load_dataset("omi-health/medical-dialogue-to-soap-summary", split=split)
    n = min(num_samples, len(ds))
    dialogues = [ds[i]["dialogue"] for i in range(n) if ds[i].get("dialogue")]
    return dialogues


def profile_once(dialogue: str) -> float:
    """
    Run one profiling iteration on a dialogue.
    Returns: latency in ms
    """
    t0 = time.perf_counter()
    clin, pat = make_dual_summaries(dialogue)
    t1 = time.perf_counter()

    assert len(clin) > 0, "Empty clinician summary"
    assert len(pat) > 0, "Empty patient summary"

    return (t1 - t0) * 1000.0


def test_performance_benchmarks(num_samples=10):
    """Test that the system meets performance benchmarks."""
    print("\n=== LATENCY TEST: Performance Benchmarks ===")
    dialogues = load_real_medical_data(num_samples)

    latencies = [profile_once(d) for d in dialogues]

    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(0.95 * len(latencies)) - 1]
    print(f"P50 latency: {p50:.2f}ms, P95 latency: {p95:.2f}ms")

    assert p50 < 5000, f"P50 {p50:.2f}ms exceeds threshold"
    assert p95 < 10000, f"P95 {p95:.2f}ms exceeds threshold"
    print("✅ PASS: Performance benchmarks met")


def test_scalability():
    """Test performance with increasing input sizes."""
    print("\n=== LATENCY TEST: Scalability Analysis ===")
    dialogues = load_real_medical_data(3)

    sizes = [200, 500, 1000]  # character lengths
    for size in sizes:
        truncated = dialogues[0][:size]
        latency = profile_once(truncated)
        print(f"Dialogue length {size} chars → {latency:.2f}ms")
