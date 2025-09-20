# test_latency.py
"""
Profile the redaction and provenance pipeline for performance.
Reports P50/P95 latencies to ensure system meets performance requirements.
"""
import time
import statistics
from backend.redact import redact_text, assert_no_phi
from backend.summarize import make_dual_summaries

def profile_once(n=50):
    """Profile a single run of the redaction and summarization pipeline."""
    spans = []
    for i in range(n):
        synthetic_text = f"Patient reports symptom {i} for 2 days. Phone +65 9123 45{i%10}{(i+1)%10}. Prescribed medication X."
        red, _ = redact_text(synthetic_text)
        assert assert_no_phi(red), f"PHI leak detected in span {i}"
        spans.append((i+1, red))
    
    t0 = time.perf_counter()
    clin_summary, pat_summary = make_dual_summaries(spans)
    t1 = time.perf_counter()
    
    assert len(clin_summary) > 0, "Empty clinician summary"
    assert len(pat_summary) > 0, "Empty patient summary"
    
    return (t1 - t0) * 1000.0  # Return latency in milliseconds

def test_performance_benchmarks():
    """Test that the system meets performance benchmarks."""
    latencies = [profile_once(100) for _ in range(10)]
    
    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(0.95*len(latencies))-1]
    
    print(f"Performance Results: P50={p50:.2f}ms, P95={p95:.2f}ms")
    
    assert p50 < 5000, f"P50 latency {p50:.2f}ms exceeds 5s threshold"
    assert p95 < 10000, f"P95 latency {p95:.2f}ms exceeds 10s threshold"
    
    return {"P50_ms": p50, "P95_ms": p95}

def test_scalability():
    """Test performance with varying input sizes."""
    sizes = [10, 50, 100, 200]
    results = {}
    
    for size in sizes:
        latency = profile_once(size)
        results[f"n_{size}"] = latency
        print(f"Size {size}: {latency:.2f}ms")
    
    assert results["n_200"] < results["n_100"] * 3, "Performance degrades too quickly with scale"
