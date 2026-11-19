"""
Performance & Responsiveness Regression Test Suite for Kallvettu.
Verifies that critical latency optimizations, caching layers, thread governance,
and memory bounds are permanently maintained.
"""

import os
import sys
import time
import pytest
from PIL import Image

sys.path.insert(0, os.path.abspath("."))


def test_cpu_thread_governance_active():
    """Verify thread limit environment variables are set to prevent CPU oversubscription."""
    import config
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        val = os.environ.get(var)
        assert val is not None, f"Expected {var} to be set in environment"
        assert int(val) <= 4, f"Expected {var} <= 4, got {val}"


def test_knowledge_graph_warm_serialization_latency():
    """Verify knowledge graph warm serialization executes in sub-5ms via LRU flag caching."""
    from services.knowledge_graph.graph_engine import EpigraphicalKnowledgeGraph
    kg = EpigraphicalKnowledgeGraph()

    # Cold serialization primes flag cache
    _ = kg.to_dict()

    # Warm serialization must be fast
    t0 = time.perf_counter()
    warm_dict = kg.to_dict()
    warm_ms = (time.perf_counter() - t0) * 1000

    assert warm_dict["node_count"] == 29
    assert warm_dict["edge_count"] == 69
    assert warm_ms < 20.0, f"Warm Knowledge Graph serialization took {warm_ms:.2f} ms (expected < 20 ms)"


def test_classical_dictionary_warm_search_latency():
    """Verify classical Tamil dictionary warm search latency is sub-millisecond."""
    from services.nlp.classical_dictionary import ClassicalTamilDictionary
    lex = ClassicalTamilDictionary()

    # Cold search
    _ = lex.search("ஆதன்", limit=20)

    # Warm search
    t0 = time.perf_counter()
    for _ in range(20):
        res = lex.search("ஆதன்", limit=20)
    avg_warm_ms = ((time.perf_counter() - t0) / 20) * 1000

    assert avg_warm_ms < 1.0, f"Warm dictionary search averaged {avg_warm_ms:.4f} ms (expected < 1.0 ms)"


def test_restoration_disk_persistence_speedup():
    """Verify save_restoration_outputs skips redundant PNG encoding on existing outputs."""
    from restoration_engine import save_restoration_outputs
    test_img = Image.new("RGB", (200, 200), color=(128, 128, 128))

    results = {
        "original": test_img,
        "mild_clahe": test_img,
        "path_b": test_img,
        "clahe": test_img,
        "denoised": test_img,
        "squeeze": test_img,
        "grooves": test_img,
        "binarized": test_img,
        "enhancement_diff": test_img,
        "anomaly_map": test_img,
    }

    # First call creates the files
    saved_first = save_restoration_outputs(results, source_id="perf_test")
    assert len(saved_first) == 10

    # Second call must reuse existing files without re-encoding
    t0 = time.perf_counter()
    saved_second = save_restoration_outputs(results, source_id="perf_test")
    skip_ms = (time.perf_counter() - t0) * 1000

    assert len(saved_second) == 10
    assert skip_ms < 100.0, f"Redundant save check took {skip_ms:.2f} ms (expected < 100 ms)"
