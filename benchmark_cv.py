"""
Scientific Performance Benchmark for Kallvettu Computer Vision Restoration Engine.
Measures wall-clock latency, throughput, and memory allocation across:
- 1024 x 768 (Standard document scan / rubbing resolution)
- 1920 x 1080 (Full HD museum photograph)
- 4000 x 3000 (12MP high-resolution camera photo with automatic dimension clamping)
"""

import time
import tracemalloc
import numpy as np
from PIL import Image
from restoration_engine import process_full_restoration


def run_benchmark():
    print("=" * 70)
    print("KALLVETTU CV ENGINE PERFORMANCE BENCHMARK")
    print("=" * 70)

    test_resolutions = [
        (1024, 768, "Standard Document Resolution"),
        (1920, 1080, "Full HD Museum Photograph"),
        (4000, 3000, "12MP Smartphone Camera Photo (Clamped)")
    ]

    results = []

    for width, height, label in test_resolutions:
        print(f"\nBenchmarking {width}x{height} px ({label})...")
        # Synthesize realistic granite-like RGB array
        np.random.seed(42)
        base = np.full((height, width, 3), [115, 110, 105], dtype=np.uint8)
        noise = np.random.randint(-20, 20, (height, width, 3), dtype=np.int16)
        test_arr = np.clip(base + noise, 0, 255).astype(np.uint8)
        test_img = Image.fromarray(test_arr, mode="RGB")

        # Start memory tracking
        tracemalloc.start()
        start_time = time.perf_counter()

        # Run full pipeline with max_dim=1600
        output = process_full_restoration(
            test_img,
            contrast_boost=2.5,
            binarize_k=0.20,
            squeeze_mode="charcoal_on_squeeze",
            max_dim=1600,
            source_identifier=f"bench_{width}x{height}"
        )

        elapsed = time.perf_counter() - start_time
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        meta = output["metadata"]
        res_info = {
            "resolution": f"{width}x{height}",
            "working_res": f"{meta['working_dimensions'][0]}x{meta['working_dimensions'][1]}",
            "downsampled": meta["was_downsampled"],
            "elapsed_seconds": round(elapsed, 4),
            "peak_memory_mb": round(peak_mem / (1024 * 1024), 2),
            "throughput_mpixels_per_sec": round((width * height / 1e6) / elapsed, 2)
        }
        results.append(res_info)

        print(f"  Working Resolution: {res_info['working_res']}")
        print(f"  Downsampled: {res_info['downsampled']}")
        print(f"  Latency: {res_info['elapsed_seconds']} s")
        print(f"  Peak Memory: {res_info['peak_memory_mb']} MB")
        print(f"  Throughput: {res_info['throughput_mpixels_per_sec']} MP/s")

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Input Resolution':<18} | {'Working Resolution':<20} | {'Latency (s)':<12} | {'Peak RAM (MB)':<14}")
    print("-" * 70)
    for r in results:
        print(f"{r['resolution']:<18} | {r['working_res']:<20} | {r['elapsed_seconds']:<12.3f} | {r['peak_memory_mb']:<14.2f}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_benchmark()

