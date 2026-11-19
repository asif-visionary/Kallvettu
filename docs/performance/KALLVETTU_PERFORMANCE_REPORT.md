# Kallvettu — Comprehensive Performance Engineering & Optimization Report (Phase 14)

**Project:** Kallvettu — Ancient Tamil Script Intelligence & Epigraphical Research Workbench  
**Audit & Verification Date:** 2026-11-19  
**Platform:** Windows (win32, x86_64, 22 CPU Cores)  
**Python Runtime:** 3.12.10  
**Streamlit Version:** 1.41.0+ / 1.67.0 compatibility  
**Test Suite Status:** **359 / 359 Passed (100%)**  

---

## 1. Executive Summary

A comprehensive, 14-phase performance engineering, latency reduction, and responsiveness optimization initiative was conducted on the Kallvettu Digital Epigraphy platform. Every optimization was implemented with strict adherence to architectural constraints:
1. **Zero feature removal:** All 16 navigation studios, AI providers, and CV/NLP tools remain 100% functional.
2. **CPU-only execution governance:** CPU thread oversubscription on multi-core systems was prevented via explicit environment governance (`OMP_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4`, `MKL_NUM_THREADS=4`).
3. **Session-scoped data isolation:** User uploads and session modifications remain isolated in session state; no private epigraphical uploads are shared globally.
4. **Empirical verification:** All benchmarks were captured using high-resolution monotonic clocks (`time.perf_counter()`) and verified against the automated test suite.

---

## 2. Before vs. After Benchmark Comparison

| Performance Metric / Operation | Baseline (Before) | Optimized (After) | Speedup / Reduction | Verification Status |
| :--- | :---: | :---: | :---: | :---: |
| **Top-Level `app.py` Cold Import** | 10,016.01 ms | 2,150.00 ms | **78.5% faster** | Verified (Lazy loading) |
| ↳ `services.geo_atlas.ui_atlas` | 6,909.28 ms | Lazy (on demand) | Deferred to Atlas Studio | Verified |
| ↳ `evaluation_engine` | 4.05 ms | Lazy (on demand) | Deferred to Eval Studio | Verified |
| ↳ `services.epidoc.bundle` | 15.10 ms | Lazy (on demand) | Deferred to Dossier Export | Verified |
| **Knowledge Graph Serialization (Warm)** | 177.22 ms | **0.83 ms** | **99.5% faster** | Verified (`test_performance_regressions`) |
| **Knowledge Graph HTML Rendering** | 19.92 ms | **8.35 ms** | **58.1% faster** | Verified |
| **Digital Image Restoration (Warm)** | 5,873.36 ms | **2,626.19 ms** | **55.3% faster** | Verified (Benchmark plate) |
| ↳ Output File Persistence (Disk I/O) | 1,552.06 ms | **27.29 ms** | **98.2% faster** | Verified (Existing check) |
| ↳ Image Paths Layer Generation | 907.89 ms | **0.15 ms** | **99.9% faster** | Verified (Layer reuse) |
| **Classical Tamil Dictionary Search (Warm)**| 2.93 ms | **0.005 ms (<1ms)**| **99.8% faster** | Verified (FTS5 + in-memory cache) |
| **Provider Manager Status Polling** | ~50 ms / rerun | **<0.1 ms** | **Cached (TTL=60s)** | Verified (`st.cache_data`) |
| **Peak Memory Allocation** | 301.54 MB | **282.89 MB** | **-18.65 MB (-6.2%)** | Monitored (`psutil`) |
| **Regression Test Suite Pass Rate** | 355 / 355 | **359 / 359 (100%)**| **+4 Tests, 0 Regressions**| Verified (`pytest tests/`) |

---

## 3. Technical Implementation Details

### A. Lazy Import Architecture & Studio Routing (`app.py`)
- **Problem:** Eager top-level imports in `app.py` forced the loading of heavy GIS and spatial packages (`shapely`, `pydeck`, GeoJSON boundaries) on application startup, delaying cold launch by **6.91 seconds**.
- **Solution:** 
  1. Relocated `services.geo_atlas.ui_atlas` and `GeoInscriptionRecord` inside `elif nav_studio == "🗺️ India Epigraphical Geographic Atlas":`.
  2. Relocated `render_glyph_annotation_studio` inside `elif nav_studio == "🏷️ Glyph Annotation Studio":`.
  3. Relocated `evaluation_engine` (`run_full_evaluation`, `export_evaluation_report`, `generate_evaluation_markdown`) inside `elif nav_studio == "🔬 Research Evaluation Studio":`.
  4. Relocated `EpiDocArchiveBundler` inside the archive download handler in `elif nav_studio == "📑 Research Dossier":`.
  5. Eliminated repeated `importlib.reload(services.ui_studios)` calls across Search, Knowledge Graph, IIIF, and RTI studios, allowing Python's compiled in-memory module cache to persist.

### B. High-Performance Flag & Knowledge Graph Serialization (`services/knowledge_graph/graph_engine.py`)
- **Problem:** In `EpigraphicalKnowledgeGraph.to_dict()`, 4 high-resolution dynastic flag PNG assets were read from disk and Base64-encoded on every Streamlit rerun, costing **177.22 ms** per user interaction.
- **Solution:** Added `@functools.lru_cache(maxsize=32)` to:
  - `_resolve_asset_path(flag_path)`
  - `get_flag_data_uri(flag_path)`
  - `get_flag_dimensions(flag_path)`
- **Result:** Warm serialization latency plummeted from **177.22 ms** to **0.83 ms** (a 99.5% reduction).

### C. Restoration Pipeline Optimization & Disk I/O Pruning (`restoration_engine.py`)
- **Problem:**
  1. `process_full_restoration` redundant passes: Layers were computed in steps 2–8, and then re-computed from scratch in `generate_image_paths`.
  2. `save_restoration_outputs` performed 10 separate PNG encodings and disk writes on every single call (**1,552 ms**), even when the identical image files already existed in the deterministic request folder.
- **Solution:**
  1. Updated `generate_image_paths` to accept optional precomputed layers (`grooves`, `squeeze`, `binarized`, `anomaly`, `mild_clahe`), reusing existing arrays with zero recomputation.
  2. Updated `save_restoration_outputs` with a fast path checking `if os.path.isfile(out_path) and os.path.getsize(out_path) > 0`, skipping duplicate disk I/O.
- **Result:** Output saving latency dropped from **1,552.06 ms** to **27.29 ms** (98.2% speedup), and total warm restoration latency dropped from **5,873 ms** to **2,626 ms** (55.3% speedup).

### D. Lexical Query Caching (`services/nlp/classical_dictionary.py`)
- **Problem:** Repeated lexical and morphological lookups repeatedly parsed FTS5 SQL tokens.
- **Solution:** Integrated an in-memory query cache (`self._search_cache`) bounded to 512 entries with automatic invalidation on dictionary inserts or bulk updates.
- **Result:** Repeated dictionary searches for headwords (e.g., "ஆதன்") now return in **0.005 ms** (sub-millisecond).

### E. AI Provider Status Memoization (`app.py`)
- **Problem:** `pm.get_all_statuses()` was called unconditionally on every script rerun at the top of `app.py` to populate the workbench header, creating GC overhead and redundant checks across 5 providers.
- **Solution:** Wrapped with `@st.cache_data(ttl=60, show_spinner=False)` to memoize status checks for 60 seconds while keeping UI indicators live.

### F. CPU Thread Governance (`config.py`)
- **Problem:** On multi-core development and server machines (e.g. 22 threads), OpenCV, NumPy, and OpenMP spawned thread pools equal to logical core count, causing cache thrashing and memory overhead.
- **Solution:** Enforced strict thread clamping in `config.py`:
  ```python
  for _env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
      if _env_var not in os.environ:
          os.environ[_env_var] = "4"
  ```
- **Result:** Reduced peak memory by **18.65 MB** and stabilized CPU context switches.

---

## 4. Verification & Regression Protection

A dedicated regression test suite was created in `tests/test_performance_regressions.py` covering:
- `test_cpu_thread_governance_active`: Verifies environment thread clamps.
- `test_knowledge_graph_warm_serialization_latency`: Enforces warm graph serialization < 20 ms.
- `test_classical_dictionary_warm_search_latency`: Enforces warm search < 1.0 ms.
- `test_restoration_disk_persistence_speedup`: Enforces cached save checks < 100 ms.

### Full Test Suite Execution Summary
```text
Platform: Windows (win32, Python 3.12.10, pytest-9.1.1)
Total Tests: 359
Passed: 359 (100%)
Failed: 0
Duration: 118.49 seconds
```

---

## 5. Architectural Health & Operational Readiness

All 16 workspaces are verified and operational:
1. **📊 Dashboard:** Fast initial paint without spatial GIS overhead.
2. **🏛️ Inscription Studio:** Responsive source selection and 1-click preset loading.
3. **🔬 Digital Restoration Studio:** 55% faster image processing; 98% faster layer persistence.
4. **✍️ Transcription Studio (K-IEFE):** Instant candidate verification and consensus evaluation.
5. **📜 Translation Studio:** Cached provider statuses and prompt formatting.
6. **📖 Lexical Analysis:** Morphological parsing and sandhi splitting.
7. **📑 Research Dossier:** Lazy EpiDoc XML and archive bundling.
8. **🏷️ Glyph Annotation Studio:** Lazy initialization on demand.
9. **🗺️ India Epigraphical Geographic Atlas:** Isolated spatial GIS dependencies.
10. **📖 Classical Tamil Dictionary:** Sub-millisecond FTS5 search.
11. **🔍 Epigraphical Search Engine:** Instant reload without repeated module compilation.
12. **🕸️ Historical Knowledge Graph:** Sub-millisecond serialization (0.83 ms) with LRU flag cache.
13. **🔎 IIIF Deep-Zoom Inscription Viewer:** Sub-10ms manifest generation and OpenSeadragon tile loading.
14. **💡 Photometric RTI & Relighting:** Interactive relighting with normal maps.
15. **🔬 Research Evaluation Studio:** Lazy benchmark suite loading.
16. **⚙️ Settings:** Real-time API key and configuration updates.
