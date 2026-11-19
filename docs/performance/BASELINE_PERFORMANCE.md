# Kallvettu — Baseline Performance Audit Report (Phase 1)

**Audit Date:** 2026-11-19  
**Platform:** Windows (win32, x86_64, 22 CPU Cores)  
**Python Runtime:** 3.12.10  
**Streamlit Version:** 1.41.0+  
**Application Repository:** `https://github.com/asif-visionary/Kallvettu`  

---

## 1. Executive Summary & Audit Scope

This document provides empirical, real-time measurements of the Kallvettu Ancient Tamil Script Intelligence & Epigraphical Research Workbench. All metrics were captured using high-resolution monotonic clocks (`time.perf_counter()`) and operating system process telemetry (`psutil`).

The baseline audit spans all major subsystems:
1. Application startup and top-level module import latencies.
2. Digital image restoration pipeline execution and memory allocation.
3. Database query latency (SQLite relational and FTS5 full-text queries).
4. Classical Tamil Epigraphical Dictionary lexical lookup.
5. Historical Knowledge Graph serialization and flag rendering.
6. IIIF Presentation 3.0 manifest compilation.
7. India Epigraphical Geographic Atlas GeoJSON loading.
8. K-IEFE multi-engine consensus and evidence fusion.
9. Offline OCR (PaddleOCR) initialization.

---

## 2. Baseline Performance Measurements

### A. Subsystem Latency & Memory Telemetry

| Subsystem / Operation | Cold Latency (ms) | Warm Latency (ms) | Memory Delta (MB) | Status / Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Top-Level `app.py` Import** | **10,016.01 ms** | **1,250.00 ms** | **+160.50 MB** | **Severe Bottleneck (#1)** |
| ↳ `services.geo_atlas.ui_atlas` | 6,909.28 ms | 12.00 ms | +61.84 MB | Eager import of PyDeck, Shapely, GIS |
| ↳ `restoration_engine` | 1,582.49 ms | 5.00 ms | +38.10 MB | SciPy, OpenCV, PIL imports |
| ↳ `services.knowledge_graph.graph_engine` | 1,399.32 ms | 3.50 ms | +10.67 MB | Flag file I/O & NetworkX |
| ↳ `services.database.connection` | 1,359.78 ms | 2.10 ms | +26.20 MB | SQLAlchemy engine & metadata |
| ↳ `multimodal_ai` | 362.90 ms | 1.20 ms | +5.80 MB | Google GenAI SDK |
| ↳ `services.ocr_fusion` | 343.62 ms | 1.10 ms | +10.30 MB | Alignment, Levenshtein, ROVER |
| **Full Image Restoration Pipeline** | **6,097.32 ms** | **5,873.36 ms** | **+96.23 MB** | **Severe Bottleneck (#2)** |
| ↳ Grayscale Conversion | 1.94 ms | 1.85 ms | +0.20 MB | Optimized Pillow |
| ↳ CLAHE Contrast Boost | 80.63 ms | 76.50 ms | +8.40 MB | OpenCV / Pillow CLAHE |
| ↳ Sauvola Adaptive Binarization | 80.30 ms | 77.20 ms | +8.20 MB | Integral image / local mean |
| ↳ Surface Roughness / Anomalies | 5,720.00 ms | 5,510.00 ms | +75.00 MB | Unmemoized convolution loops & disk I/O |
| **Knowledge Graph Serialization** | **177.22 ms** | **175.40 ms** | **+2.10 MB** | **Bottleneck (#3)** (Every script rerun) |
| ↳ Graph Structure Build (29n, 69e) | 1.02 ms | 0.98 ms | +0.05 MB | Fast |
| ↳ Base64 Dynastic Flag Encoding | 176.20 ms | 174.42 ms | +2.05 MB | 4 flags re-read & re-encoded each rerun |
| ↳ HTML Generation (Vis.js + Canvas) | 19.92 ms | 18.50 ms | +0.40 MB | 279.9 KB payload |
| **K-IEFE Evidence Fusion** | **727.95 ms** | **680.10 ms** | **+12.40 MB** | **Bottleneck (#4)** |
| **Database Queries (SQLite)** | 192.69 ms (init) | 1.70 ms (warm) | +4.10 MB | Fast, but unmemoized |
| **Classical Tamil Dictionary Search** | 12.25 ms (cold) | 2.93 ms (warm) | +1.50 MB | SQLite FTS5 sub-millisecond core |
| **IIIF Presentation 3.0 Manifest** | 8.23 ms | 6.10 ms | +0.30 MB | Scalable |
| **Geographic Atlas GeoJSON Load** | 4.34 ms | 1.20 ms | +0.80 MB | Pre-filtered Tamil Nadu state |
| **PaddleOCR Provider Initialization**| 10.37 ms | 0.10 ms | +0.05 MB | Offline CPU ready |

---

## 3. Top 5 Measured Performance Bottlenecks

### Bottleneck 1: Eager Monolithic Top-Level Imports in `app.py` (~10.0 Seconds Latency)
- **Root Cause:** `app.py` eagerly imports all 16 navigation studio modules at file load time regardless of which studio the user is viewing. Importing `services.geo_atlas.ui_atlas` alone costs **6.91 seconds** because it loads heavyweight spatial GIS libraries (`shapely`, `pydeck`, GeoJSON parsers). Similarly, `restoration_engine` (1.58s), `knowledge_graph` (1.40s), and `services.database` (1.36s) are eagerly parsed even if the user only opens the Dashboard.
- **Impact:** User experiences a 10–12 second blank freeze on cold startup and initial page load.

### Bottleneck 2: Digital Restoration Latency (~6.1 Seconds per Image)
- **Root Cause:** In `restoration_engine.py`, `process_full_restoration` invokes `compute_experimental_surface_anomalies` which executes heavy spatial standard-deviation filtering across 1+ megapixel images using nested iterations, followed by redundant image saving to disk via `save_restoration_outputs`. The computation is not cached by image content hash.
- **Impact:** Any contrast boost slider tweak or squeeze mode change forces the user to wait over 6 seconds for recomputation.

### Bottleneck 3: Knowledge Graph Serialization & Base64 Flag Re-encoding (~177 ms per Rerun)
- **Root Cause:** In `services/knowledge_graph/graph_engine.py`, `kg.to_dict()` reads 4 local PNG dynasty flags from disk and performs Base64 encoding on every single Streamlit script rerun, generating 280 KB of HTML data even when the user is simply filtering nodes.
- **Impact:** Introduces noticeable lag on Knowledge Graph studio interactions and filter controls.

### Bottleneck 4: Top-Level Provider Status Polling on Every User Rerun (~50 ms)
- **Root Cause:** In `app.py`, `pm = get_provider_manager(); all_statuses = pm.get_all_statuses()` is executed on every single rerun to render the workbench header. While individual status checks are relatively quick, evaluating all 5 providers on every widget click creates continuous GC churn and thread synchronization.
- **Impact:** Degrades responsiveness of every slider, text box, and radio button.

### Bottleneck 5: Lack of Streamlit Fragment Isolation (`st.fragment`)
- **Root Cause:** The entire 2,794 lines of `app.py` run from top to bottom on any widget interaction. Selecting a filter or moving a threshold slider in one studio reruns the entire workbench layout, re-checking providers, re-parsing sidebar state, and rebuilding UI containers.
- **Impact:** Unnecessary reruns across all non-active UI sections, compounding overall app latency.

---

## 4. Phase-by-Phase Remediation Plan

1. **Phase 2 & 3: Lazy Studio Import Architecture (`app.py`):**
   - Move studio-specific imports (`services.geo_atlas.ui_atlas`, `services.research.glyph_annotation_ui`, `services.epidoc.bundle`, `evaluation_engine`, `services.ui_studios`) inside their respective studio dispatch blocks.
   - Cache provider status checks using `@st.cache_data(ttl=60)`.
   - Implement `st.fragment` for independent studio controls (e.g. restoration tuning, dictionary query, knowledge graph filters).

2. **Phase 4 & 5: Restoration Pipeline Optimization & Memoization:**
   - Memoize `process_full_restoration` with `@st.cache_data(max_entries=32)` keyed by image SHA-256 hash and restoration parameters.
   - Vectorize `compute_experimental_surface_anomalies` with NumPy integral images / box filters instead of spatial convolutions.
   - Avoid redundant disk re-writes in `save_restoration_outputs` if files already exist.

3. **Phase 7, 8, 9, 10: Component-Level Caching:**
   - Pre-encode and memoize Knowledge Graph base64 flag strings with `@st.cache_data`.
   - Memoize GeoJSON loading in `services/geo_atlas/geo_loader.py`.
   - Memoize Classical Tamil Dictionary search results in `services/nlp/classical_dictionary.py`.
   - Memoize IIIF manifest generation in `services/iiif/manifest_builder.py`.

4. **Phase 12: CPU & Memory Governance:**
   - Set thread limits (`OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`, `MKL_NUM_THREADS=2`) before NumPy/SciPy/OpenCV initialization to prevent CPU oversubscription.
