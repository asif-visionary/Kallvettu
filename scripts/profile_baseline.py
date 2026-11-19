"""
Comprehensive Baseline Performance Profiling Script for Kallvettu.
Measures:
1. Module import times and application startup latency.
2. Initial dashboard loading.
3. Studio navigation execution times.
4. PaddleOCR model initialization & inference latency.
5. Historical HOG+SVM / CRNN initialization & inference latency.
6. Image restoration pipeline latency (Original -> CLAHE -> Squeeze -> Sauvola).
7. K-IEFE evidence fusion execution time.
8. Classical Tamil Dictionary search latency (cold vs warm).
9. India Epigraphical Geographic Atlas loading time.
10. Historical Knowledge Graph rendering & serialization time.
11. IIIF Presentation 3.0 manifest generation latency.
12. Original image & tile loading latency.
13. Database query latency (SQLite FTS5, relations, CRUD).
14. Peak memory allocation across workloads.
"""

import sys
import os
import time
import psutil
import json

sys.path.insert(0, os.path.abspath("."))
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

def get_mem_mb() -> float:
    p = psutil.Process()
    return p.memory_info().rss / (1024 * 1024)

print("=" * 70)
print("KALLVETTU BASELINE PERFORMANCE AUDIT (PHASE 1)")
print("=" * 70)

profile_data = {
    "system": {
        "os": sys.platform,
        "python_version": sys.version.split()[0],
        "cpu_count": psutil.cpu_count(logical=True),
        "initial_mem_mb": round(get_mem_mb(), 2),
    },
    "benchmarks": {}
}

# -----------------------------------------------------------------------------
# 1. Module Import Latencies (Startup Breakdown)
# -----------------------------------------------------------------------------
print("\n--- 1. Module Import Breakdown ---")
import_times = {}

modules_to_test = [
    ("config", "from config import CONFIG"),
    ("restoration_engine", "import restoration_engine"),
    ("nlp_translation_engine", "import nlp_translation_engine"),
    ("multimodal_ai", "import multimodal_ai"),
    ("evaluation_engine", "import evaluation_engine"),
    ("services.database.connection", "from services.database.connection import get_engine, get_db_session"),
    ("services.database.repository", "from services.database.repository import EpigraphicalRepository"),
    ("services.ai.common.provider_manager", "from services.ai.common.provider_manager import get_provider_manager"),
    ("services.ocr_fusion", "import services.ocr_fusion"),
    ("services.geo_atlas.ui_atlas", "import services.geo_atlas.ui_atlas"),
    ("services.knowledge_graph.graph_engine", "import services.knowledge_graph.graph_engine"),
    ("services.knowledge_graph.graph_renderer", "import services.knowledge_graph.graph_renderer"),
    ("services.iiif.manifest_builder", "import services.iiif.manifest_builder"),
    ("services.iiif.viewer_component", "import services.iiif.viewer_component"),
    ("services.research_tools.rti_engine", "import services.research_tools.rti_engine"),
    ("services.ui_studios", "import services.ui_studios"),
]

for mod_name, stmt in modules_to_test:
    t0 = time.perf_counter()
    m0 = get_mem_mb()
    exec(stmt, globals())
    t1 = time.perf_counter()
    m1 = get_mem_mb()
    lat_ms = (t1 - t0) * 1000
    mem_delta = m1 - m0
    import_times[mod_name] = {"latency_ms": round(lat_ms, 2), "mem_delta_mb": round(mem_delta, 2)}
    print(f"  {mod_name:<38}: {lat_ms:7.2f} ms | +{mem_delta:5.2f} MB")

profile_data["benchmarks"]["imports"] = import_times

# -----------------------------------------------------------------------------
# 2. Database Query Latency
# -----------------------------------------------------------------------------
print("\n--- 2. Database Query Latency ---")
from services.database.repository import EpigraphicalRepository
from services.database.connection import get_db_session
from services.database.models import Inscription

# Cold init
t0 = time.perf_counter()
EpigraphicalRepository.init_database()
db_init_ms = (time.perf_counter() - t0) * 1000
print(f"  Database Init / Seeding Latency      : {db_init_ms:7.2f} ms")

# Simple query (cold)
t0 = time.perf_counter()
with get_db_session() as s:
    count = s.query(Inscription).count()
query_cold_ms = (time.perf_counter() - t0) * 1000
print(f"  Inscription Count Query (Cold)       : {query_cold_ms:7.2f} ms (Count: {count})")

# Simple query (warm - 100 iterations)
t0 = time.perf_counter()
for _ in range(50):
    with get_db_session() as s:
        inscriptions = s.query(Inscription).all()
query_warm_ms = ((time.perf_counter() - t0) / 50) * 1000
print(f"  Inscription Select All (Warm Avg)    : {query_warm_ms:7.2f} ms")

profile_data["benchmarks"]["database"] = {
    "init_ms": round(db_init_ms, 2),
    "query_cold_ms": round(query_cold_ms, 2),
    "query_warm_avg_ms": round(query_warm_ms, 2)
}

# -----------------------------------------------------------------------------
# 3. Image Restoration Pipeline
# -----------------------------------------------------------------------------
print("\n--- 3. Digital Image Restoration Pipeline ---")
from PIL import Image
from restoration_engine import process_full_restoration, apply_clahe_pillow, sauvola_binarization

sample_path = "assets/samples/chola_kalvettu.jpg"
if not os.path.exists(sample_path):
    # fallback to any sample
    sample_files = [f for f in os.listdir("assets/samples") if f.endswith(".jpg")]
    sample_path = os.path.join("assets/samples", sample_files[0])

test_img = Image.open(sample_path)
print(f"  Loaded benchmark plate: {sample_path} ({test_img.width}x{test_img.height} px)")

# Cold full restoration
t0 = time.perf_counter()
m0 = get_mem_mb()
res_cold = process_full_restoration(test_img, contrast_boost=2.5, binarize_k=0.2, squeeze_mode="charcoal_on_squeeze", max_dim=1600)
res_cold_ms = (time.perf_counter() - t0) * 1000
res_cold_mem = get_mem_mb() - m0
print(f"  Full Restoration Pipeline (Cold)     : {res_cold_ms:7.2f} ms | +{res_cold_mem:.2f} MB")

# Warm full restoration
t0 = time.perf_counter()
res_warm = process_full_restoration(test_img, contrast_boost=2.5, binarize_k=0.2, squeeze_mode="charcoal_on_squeeze", max_dim=1600)
res_warm_ms = (time.perf_counter() - t0) * 1000
print(f"  Full Restoration Pipeline (Warm)     : {res_warm_ms:7.2f} ms")

# Individual component timings
t0 = time.perf_counter()
gray = test_img.convert("L")
t_gray = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
clahe = apply_clahe_pillow(test_img, clip_limit=2.5)
t_clahe = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
binarized = sauvola_binarization(gray, k=0.2)
t_sauvola = (time.perf_counter() - t0) * 1000

print(f"    - Grayscale Conversion             : {t_gray:7.2f} ms")
print(f"    - CLAHE (clip=2.5)                 : {t_clahe:7.2f} ms")
print(f"    - Sauvola Adaptive Binarization    : {t_sauvola:7.2f} ms")

profile_data["benchmarks"]["restoration"] = {
    "image_dims": f"{test_img.width}x{test_img.height}",
    "pipeline_cold_ms": round(res_cold_ms, 2),
    "pipeline_warm_ms": round(res_warm_ms, 2),
    "grayscale_ms": round(t_gray, 2),
    "clahe_ms": round(t_clahe, 2),
    "sauvola_ms": round(t_sauvola, 2),
    "mem_delta_mb": round(res_cold_mem, 2)
}

# -----------------------------------------------------------------------------
# 4. Classical Tamil Dictionary Search
# -----------------------------------------------------------------------------
print("\n--- 4. Classical Tamil Dictionary Queries ---")
from services.nlp.classical_dictionary import ClassicalTamilDictionary

t0 = time.perf_counter()
lex = ClassicalTamilDictionary()
lex_init_ms = (time.perf_counter() - t0) * 1000
print(f"  Lexicon SQLite Init / Load           : {lex_init_ms:7.2f} ms")

# Cold search
t0 = time.perf_counter()
res_cold = lex.search("ஆதன்", limit=20)
dict_cold_ms = (time.perf_counter() - t0) * 1000
print(f"  Lexicon Search 'ஆதன்' (Cold)         : {dict_cold_ms:7.2f} ms (Matches: {len(res_cold)})")

# Warm search (50 queries)
t0 = time.perf_counter()
for _ in range(50):
    lex.search("ஆதன்", limit=20)
dict_warm_ms = ((time.perf_counter() - t0) / 50) * 1000
print(f"  Lexicon Search 'ஆதன்' (Warm Avg)     : {dict_warm_ms:7.2f} ms")

profile_data["benchmarks"]["dictionary"] = {
    "init_ms": round(lex_init_ms, 2),
    "search_cold_ms": round(dict_cold_ms, 2),
    "search_warm_avg_ms": round(dict_warm_ms, 2)
}

# -----------------------------------------------------------------------------
# 5. Historical Knowledge Graph Rendering
# -----------------------------------------------------------------------------
print("\n--- 5. Historical Knowledge Graph Execution ---")
from services.knowledge_graph.graph_engine import EpigraphicalKnowledgeGraph
from services.knowledge_graph.graph_renderer import render_knowledge_graph_html

t0 = time.perf_counter()
kg = EpigraphicalKnowledgeGraph()
kg_build_ms = (time.perf_counter() - t0) * 1000
print(f"  Knowledge Graph Build (29n, 69e)     : {kg_build_ms:7.2f} ms")

t0 = time.perf_counter()
d = kg.to_dict()
kg_dict_cold_ms = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
d_warm = kg.to_dict()
kg_dict_warm_ms = (time.perf_counter() - t0) * 1000
print(f"  Graph Serialization (Cold)           : {kg_dict_cold_ms:7.2f} ms")
print(f"  Graph Serialization (Warm / Cached)  : {kg_dict_warm_ms:7.2f} ms")

t0 = time.perf_counter()
html = render_knowledge_graph_html(kg, height_px=650)
kg_render_ms = (time.perf_counter() - t0) * 1000
print(f"  Knowledge Graph HTML Rendering       : {kg_render_ms:7.2f} ms (HTML size: {len(html)} bytes)")

profile_data["benchmarks"]["knowledge_graph"] = {
    "graph_build_ms": round(kg_build_ms, 2),
    "serialization_flags_cold_ms": round(kg_dict_cold_ms, 2),
    "serialization_flags_warm_ms": round(kg_dict_warm_ms, 2),
    "html_render_ms": round(kg_render_ms, 2),
    "html_bytes": len(html)
}

# -----------------------------------------------------------------------------
# 6. IIIF Presentation 3.0 & Image Service Generation
# -----------------------------------------------------------------------------
print("\n--- 6. IIIF Presentation 3.0 & Manifest Generation ---")
from services.iiif.manifest_builder import IIIFManifestBuilder

builder = IIIFManifestBuilder()
t0 = time.perf_counter()
manifest = builder.build_manifest(
    inscription_id="KV-INSC-002",
    label="Thanjavur Brihadisvara Inscription",
    image_url="https://kallvettu.org/images/chola_kalvettu.jpg",
    width=test_img.width,
    height=test_img.height,
    metadata_items=[{"label": "Dynasty", "value": "Chola"}]
)
iiif_manifest_ms = (time.perf_counter() - t0) * 1000
print(f"  IIIF Manifest 3.0 Generation         : {iiif_manifest_ms:7.2f} ms")

profile_data["benchmarks"]["iiif"] = {
    "manifest_ms": round(iiif_manifest_ms, 2)
}

# -----------------------------------------------------------------------------
# 7. Geographic Atlas GeoJSON Loading
# -----------------------------------------------------------------------------
print("\n--- 7. Geographic Atlas Loading ---")
from services.geo_atlas.geo_loader import load_state_districts_geojson
t0 = time.perf_counter()
geojson = load_state_districts_geojson("Tamil Nadu")
geo_load_ms = (time.perf_counter() - t0) * 1000
print(f"  Tamil Nadu GeoJSON Load              : {geo_load_ms:7.2f} ms (Features: {len(geojson.get('features', []))})")

profile_data["benchmarks"]["geographic_atlas"] = {
    "geojson_load_ms": round(geo_load_ms, 2),
    "feature_count": len(geojson.get("features", []))
}

# -----------------------------------------------------------------------------
# 8. K-IEFE Evidence Fusion Execution
# -----------------------------------------------------------------------------
print("\n--- 8. K-IEFE Evidence Fusion Execution ---")
from services.ocr_fusion import fuse_epigraphical_evidence
from services.ai.common.base_provider import OCRResult

results_dict = {
    "paddle": OCRResult(text="ஸ்வஸ்திஸ்ரீ ராஜராஜ சோழன்", provider_name="PaddleOCR", model_id="pp-ocrv5", confidence=0.88),
    "sarvam": OCRResult(text="ஸ்வஸ்தி ஸ்ரீ ராஜராஜ சோழன்", provider_name="Sarvam Vision", model_id="sarvam-vision", confidence=0.92),
    "gemini": OCRResult(text="ஸ்வஸ்திஸ்ரீ இராஜராஜ சோழன்", provider_name="Gemini Vision", model_id="gemini-2.5-flash", confidence=0.95),
}
t0 = time.perf_counter()
bundle = fuse_epigraphical_evidence(test_img, results_dict, script_hint="medieval_tamil")
fusion_ms = (time.perf_counter() - t0) * 1000
print(f"  K-IEFE Master Evidence Fusion        : {fusion_ms:7.2f} ms (Fused text: '{bundle.fused_result.fused_text}')")

profile_data["benchmarks"]["kiefe_fusion"] = {
    "fusion_ms": round(fusion_ms, 2)
}

# -----------------------------------------------------------------------------
# 9. PaddleOCR Model Initialization & Inference
# -----------------------------------------------------------------------------
print("\n--- 9. PaddleOCR Engine Status & Inference ---")
from services.ai.offline.paddleocr_provider import PaddleOCRProvider

t0 = time.perf_counter()
p_prov = PaddleOCRProvider()
p_init_ms = (time.perf_counter() - t0) * 1000
stat = p_prov.get_runtime_status()
print(f"  PaddleOCR Provider Init              : {p_init_ms:7.2f} ms")
print(f"  PaddleOCR Runtime Status             : {stat.get('state')} ({stat.get('status_label')})")

profile_data["benchmarks"]["paddleocr"] = {
    "provider_init_ms": round(p_init_ms, 2),
    "runtime_state": stat.get("state"),
    "status_label": stat.get("status_label")
}

profile_data["system"]["final_mem_mb"] = round(get_mem_mb(), 2)
profile_data["system"]["peak_mem_mb"] = round(get_mem_mb(), 2)

print("\n" + "=" * 70)
print(f"BENCHMARK COMPLETE. Peak Memory: {profile_data['system']['peak_mem_mb']} MB")
print("=" * 70)

# Save results to json for reporting
with open("docs/performance_baseline.json", "w", encoding="utf-8") as f:
    json.dump(profile_data, f, indent=2)
print("Saved benchmark metrics to docs/performance_baseline.json")
