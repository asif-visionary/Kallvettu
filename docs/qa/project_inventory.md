# Kallvettu Digital Epigraphy Platform — Project Inventory

**Date:** 2026-11-19  
**Version:** Kallvettu v3.1+ (Research Infrastructure Edition)  
**Author:** QA Automation & Epigraphical Systems Architecture Team  

---

## 1. System Overview & Technology Stack

| Layer | Technologies | Key Modules |
| :--- | :--- | :--- |
| **Presentation / UI** | Streamlit 1.67.0, OpenSeadragon 4.1.1, MapLibre GL, Vis.js Network | `app.py`, `ui_components.py`, `services/ui_studios.py` |
| **Computer Vision** | OpenCV (`opencv-contrib-python`), NumPy, SciPy, Pillow | `restoration_engine.py`, `services/research_tools/rti_engine.py` |
| **OCR & AI Inference** | Google Gemini Vision AI, Sarvam AI (Translate, Mayura, 105B, Vision), PaddleOCR (PP-OCRv5 Tamil) | `services/ai/`, `services/ocr_fusion/`, `multimodal_ai.py` |
| **Evidence Fusion (K-IEFE)** | Information-Theoretic Shannon Entropy, Grapheme Alignment, Stroke Visual Verification | `services/ocr_fusion/` |
| **Classical Tamil NLP** | Unicode NFC normalization, Sandhi splitting, Morphological Parsing, Lexicon lookup | `nlp_translation_engine.py`, `services/nlp/` |
| **Data Persistence** | SQLAlchemy 2.0 ORM, SQLite FTS5 (`datasets/kallvettu_research.db`), PostgreSQL compatibility | `services/database/` |
| **Digital Epigraphy Standards**| EpiDoc TEI P5 XML, IIIF Presentation 3.0 & Image 3.0 API, GraphML | `services/epidoc/`, `services/iiif/`, `services/knowledge_graph/` |
| **Public Research API** | FastAPI, Uvicorn, Pydantic V2 | `services/api/` |

---

## 2. Navigation Studios & Feature Matrix

The platform is organized into **16 specialized research workspaces (studios)** in the sidebar navigation:

| # | Studio Name | Entry Point & UI Implementation | Supporting Backend Services | Controls & User Actions |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **📊 Dashboard** | `app.py` (Lines 879–940) | `services/database/`, `config.py` | Session metric cards, active inscription badge, 1-click preset loader, research activity log. |
| **2** | **🏛️ Inscription Studio** | `app.py` (Lines 942–1025) | `restoration_engine.py`, `sample_generator.py` | File upload (drag & drop), camera capture, curated preset selector, full pipeline execution button. |
| **3** | **🔬 Digital Restoration Studio** | `app.py` (Lines 1025–1440) | `restoration_engine.py` | Before/After split viewer, CLAHE clip & grid sliders, Sauvola/Niblack/Otsu binarization, Squeeze modes, Sobel groove extraction, ROI zoom, difference map, surface roughness Z-score, processed image PNG download. |
| **4** | **✍️ Transcription Studio** | `app.py` (Lines 1442–1845) | `services/ocr_fusion/ui_renderer.py`, `services/ocr_fusion/` | K-IEFE 9-tab cross-verification panel, provider selection (PaddleOCR, Sarvam, Gemini), grapheme consensus table, Shannon entropy analysis, disputed-zone visual heatmap, scholar adjudication actions (`ACCEPT`, `ADOPT_PROVIDER`, `EDIT_SCHOLAR`, `MARK_ILLEGIBLE`, `DEFER`), cryptographic SHA-256 seal. |
| **5** | **📜 Translation Studio** | `app.py` (Lines 1847–1940) | `nlp_translation_engine.py`, `services/ai/sarvam/` | Contemporary Tamil translation, scholarly English translation, cultural/historical commentary, Sarvam Translate & Mayura colloquial mode switching. |
| **6** | **📖 Lexical Analysis** | `app.py` (Lines 1942–2050) | `nlp_translation_engine.py`, `services/nlp/classical_dictionary.py` | Searchable/sortable vocabulary table, root word identification, grammatical category, case marker extraction, etymological notes. |
| **7** | **📑 Research Dossier** | `app.py` (Lines 2052–2195) | `services/epidoc/exporter.py`, `services/epidoc/bundle.py` | Multi-tier diplomatic vs normalized transcription, historical period, royal attribution, EpiDoc TEI XML download, structured JSON export, scholarly Markdown export, 5-file research archive ZIP bundle. |
| **8** | **🏷️ Glyph Annotation Studio** | `app.py` (Lines 2197–2245) | `services/research/glyph_annotation_ui.py`, `services/research/historical_glyph_dataset.py` | Inscription selection, interactive image canvas, bounding box & polygon creation, character labeling, diplomatic/normalized readings, damage region marking, JSON/CSV annotation export, leakage-safe dataset splitting. |
| **9** | **🗺️ India Epigraphical Geographic Atlas** | `app.py` (Lines 2247–2283) | `services/geo_atlas/ui_atlas.py`, `services/geo_atlas/map_builder.py`, `services/geo_atlas/geo_loader.py` | Full-screen MapLibre interactive map, 38 Tamil Nadu district boundaries, 24+ iconic archaeological site markers, district drill-down, spatial coordinate queries. |
| **10** | **📖 Classical Tamil Dictionary** | `app.py` (Lines 2260–2283) | `services/nlp/dictionary_ui.py`, `services/nlp/classical_dictionary.py` | Tamil/English/ISO-15919 transliteration search, SQLite FTS5 full-text indexing, *Nannūl* Sandhi breakdown, 5-section epigraphical cards, morphological inspector. |
| **11** | **🔍 Epigraphical Search Engine** | `app.py` (Lines 2285–2294) | `services/ui_studios.py`, `services/search/corpus_search.py` | Multi-field corpus search (ID, Tamil text, translation, dynasty, ruler, site), faceted aggregations, highlighted `<mark>` snippets, search detail pages. |
| **12** | **🕸️ Historical Knowledge Graph** | `app.py` (Lines 2296–2307) | `services/ui_studios.py`, `services/knowledge_graph/graph_engine.py`, `services/knowledge_graph/graph_renderer.py` | Dual-engine visualizer (Vis.js Network + HTML5 Canvas Fallback), 29 entities, 69 verified relations, 4 royal dynasty flag image nodes (Pallava, Chera, Chola, Pandya) with contain-fit preservation, entity search, category filter, Fit View, physics toggle, entity inspection dossier panel, JSON/GraphML download. |
| **13** | **🔎 IIIF Deep-Zoom Inscription Viewer** | `app.py` (Lines 2309–2321) | `services/ui_studios.py`, `services/iiif/viewer_component.py`, `services/iiif/manifest_builder.py` | OpenSeadragon 4.1.1 viewer, IIIF Presentation 3.0 manifest generator, Image API 3.0 `info.json`, master image retrieval, sub-millimeter deep-zoom, rotation (90° CW/CCW), fit-to-home, synchronized side-by-side comparison, persistent character bounding box overlays, shareable spatial regions. |
| **14** | **💡 Photometric RTI & Relighting** | `app.py` (Lines 2323–2335) | `services/ui_studios.py`, `services/research_tools/rti_engine.py` | Multi-image directional lighting ingestion, virtual light azimuth and elevation controllers, surface normal estimation, diffuse gain, specular enhancement, raking light simulation. |
| **15** | **🔬 Research Evaluation Studio** | `app.py` (Lines 2337–2385) | `services/evaluation/benchmark_ui.py`, `services/evaluation/benchmark_engine.py` | Scientific benchmark dashboard: CER, WER, Top-1/3/5 accuracy, Expected Calibration Error (ECE), Brier score, confusion matrix, PSNR, SSIM, ablation experiment metadata, Markdown/JSON export. |
| **16** | **⚙️ Settings** | `app.py` (Lines 2387–2790) | `config.py`, `services/database/connection.py` | Multimodal AI credentials (Gemini, Sarvam), vision model selector, PaddleOCR device status (CPU/GPU), database connection diagnostics, storage paths, security controls. |

---

## 3. Database Schema & Persistence Inventory

The platform integrates **SQLAlchemy 2.0 ORM** with **SQLite FTS5** (`datasets/kallvettu_research.db`) and PostgreSQL support:

### Core Relational Models (`services/database/models.py`)
1. **`Inscription`**: Core charter record (ID, titles, script, historical period, site, district, material, dynasty, ruler, regnal year, description).
2. **`ImageAsset`**: Multimodal assets (master photos, CLAHE, squeeze rubbings, RTI source stacks, resolution, hashes).
3. **`Transcription`**: Epigraphical texts stratified across 5 tiers (Diplomatic, Normalized, Segmented, etc.).
4. **`ReadingCandidate`**: Multi-engine OCR readings per engine with grapheme alignment and confidence scores.
5. **`GlyphAnnotation`**: Character-level spatial annotations (normalized bounding boxes, polygons, legibility status).
6. **`Translation`**: Bilingual translations (Modern Tamil, Scholarly English) with commentary.
7. **`HistoricalReference`**: Linked monarchs, administrative bodies, and historical events.
8. **`Citation`**: Academic bibliography (South Indian Inscriptions, Epigraphia Indica, ARE).
9. **`Researcher`**: Scholar accounts with RBAC roles (`RESEARCHER`, `REVIEWER`, `ADMINISTRATOR`).
10. **`Review`**: Scholar adjudication records (`ACCEPT`, `ADOPT_PROVIDER`, `EDIT_SCHOLAR`, `MARK_ILLEGIBLE`, `DEFER`).
11. **`Revision`**: Immutable audit logs of text changes with rationales and diffs.
12. **`AuditEvent`**: Cryptographic system events with SHA-256 signatures.
13. **`ModelRun`**: Execution metadata for OCR and computer vision pipelines.
14. **`DatasetVersion`**: Versioned training dataset splits.
15. **`KnowledgeEntity`**: Entities mapped into the Epigraphical Knowledge Graph.
16. **`KnowledgeRelation`**: Directional verified historical relationships.

---

## 4. AI Models & Hardware Acceleration

| Provider / Model | Execution Target | Model Weights Location | Purpose |
| :--- | :--- | :--- | :--- |
| **PaddleOCR (`PP-OCRv5 Tamil`)** | Local CPU | `.paddlex_cache/official_models/` | Offline Tamil text-line detection (`PP-OCRv6_medium_det`) & recognition (`ta_PP-OCRv5_mobile_rec`). |
| **Google Gemini Vision AI** | Cloud API (`gemini-2.5-flash`, `gemini-1.5-pro`) | External Google Cloud | Zero-shot visual decipherment of ancient unpointed Tamil-Brahmi and Vatteluttu. |
| **Sarvam Vision AI** | Cloud API (`doc_ai.parse`) | External Sarvam AI Cloud | Lithic document parsing & structured text extraction. |
| **Sarvam Translate AI** | Cloud API (`sarvam-translate:v1`, `mayura:v1`) | External Sarvam AI Cloud | Indic formal and colloquial translations across 10 Indian languages. |
| **Sarvam-105B Scholar** | Cloud API (`sarvam-105b`) | External Sarvam AI Cloud | Deep Classical Tamil epigraphical reasoning and historical commentary. |
| **Classical NLP Heuristics** | Local CPU | Built-in Python rules | Nannūl Sandhi splitting, morphological case decomposition, rule-guided lacuna inpainting. |

---

## 5. Automated Test Suite Inventory

The project maintains 28 test suites in `tests/`:

1. `test_adaptive_routing_cpu.py`: CPU-governed multi-tier inference routing.
2. `test_classical_dictionary.py`: SQLite FTS5 search, ISO 15919 transliteration, Sandhi decomposition.
3. `test_collaboration.py`: RBAC, reviewer review workflows, audit logging.
4. `test_database_persistence.py`: Relational CRUD operations, transactions, foreign key enforcement.
5. `test_dispute_heatmap.py`: Spatial OCR dispute heatmap, transparent overlay generation.
6. `test_dossier_reliability.py`: Provenance separation, anti-hallucination classification, lacuna reporting.
7. `test_epidoc_export.py`: TEI P5 XML validation, schema compliance, archive ZIP bundling.
8. `test_evidence_integrity_monitor.py`: Shannon entropy calculation, dispersion vs veracity validation.
9. `test_geo_atlas.py`: MapLibre configuration, GeoJSON boundaries, site coordinates.
10. `test_geo_loader_audit.py`: GeoJSON file validation and caching.
11. `test_historical_glyph_dataset.py`: Glyph annotation schemas, leakage-safe dataset splitting.
12. `test_historical_ocr_training.py`: Historical character classes and training harness.
13. `test_historical_script_research.py`: Script classification inventories and features.
14. `test_hog_svm_historical.py`: HOG feature extraction and SVM classification.
15. `test_iiif_deep_zoom.py`: IIIF Presentation 3.0 manifest, Image 3.0 info.json, OpenSeadragon viewer component.
16. `test_inscription_search.py`: Corpus search, faceted filtering, context snippet generation.
17. `test_kallvettu.py`: Core pipeline, image restoration, CLAHE, Squeeze rubbings, binarization.
18. `test_knowledge_graph.py`: NetworkX graph engine, 29 entities, 69 edges, dynasty flags, Vis.js/Canvas renderer.
19. `test_line_recognizer.py`: CRNN+CTC text line recognition.
20. `test_maplibre_component.py`: MapLibre HTML component generation and events.
21. `test_multi_ocr_comparison.py`: Token alignment, consensus ratio calculation.
22. `test_multi_provider_pipeline.py`: Provider registration and orchestration.
23. `test_multi_spectral_restoration.py`: 4 spectral processing paths, difference maps, roughness maps.
24. `test_ocr_fusion.py`: K-IEFE evidence fusion, grapheme alignment, consensus entropy.
25. `test_ocr_verification_accuracy.py`: Visual stroke verifier, geometric verification.
26. `test_paddleocr_provider.py`: Local PaddleOCR execution and lifecycle states.
27. `test_provider_foundation.py`: BaseProvider, SourceIdentity, ModelResponse contracts.
28. `test_public_api.py`: FastAPI REST API endpoints at `/api/v1`.
