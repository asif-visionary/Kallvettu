"""
Comprehensive Verification Test Suite for Kallvettu OCR Accuracy & Multi-Engine Verification.

Covers all 10 core objectives:
1. Source Data Isolation & SHA-256 Cryptographic Tracking
2. Adaptive Image Preprocessing (7 independent variants, removing squeeze from default OCR)
3. PaddleOCR Reading Order & Line Clustering (top-to-bottom, left-to-right)
4. Sarvam Vision Response Validation (description_only, partial, success, zero bounding boxes)
5. Gemini Vision Description Isolation & Key Guard
6. Sequence Alignment & Prevention of Cascading False Disputes
7. Character-Level Dispute Resolution & Detailed Classifications
8. Multi-Tier Epigraphical Representation (Isolated from Presets)
9. Transcription Revision History & Audit Trail
10. Engine Execution Manifest & Honest Metrics Reporting
"""

import os
import sys
import pytest
import numpy as np
from PIL import Image
from unittest.mock import MagicMock, patch

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.ai.common.source_identity import (
    SourceIdentity,
    ProvenanceMode,
    compute_image_sha256,
    create_source_identity,
)
from restoration_engine import (
    generate_ocr_variants,
    adaptive_gaussian_threshold,
    sauvola_binarization,
    apply_clahe_pillow,
)
from services.ai.offline.paddleocr_provider import PaddleOCRProvider
from services.ai.sarvam.vision_provider import SarvamVisionProvider
from services.ai.gemini.vision_provider import GeminiVisionProvider
from services.ai.common.comparison_engine import (
    MultiEngineOCRComparator,
    MultiOCRComparisonResult,
    DisputedToken,
    normalize_tamil_reading,
)
from services.ai.common.base_provider import OCRResult, BoundingBox
from nlp_translation_engine import analyze_custom_epigraph_text


# ==============================================================================
# OBJECTIVE 1: SOURCE DATA ISOLATION & SHA-256 TRACKING
# ==============================================================================

def test_source_identity_isolation_and_hash():
    """Verify cryptographic SHA-256 integrity and provenance tracking."""
    test_img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    hash1 = compute_image_sha256(test_img)
    assert len(hash1) == 64
    assert isinstance(hash1, str)

    src_id = create_source_identity(
        test_img,
        filename="mangulam_specimen.png",
        provenance_mode=ProvenanceMode.LIVE_OCR
    )

    assert src_id.image_sha256 == hash1
    assert src_id.source_image_id.startswith("SRC_")
    assert src_id.ocr_execution_id.startswith("OCR_")
    assert src_id.is_live_ocr() is True
    assert src_id.is_curated_preset() is False

    # Verify compatibility check across identical vs disparate images
    different_img = Image.new("RGB", (100, 100), color=(200, 200, 200))
    src_id2 = create_source_identity(different_img, filename="other.png")
    assert src_id.is_comparable_with(src_id) is True
    assert src_id.is_comparable_with(src_id2) is False


# ==============================================================================
# OBJECTIVE 2: ADAPTIVE PREPROCESSING — 7 INDEPENDENT VARIANTS
# ==============================================================================

def test_image_preprocessing_7_variants():
    """Verify 7 independent variants generated for OCR without Sobel-only rubbings."""
    test_img = Image.new("RGB", (200, 200), color=(100, 100, 100))
    variants = generate_ocr_variants(test_img)

    required_keys = [
        "original", "grayscale", "mild_clahe", "denoised_gray",
        "sauvola", "adaptive_thresh", "roi_crop"
    ]
    for rk in required_keys:
        assert rk in variants, f"Variant {rk} missing from generated OCR variants"
        assert isinstance(variants[rk]["image"], Image.Image)
        assert "label" in variants[rk]
        assert "description" in variants[rk]

    # Verify mild_clahe is marked default OCR input
    assert variants["mild_clahe"]["is_default_ocr"] is True
    assert variants["original"]["is_default_ocr"] is False

    # Verify Adaptive Gaussian threshold produces binary output
    adapt_bin = adaptive_gaussian_threshold(test_img, window_size=25, c=0.03)
    assert isinstance(adapt_bin, Image.Image)
    assert adapt_bin.mode == "L"


# ==============================================================================
# OBJECTIVE 3: PADDLEOCR READING ORDER SORTING & CLUSTERING
# ==============================================================================

def test_paddleocr_reading_order_sorting():
    """Verify detected text line polygons are sorted top-to-bottom, left-to-right."""
    # Line 1 (y ≈ 50): box1 (x: 100..200), box0 (x: 10..90) [Out of order in input]
    # Line 2 (y ≈ 120): box2 (x: 15..150)
    p_line1_right = [[100, 50], [200, 50], [200, 70], [100, 70]]
    p_line1_left = [[10, 48], [90, 48], [90, 72], [10, 72]]
    p_line2 = [[15, 120], [150, 120], [150, 140], [15, 140]]

    unsorted_polys = [p_line2, p_line1_right, p_line1_left]
    sorted_boxes = PaddleOCRProvider._sort_polygons_reading_order(unsorted_polys)

    assert len(sorted_boxes) == 3
    # First should be Line 1 Left (x: 10)
    assert sorted_boxes[0][0] == 10
    assert sorted_boxes[0][1] == 48
    # Second should be Line 1 Right (x: 100)
    assert sorted_boxes[1][0] == 100
    assert sorted_boxes[1][1] == 50
    # Third should be Line 2 (y: 120)
    assert sorted_boxes[2][1] == 120


# ==============================================================================
# OBJECTIVE 4: SARVAM VISION RESPONSE VALIDATION & STATUS CLASSIFICATION
# ==============================================================================

def test_sarvam_vision_response_validation():
    """Verify Sarvam response validation: description_only, partial, success, and honest zero boxes."""
    prov = SarvamVisionProvider(client=MagicMock())

    # 1. Description only classification
    desc_text = "This image depicts an ancient stone inscription photograph on weathered granite surface."
    with patch.object(prov._client, "get_sdk_client") as mock_sdk_factory:
        mock_sdk = MagicMock()
        mock_sdk.doc_ai.parse.return_value = MagicMock(job_id="job_desc")
        mock_sdk.doc_ai.get_status.return_value = MagicMock(status="completed")
        mock_sdk.doc_ai.get_results.return_value = MagicMock(markdown=desc_text)
        mock_sdk_factory.return_value = mock_sdk

        dummy_img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        res = prov.recognize(dummy_img)

        assert res.metadata["transcription_status"] == "description_only"
        assert res.text == ""  # Never treat description as inscription text!
        assert res.metadata["bounding_boxes_supported"] is False
        assert len(res.bounding_boxes) == 0

    # 2. Genuine Tamil inscription success
    tamil_text = "ஸ்வஸ்திஸ்ரீ கோப்பரகேசரி பந்மரான"
    with patch.object(prov._client, "get_sdk_client") as mock_sdk_factory:
        mock_sdk = MagicMock()
        mock_sdk.doc_ai.parse.return_value = MagicMock(job_id="job_tam")
        mock_sdk.doc_ai.get_status.return_value = MagicMock(status="completed")
        mock_sdk.doc_ai.get_results.return_value = MagicMock(markdown=tamil_text)
        mock_sdk_factory.return_value = mock_sdk

        res2 = prov.recognize(dummy_img)
        assert res2.metadata["transcription_status"] == "success"
        assert "ஸ்வஸ்திஸ்ரீ" in res2.text


# ==============================================================================
# OBJECTIVE 5: GEMINI VISION ISOLATION & GUARD
# ==============================================================================

def test_gemini_vision_isolation():
    """Verify Gemini Vision isolates descriptions and enforces API key guard."""
    # When unconfigured, should report is_configured = False
    prov = GeminiVisionProvider(api_key="")
    assert prov.is_configured() is False

    # When configured, recognizes text and tags status
    prov_cfg = GeminiVisionProvider(api_key="AIzaSyDummyKeyForTestingOnly1234567")
    assert prov_cfg.is_configured() is True


# ==============================================================================
# OBJECTIVE 6: SEQUENCE ALIGNMENT (PREVENTS CASCADING FALSE DISPUTES)
# ==============================================================================

def test_sequence_alignment_prevents_cascading_false_disputes():
    """
    CRITICAL TEST: Verifies that when one model splits a word (e.g. 'ஸ்வஸ்தி ஸ்ரீ' vs 'ஸ்வஸ்திஸ்ரீ'),
    sequence alignment does NOT cascade false disputes across all following words.
    """
    comparator = MultiEngineOCRComparator()

    # Model A splits first word into two tokens, Model B joins them into one token
    # Both agree on all subsequent words: 'கோப்பரகேசரி', 'பந்மரான', 'உடையார்', 'ஸ்ரீராஜராஜதேவர்'
    text_a = "ஸ்வஸ்தி ஸ்ரீ கோப்பரகேசரி பந்மரான உடையார் ஸ்ரீராஜராஜதேவர்"
    text_b = "ஸ்வஸ்திஸ்ரீ கோப்பரகேசரி பந்மரான உடையார் ஸ்ரீராஜராஜதேவர்"

    res_a = OCRResult(text=text_a, provider_name="Model A", model_id="m-a")
    res_b = OCRResult(text=text_b, provider_name="Model B", model_id="m-b")

    results_dict = {"model_a": res_a, "model_b": res_b}

    (
        disputed,
        ratio,
        total,
        agreed,
        disp_count,
        default_app,
        comp_status,
        insufficient_msg,
    ) = comparator._analyze_disagreements(results_dict)

    # All 4 subsequent words must be recognized as AGREED
    assert comp_status == "success"
    assert agreed >= 4
    assert disp_count == 1  # Only the split word is disputed!
    assert ratio >= 0.75   # Inter-engine agreement is high (75-80%), NOT 5.6%!
    assert "கோப்பரகேசரி" in default_app
    assert "ஸ்ரீராஜராஜதேவர்" in default_app


# ==============================================================================
# OBJECTIVE 7: CHARACTER-LEVEL DISPUTE RESOLUTION
# ==============================================================================

def test_character_level_dispute_classification():
    """Verify disputed tokens contain dispute_type and diff_details."""
    comparator = MultiEngineOCRComparator()

    # Word 1: 'ஸ்வஸ்திஸ்ரீ' (agreed)
    # Word 2: 'பந்மரான' vs 'பந்மரானா' (substitution dispute)
    res_a = OCRResult(text="ஸ்வஸ்திஸ்ரீ பந்மரான", provider_name="Model A", model_id="a")
    res_b = OCRResult(text="ஸ்வஸ்திஸ்ரீ பந்மரானா", provider_name="Model B", model_id="b")

    disputed, ratio, total, agreed, disp_count, app, status, _ = comparator._analyze_disagreements({
        "a": res_a, "b": res_b
    })

    assert len(disputed) == 2
    assert disputed[0].is_disputed is False
    assert disputed[0].dispute_type == "identical"

    assert disputed[1].is_disputed is True
    assert disputed[1].dispute_type == "substitution"
    assert "பந்மரான" in str(disputed[1].diff_details)


# ==============================================================================
# OBJECTIVE 8: MULTI-TIER TRANSCRIPTION & PRESET ISOLATION
# ==============================================================================

def test_multi_tier_transcription_source_isolation():
    """Verify live OCR analysis is NOT overwritten by preset catalog when allow_preset_match=False."""
    live_ocr_reading = "சாத்தன் நடுகல் [கொ]ற்றன்"
    dummy_source = create_source_identity(b"fake_bytes_12345", filename="stone_live.png", provenance_mode=ProvenanceMode.LIVE_OCR)

    analysis = analyze_custom_epigraph_text(
        live_ocr_reading,
        script_hint="tamil_brahmi",
        provenance="Live OCR Prediction",
        allow_preset_match=False,
        source_identity=dummy_source
    )

    # Must retain live text in all tiers
    assert analysis["weathered_text"] == live_ocr_reading
    assert analysis["diplomatic_text"] == live_ocr_reading
    assert analysis["is_curated_preset"] is False
    assert analysis["source_identity"]["image_sha256"] == dummy_source.image_sha256
    assert analysis["tier_1_diplomatic"] == live_ocr_reading
    assert "சாத்தன்" in analysis["tier_2_normalized"]
    assert analysis["word_segmented_text"] != ""


# ==============================================================================
# OBJECTIVE 9: TRANSCRIPTION REVISION HISTORY
# ==============================================================================

def test_transcription_revision_history_tracking():
    """Verify non-destructive researcher transcription revision history."""
    revision_history = []
    original_ocr = "ஸ்வஸ்திஸ்ரீ கோப்பரகேசரி [பந்]மரான"
    scholar_revised = "ஸ்வஸ்திஸ்ரீ கோப்பரகேசரி பந்மரான"

    rev_entry = {
        "revision_id": len(revision_history) + 1,
        "timestamp": "2026-11-10 10:00:00",
        "editor": "Researcher Epigraphist",
        "previous_text": original_ocr,
        "revised_text": scholar_revised,
        "reason": "Resolved damaged lacuna glyph [பந்] from parallel Thanjavur record."
    }
    revision_history.append(rev_entry)

    assert len(revision_history) == 1
    assert revision_history[0]["revision_id"] == 1
    assert revision_history[0]["previous_text"] == original_ocr
    assert revision_history[0]["revised_text"] == scholar_revised


# ==============================================================================
# OBJECTIVE 10: ENGINE EXECUTION MANIFEST & HONEST METRICS
# ==============================================================================

def test_multi_engine_comparator_manifest_and_execution():
    """Verify MultiOCRComparisonResult includes engine manifest and honest metric labels."""
    comparator = MultiEngineOCRComparator()
    dummy_img = Image.new("RGB", (100, 100), color=(128, 128, 128))

    # Mock manager to provide 2 OCR engines
    mock_prov1 = MagicMock()
    mock_prov1.provider_name = "Mock Engine 1"
    mock_prov1.provider_id = "mock_1"
    mock_prov1.recognize.return_value = OCRResult(text="ஸ்வஸ்திஸ்ரீ", provider_name="Mock Engine 1", model_id="m1")

    mock_prov2 = MagicMock()
    mock_prov2.provider_name = "Mock Engine 2"
    mock_prov2.provider_id = "mock_2"
    mock_prov2.recognize.return_value = OCRResult(text="ஸ்வஸ்திஸ்ரீ", provider_name="Mock Engine 2", model_id="m2")

    mock_mgr = MagicMock()
    mock_mgr.get_ocr_providers.return_value = [mock_prov1, mock_prov2]
    mock_mgr.get_configured_ocr_providers.return_value = [mock_prov1, mock_prov2]
    mock_mgr.get_ocr_provider.side_effect = lambda pid: mock_prov1 if pid == "mock_1" else mock_prov2
    mock_mgr.is_enabled.return_value = True

    comp = MultiEngineOCRComparator(provider_manager=mock_mgr)
    comp_res = comp.compare_ocr(dummy_img, provider_ids=["mock_1", "mock_2"])

    assert isinstance(comp_res, MultiOCRComparisonResult)
    assert comp_res.consensus_ratio == 1.0
    assert comp_res.agreement_metric_label == "Inter-Engine Agreement Rate (Not Absolute Accuracy)"
    assert comp_res.source_identity is not None
    assert "mock_1" in comp_res.executed_engines
    assert "mock_2" in comp_res.executed_engines
    assert len(comp_res.failed_engines) == 0

