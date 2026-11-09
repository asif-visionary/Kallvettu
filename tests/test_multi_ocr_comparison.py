"""
Phase 7 Unit Tests: Multi-Engine Epigraphical Verification & Comparison System.
Covers:
- PaddleOCRProvider local execution, bounding boxes, and offline fallback
- MultiEngineOCRComparator 4-engine parallel aggregation (PaddleOCR, BHASHINI, Sarvam Vision, Gemini)
- Disputed token identification and consensus calculation
- Researcher approval workflow without blind majority vote
"""

import os
import sys
import pytest
import numpy as np
from PIL import Image
from unittest.mock import MagicMock

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.ai.offline.paddleocr_provider import PaddleOCRProvider
from services.ai.common.base_provider import OCRResult, BoundingBox
from services.ai.common.comparison_engine import (
    MultiEngineOCRComparator,
    MultiOCRComparisonResult,
    DisputedToken,
    get_ocr_comparator,
)
from services.ai.common.provider_manager import ProviderManager, get_provider_manager


# ==============================================================================
# 1. PADDLEOCR PROVIDER TESTS
# ==============================================================================

def test_paddleocr_provider_metadata():
    """Verify PaddleOCR offline provider metadata."""
    prov = PaddleOCRProvider()
    assert prov.provider_name == "PaddleOCR Local Engine (PP-OCRv5 Tamil)"
    assert prov.provider_id == "paddleocr_local"


def test_paddleocr_mocked_execution_with_boxes():
    """Verify PaddleOCR extracts lines, confidences, and bounding boxes."""
    prov = PaddleOCRProvider()

    # Mock internal PaddleOCR engine predict
    mock_engine = MagicMock()
    mock_engine.predict.return_value = [
        {"rec_text": "ஸ்வஸ்திஸ்ரீ", "rec_score": 0.95},
        {"rec_text": "கோப்பரகேசரி", "rec_score": 0.92},
    ]

    prov.set_mock_engine(mock_engine)

    img_arr = np.zeros((100, 200, 3), dtype=np.uint8)
    res = prov.recognize(img_arr)

    assert isinstance(res, OCRResult)
    assert "ஸ்வஸ்திஸ்ரீ" in res.text
    assert "கோப்பரகேசரி" in res.text
    assert res.provider_name == "PaddleOCR Local Engine (PP-OCRv5 Tamil)"
    assert res.is_cloud is False
    assert res.confidence is not None
    assert len(res.bounding_boxes) == 2
    assert res.bounding_boxes[0].text == "ஸ்வஸ்திஸ்ரீ"
    assert "script_limitation_note" in res.metadata


# ==============================================================================
# 2. MULTI-ENGINE OCR COMPARISON TESTS
# ==============================================================================

def test_multi_engine_ocr_dispute_and_consensus():
    """Verify comparison engine correctly classifies agreed vs disputed words."""
    comparator = MultiEngineOCRComparator()

    # Create 3 synthetic provider outputs with deliberate consensus and dispute
    # Word 1: 'ஸ்வஸ்திஸ்ரீ' (agreed by all)
    # Word 2: 'கோப்பரகேசரி' (agreed by all)
    # Word 3: 'பந்மரான' (Prov A) vs 'பந்மரானா' (Prov B) vs 'பந்மர்' (Prov C) -> DISPUTED
    res_a = OCRResult(
        text="ஸ்வஸ்திஸ்ரீ கோப்பரகேசரி பந்மரான",
        provider_name="Provider A",
        model_id="model-a"
    )
    res_b = OCRResult(
        text="ஸ்வஸ்திஸ்ரீ கோப்பரகேசரி பந்மரானா",
        provider_name="Provider B",
        model_id="model-b"
    )
    res_c = OCRResult(
        text="ஸ்வஸ்திஸ்ரீ கோப்பரகேசரி பந்மர்",
        provider_name="Provider C",
        model_id="model-c"
    )

    results_dict = {
        "prov_a": res_a,
        "prov_b": res_b,
        "prov_c": res_c,
    }

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

    assert total == 3
    assert agreed == 2
    assert disp_count == 1
    assert round(ratio, 2) == 0.67
    assert comp_status == "success"

    # Word 1 and 2 not disputed
    assert disputed[0].is_disputed is False
    assert disputed[0].consensus_candidate == "ஸ்வஸ்திஸ்ரீ"
    assert disputed[1].is_disputed is False
    assert disputed[1].consensus_candidate == "கோப்பரகேசரி"

    # Word 3 is disputed
    assert disputed[2].is_disputed is True
    assert len(disputed[2].readings) == 3


def test_multi_engine_insufficient_output_no_false_disputes():
    """Verify that when only 1 engine yields text, no false disputes are calculated."""
    comparator = MultiEngineOCRComparator()
    results_dict = {
        "prov_a": OCRResult(text="சாத்தன் நடுகல்", provider_name="A", model_id="m1"),
        "prov_b": OCRResult(text="", provider_name="B", model_id="m2"),
    }

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

    assert comp_status == "insufficient_candidates"
    assert insufficient_msg == "Insufficient OCR output for palaeographical comparison."
    assert len(disputed) == 0
    assert disp_count == 0
    assert default_app == "சாத்தன் நடுகல்"


def test_comparison_result_dataclass():
    """Verify MultiOCRComparisonResult serialization."""
    res_a = OCRResult(text="சாத்தன் நடுகல்", provider_name="A", model_id="m1")
    comp = MultiOCRComparisonResult(
        provider_results={"a": res_a},
        disputed_tokens=[],
        consensus_ratio=1.0,
        total_tokens_evaluated=2,
        agreed_tokens_count=2,
        disputed_tokens_count=0,
        approved_transcription="சாத்தன் நடுகல்",
        is_researcher_approved=True,
        execution_time_seconds=0.45
    )

    d = comp.to_dict()
    assert d["consensus_ratio"] == 1.0
    assert d["is_researcher_approved"] is True
    assert d["approved_transcription"] == "சாத்தன் நடுகல்"


def test_provider_manager_paddleocr_registration():
    """Verify ProviderManager auto-registers PaddleOCR."""
    mgr = get_provider_manager()
    prov = mgr.get_ocr_provider("paddleocr_local")
    assert prov is not None
    assert isinstance(prov, PaddleOCRProvider)

