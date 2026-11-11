"""
End-to-End Multi-Provider Pipeline & Scientific Validation Tests.
Verifies cross-provider coordination, OCR dispute consensus,
BHASHINI linguistic extraction, and Sarvam AI integration across authentic epigraphical plates.
"""

import os
from unittest.mock import MagicMock, patch
from PIL import Image
import numpy as np
import pytest

from config import CONFIG
from nlp_translation_engine import find_preset_by_id, analyze_custom_epigraph_text
from services.ai.common.provider_manager import get_provider_manager
from services.ai.common.comparison_engine import MultiEngineOCRComparator, ComparisonResult
from services.ai.common.base_provider import OCRResult, TranslationResult, InterpretationResult


def test_multi_provider_pipeline_preset_integration():
    """Validates that loading a curated preset feeds into Classical Tamil linguistic analysis."""
    preset = find_preset_by_id("mangulam_brahmi")
    assert preset is not None
    assert "நெடுஞ்செழியன்" in preset["restored_text"]

    analysis = analyze_custom_epigraph_text(preset["restored_text"])
    assert "words" in analysis
    words = [w["word"] for w in analysis["words"]]
    assert any("செழியன்" in w or "நெடுஞ்செழியன்" in w for w in words)


def test_multi_engine_ocr_comparator_on_sample_plate():
    """Runs MultiEngineOCRComparator on authentic sample plate image with mocked engines."""
    sample_path = os.path.join(CONFIG.SAMPLES_DIR, "mangulam.jpg")
    assert os.path.exists(sample_path)
    img = Image.open(sample_path)

    pm = get_provider_manager()

    # Create mock engines
    mock_gemini = MagicMock()
    mock_gemini.provider_name = "Google Gemini Vision AI"
    mock_gemini.recognize.return_value = OCRResult(
        text="கணிய நந்தஸிரிய் குவ அந்துவன்",
        provider_name="Google Gemini Vision AI",
        model_id="gemini-2.5-flash",
        confidence=0.92,
        latency_seconds=1.1,
        is_cloud=True
    )

    mock_paddle = MagicMock()
    mock_paddle.provider_name = "PaddleOCR Local Engine (PP-OCRv5 Tamil)"
    mock_paddle.recognize.return_value = OCRResult(
        text="கணிய நந்தஸிரிய் குவ அந்தவன்",  # deliberate 1-char difference
        provider_name="PaddleOCR Local Engine (PP-OCRv5 Tamil)",
        model_id="ta_PP-OCRv5_mobile_rec",
        confidence=0.88,
        latency_seconds=0.3,
        is_cloud=False
    )

    mock_sarvam = MagicMock()
    mock_sarvam.provider_name = "Sarvam Vision (Document AI)"
    mock_sarvam.recognize.return_value = OCRResult(
        text="கணிய நந்தஸிரிய் குவ அந்துவன்",
        provider_name="Sarvam Vision (Document AI)",
        model_id="sarvam-vision-3b",
        confidence=0.89,
        latency_seconds=0.7,
        is_cloud=True
    )

    comparator = MultiEngineOCRComparator(pm)

    with patch.object(pm, "get_configured_ocr_providers", return_value=[mock_gemini, mock_paddle, mock_sarvam]):
        comp = comparator.compare_all(img, script_hint="Tamil-Brahmi")

        assert isinstance(comp, ComparisonResult)
        assert len(comp.results_by_provider) == 3
        assert comp.fastest_provider == "PaddleOCR Local Engine (PP-OCRv5 Tamil)"
        assert comp.consensus_ratio >= 0.60

        # Disputed token should catch அந்துவன் vs அந்தவன்
        actually_disputed = [t for t in comp.disputed_tokens if t.is_disputed]
        assert len(actually_disputed) >= 1
        assert actually_disputed[0].is_disputed is True
        assert "அந்துவன்" in comp.recommended_text or "அந்தவன்" in comp.recommended_text


def test_sarvam_translation_and_mayura_integration():
    """Tests Sarvam translation flow on classic epigraph text."""
    pm = get_provider_manager()
    sarvam_prov = pm.get_provider("sarvam_translate")
    mayura_prov = pm.get_provider("sarvam_mayura")

    epigraph_text = "ஸ்வஸ்திஸ்ரீ கோப்பரகேசரிவன்மர்க்கு யாண்டு"

    mock_sdk = MagicMock()
    mock_sdk.text.translate.return_value = MagicMock(
        translated_text="Prosperity and Grace to King Parakesarivarman, in the year",
        request_id="req_test_123"
    )

    sarvam_prov.set_client_for_testing(mock_sdk)
    t_res = sarvam_prov.translate(epigraph_text, target_language_code="en-IN")

    assert isinstance(t_res, TranslationResult)
    assert "Parakesarivarman" in t_res.translated_text
    assert t_res.model_id == "sarvam-translate:v1"
    assert t_res.request_id == "req_test_123"

    # Test Mayura with roman script
    mock_sdk.text.translate.return_value = MagicMock(
        translated_text="svastisri kopparakesarivanmarkku yantu",
        request_id="req_mayura_123"
    )
    mayura_prov.set_client_for_testing(mock_sdk)
    m_res = mayura_prov.translate(epigraph_text, target_language_code="en-IN", output_script="roman")

    assert isinstance(m_res, TranslationResult)
    assert m_res.model_id == "mayura:v1"


def test_sarvam_chat_interpretation_integration():
    """Tests Sarvam Chat interpretation extraction."""
    pm = get_provider_manager()
    chat_prov = pm.get_provider("sarvam_chat_interpretation")

    mock_sdk = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = (
        "Historical Context: Medieval Chola epigraph under Rajaraja I.\n"
        "Key Findings:\n"
        "- Documents endowment to the Thanjavur Brihadisvara Temple.\n"
        "- Tax exemption granted on agricultural lands.\n"
        "Entities: இராஜராஜ, தஞ்சாவூர், பெருங்கோயில்."
    )
    mock_choice.message.reasoning_content = None
    mock_sdk.chat.completions.return_value = MagicMock(choices=[mock_choice])

    chat_prov.set_client_for_testing(mock_sdk)
    interp = chat_prov.interpret("ஸ்வஸ்திஸ்ரீ இராஜராஜ தேவர்க்கு யாண்டு...")

    assert isinstance(interp, InterpretationResult)
    assert "Historical Context" in interp.interpretation_text
    assert len(interp.key_takeaways) >= 2
    assert any("Brihadisvara" in tk or "endowment" in tk for tk in interp.key_takeaways)
    assert len(interp.entities_identified) >= 2



