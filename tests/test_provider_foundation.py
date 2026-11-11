"""
Phase 1 Unit Tests: Unified AI Provider Architecture & Foundation.
Covers:
- Typed data models (OCRResult, TranslationResult, InterpretationResult, BoundingBox)
- Error handling & credential leakage protection (sanitize_error_string, format_provider_error)
- Response validation (Tamil Unicode detection, character ratio, bounding box normalization)
- ProviderManager registration, enablement toggles, status reporting
- GeminiVisionProvider and LocalCatalogOCRProvider interface compliance
"""

import os
import sys
import pytest
import numpy as np
from PIL import Image

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.ai.common.base_provider import (
    ProviderType,
    ProviderStatus,
    BoundingBox,
    OCRResult,
    TranslationResult,
    InterpretationResult,
    LinguisticAnalysisResult,
)
from services.ai.common.error_handler import (
    KallvettuAIError,
    ProviderNotConfiguredError,
    AuthenticationError,
    RateLimitExceededError,
    CharacterLimitExceededError,
    sanitize_error_string,
    format_provider_error,
)
from services.ai.common.response_validator import (
    validate_tamil_unicode,
    calculate_tamil_character_ratio,
    clamp_confidence,
    normalize_bounding_box,
    clean_text_artifacts,
)
from services.ai.common.provider_manager import (
    ProviderManager,
    get_provider_manager,
)
from services.ai.gemini.vision_provider import GeminiVisionProvider
from services.ai.offline.offline_ocr_provider import LocalCatalogOCRProvider


# ==============================================================================
# 1. DATA MODEL TESTS
# ==============================================================================

def test_bounding_box_to_dict():
    """Verify BoundingBox data model serializes correctly with rounded coordinates."""
    box = BoundingBox(ymin=0.12345, xmin=0.23456, ymax=0.34567, xmax=0.45678, confidence=0.88888, text="வழுதி")
    d = box.to_dict()
    assert d["ymin"] == 0.1235
    assert d["xmin"] == 0.2346
    assert d["ymax"] == 0.3457
    assert d["xmax"] == 0.4568
    assert d["confidence"] == 0.8889
    assert d["text"] == "வழுதி"


def test_ocr_result_to_dict():
    """Verify OCRResult serializes with all required epigraphical fields."""
    res = OCRResult(
        text="ஸ்வஸ்திஸ்ரீ",
        provider_name="Test OCR",
        model_id="test-v1",
        confidence=0.95,
        latency_seconds=1.2345,
        is_cloud=True,
        is_historical_script=True,
        provenance_notes="Verified Inscription",
        metadata={"dynasty": "Chola"}
    )
    d = res.to_dict()
    assert d["text"] == "ஸ்வஸ்திஸ்ரீ"
    assert d["provider_name"] == "Test OCR"
    assert d["confidence"] == 0.95
    assert round(d["latency_seconds"], 2) == 1.23
    assert d["is_cloud"] is True
    assert d["metadata"]["dynasty"] == "Chola"


def test_translation_result_to_dict():
    """Verify TranslationResult contract structure."""
    trans = TranslationResult(
        translated_text="Hail Prosperity!",
        source_text="ஸ்வஸ்திஸ்ரீ",
        source_language="ta",
        target_language="en",
        provider_name="Test NMT",
        model_id="test-translate-v1",
        latency_seconds=0.456,
        confidence=0.98
    )
    d = trans.to_dict()
    assert d["translated_text"] == "Hail Prosperity!"
    assert d["source_language"] == "ta"
    assert d["target_language"] == "en"
    assert d["confidence"] == 0.98


# ==============================================================================
# 2. SECURITY & ERROR HANDLER TESTS
# ==============================================================================

def test_sanitize_error_string_strips_credentials():
    """Verify error string sanitizer removes API keys and tokens."""
    raw = "Request failed with AIzaSyD-1234567890abcdef and token=abcd1234efgh5678"
    sanitized = sanitize_error_string(raw)
    assert "AIzaSyD-1234567890abcdef" not in sanitized
    assert "[REDACTED_KEY]" in sanitized
    assert "[REDACTED_TOKEN]" in sanitized


def test_format_provider_error_unconfigured():
    """Verify unconfigured provider generates polite guidance message."""
    err = ProviderNotConfiguredError("Missing key", provider="Sarvam AI")
    msg = format_provider_error(err, "Sarvam AI")
    assert "⚠️" in msg
    assert "Sarvam AI is not configured" in msg


def test_format_provider_error_auth_and_rate_limit():
    """Verify HTTP 401 and 429 errors are translated cleanly."""
    err401 = AuthenticationError("401 Unauthorized request", provider="Sarvam AI")
    assert "🔒" in format_provider_error(err401, "Sarvam AI")

    err429 = RateLimitExceededError("429 Too Many Requests", provider="Sarvam")
    assert "⏱️" in format_provider_error(err429, "Sarvam")


# ==============================================================================
# 3. RESPONSE VALIDATOR TESTS
# ==============================================================================

def test_validate_tamil_unicode():
    """Verify Tamil Unicode character detection."""
    assert validate_tamil_unicode("கல்வெட்டு") is True
    assert validate_tamil_unicode("Brahmi 123") is False
    assert validate_tamil_unicode("") is False
    assert validate_tamil_unicode(None) is False
    assert validate_tamil_unicode("English with தமிழ் word") is True


def test_calculate_tamil_character_ratio():
    """Verify Tamil character proportion metric."""
    assert calculate_tamil_character_ratio("கல்வெட்டு") == 1.0
    assert calculate_tamil_character_ratio("Stone Inscription") == 0.0
    # Half Tamil, half ASCII
    ratio = calculate_tamil_character_ratio("தமிழ் text")
    assert 0.4 < ratio < 0.8


def test_clamp_confidence():
    """Verify confidence score normalization."""
    assert clamp_confidence(0.85) == 0.85
    assert clamp_confidence(85.0) == 0.85  # Percentage to 0-1
    assert clamp_confidence(1.5) == 1.0    # Capped at 1.0
    assert clamp_confidence(-0.5) == 0.0   # Bounded at 0.0
    assert clamp_confidence(None) is None


def test_normalize_bounding_box_absolute_and_normalized():
    """Verify coordinate normalization from pixel bounds to [0.0, 1.0]."""
    # Absolute pixel box in 1000x800 image
    box = normalize_bounding_box([100, 200, 300, 400], image_width=1000, image_height=800)
    assert box is not None
    assert box.ymin == 0.125
    assert box.xmin == 0.2
    assert box.ymax == 0.375
    assert box.xmax == 0.4

    # Dict format
    box_dict = normalize_bounding_box({"ymin": 0.1, "xmin": 0.2, "ymax": 0.5, "xmax": 0.6}, 100, 100)
    assert box_dict is not None
    assert box_dict.ymin == 0.1
    assert box_dict.xmax == 0.6


def test_clean_text_artifacts():
    """Verify markdown fences and control characters are stripped."""
    dirty = "```json\n{\"text\": \"கல்வெட்டு\"}\n```\x00"
    cleaned = clean_text_artifacts(dirty)
    assert "```" not in cleaned
    assert "\x00" not in cleaned
    assert "கல்வெட்டு" in cleaned


# ==============================================================================
# 4. PROVIDER MANAGER TESTS
# ==============================================================================

def test_provider_manager_lifecycle():
    """Verify registration, enabling/disabling, and retrieval in ProviderManager."""
    mgr = ProviderManager()
    gemini_prov = GeminiVisionProvider()
    catalog_prov = LocalCatalogOCRProvider()

    mgr.register_provider(catalog_prov, is_default=True)
    mgr.register_provider(gemini_prov)

    # Retrieval
    assert mgr.get_provider("local_catalog_ocr") is catalog_prov
    assert mgr.get_ocr_provider() is catalog_prov

    # Enablement toggle
    assert mgr.is_enabled("local_catalog_ocr") is True
    mgr.set_enabled("local_catalog_ocr", False)
    assert mgr.is_enabled("local_catalog_ocr") is False
    assert mgr.get_ocr_provider("local_catalog_ocr") is None

    # Re-enable
    mgr.set_enabled("local_catalog_ocr", True)
    assert mgr.get_ocr_provider("local_catalog_ocr") is catalog_prov


def test_provider_manager_status_reporting():
    """Verify get_all_statuses returns structured metadata for UI rendering."""
    mgr = ProviderManager()
    prov = LocalCatalogOCRProvider()
    mgr.register_provider(prov)

    statuses = mgr.get_all_statuses()
    assert "local_catalog_ocr" in statuses
    assert statuses["local_catalog_ocr"]["configured"] in [True, "YES"]
    assert statuses["local_catalog_ocr"]["enabled"] in [True, "Active"]
    assert statuses["local_catalog_ocr"]["status"] in ["available", "configured", "Available"]


def test_local_catalog_ocr_execution():
    """Verify LocalCatalogOCRProvider successfully looks up preset ground truth."""
    prov = LocalCatalogOCRProvider()
    assert prov.is_configured() is True

    # Execute preset lookup
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    res = prov.recognize(dummy_img, preset_id="mangulam_brahmi")
    assert isinstance(res, OCRResult)
    assert "நந்த" in res.text or "இகுவன்" in res.text or "பாளிய்" in res.text
    assert res.is_cloud is False
    assert res.confidence == 1.0


def test_unconfigured_gemini_raises_gracefully():
    """Verify GeminiVisionProvider raises ProviderNotConfiguredError when key is empty."""
    prov = GeminiVisionProvider(api_key="")
    # Health check reflects state
    healthy, msg = prov.check_health()
    assert healthy is False

    dummy_img = np.zeros((50, 50, 3), dtype=np.uint8)
    with pytest.raises(ProviderNotConfiguredError):
        prov.recognize(dummy_img)
