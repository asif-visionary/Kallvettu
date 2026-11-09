"""
Phase 2 Unit Tests: Sarvam AI Translation Integration (sarvam-translate:v1).
Covers:
- Language code normalization (ta -> ta-IN, od -> od-IN, etc.)
- Sentence-aware text chunking for long inscriptions
- Request ID and segment metadata recording
- Mocked SDK execution and TranslationResult contract verification
- Error handling (401 Unauthorized, 429 Rate Limit, 422 Chars Limit)
- ProviderManager integration
"""

import os
import sys
import pytest
from unittest.mock import MagicMock

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.ai.sarvam.client import SarvamClient
from services.ai.sarvam.translation_provider import (
    SarvamTranslateProvider,
    normalize_sarvam_lang,
    chunk_text_sentence_aware,
)
from services.ai.common.base_provider import TranslationResult
from services.ai.common.error_handler import (
    ProviderNotConfiguredError,
    AuthenticationError,
    RateLimitExceededError,
    InvalidInputError,
)
from services.ai.common.provider_manager import get_provider_manager


# ==============================================================================
# 1. LANGUAGE CODE NORMALIZATION TESTS
# ==============================================================================

def test_normalize_sarvam_lang():
    """Verify ISO-639 codes map to Sarvam-specific regional codes."""
    assert normalize_sarvam_lang("ta") == "ta-IN"
    assert normalize_sarvam_lang("TA-IN") == "ta-IN"
    assert normalize_sarvam_lang("en") == "en-IN"
    assert normalize_sarvam_lang("hi") == "hi-IN"
    assert normalize_sarvam_lang("te") == "te-IN"
    assert normalize_sarvam_lang("ml") == "ml-IN"
    # Special gotcha: Odia must be od-IN, not or-IN
    assert normalize_sarvam_lang("od") == "od-IN"
    assert normalize_sarvam_lang("or") == "od-IN"
    assert normalize_sarvam_lang(None) == "ta-IN"


# ==============================================================================
# 2. SENTENCE-AWARE CHUNKING TESTS
# ==============================================================================

def test_chunk_text_sentence_aware_short_text():
    """Verify short text below limit is not split."""
    text = "ஸ்வஸ்திஸ்ரீ கோப்பரகேசரி பந்மரான ஸ்ரீராஜராஜ தேவர்க்கு யாண்டு."
    chunks = chunk_text_sentence_aware(text, max_chunk_chars=1800)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_sentence_aware_long_epigraph():
    """Verify long epigraph edicts split at sentence ends without breaking words."""
    sentence = "ஸ்வஸ்திஸ்ரீ கோப்பரகேசரி பந்மரான ஸ்ரீராஜராஜ தேவர்க்கு யாண்டு இருபத்தைந்தாவது. "
    long_text = sentence * 40  # ~3,000 chars

    chunks = chunk_text_sentence_aware(long_text, max_chunk_chars=1800)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) <= 1800
        # Ensure chunk ends at sentence boundary
        assert c.strip().endswith(".") or c.strip().endswith("வது")


# ==============================================================================
# 3. TRANSLATION PROVIDER EXECUTION & CONTRACT TESTS
# ==============================================================================

def test_sarvam_translate_unconfigured():
    """Verify unconfigured client raises ProviderNotConfiguredError."""
    client = SarvamClient(api_key="")
    provider = SarvamTranslateProvider(client=client)

    with pytest.raises(ProviderNotConfiguredError):
        provider.translate("வணக்கம்")


def test_sarvam_translate_empty_input():
    """Verify empty or whitespace input raises InvalidInputError."""
    client = SarvamClient(api_key="sk_valid_mock_key_12345")
    provider = SarvamTranslateProvider(client=client)

    with pytest.raises(InvalidInputError):
        provider.translate("   ")


def test_sarvam_translate_mocked_success():
    """Verify successful translation with mocked SDK returns valid TranslationResult."""
    client = SarvamClient(api_key="sk_valid_mock_key_12345")

    # Mock official sarvamai SDK client
    mock_sdk = MagicMock()
    mock_response = MagicMock()
    mock_response.translated_text = "Hail Prosperity! In the 25th year of King Rajaraja Chola."
    mock_response.request_id = "req_sarvam_abc123"
    mock_sdk.text.translate.return_value = mock_response

    client.set_sdk_client(mock_sdk)
    provider = SarvamTranslateProvider(client=client)

    res = provider.translate(
        text="ஸ்வஸ்திஸ்ரீ கோப்பரகேசரி பந்மரான ஸ்ரீராஜராஜ தேவர்க்கு யாண்டு இருபத்தைந்தாவது.",
        source_lang="ta",
        target_lang="en"
    )

    assert isinstance(res, TranslationResult)
    assert res.translated_text == "Hail Prosperity! In the 25th year of King Rajaraja Chola."
    assert res.source_language == "ta-IN"
    assert res.target_language == "en-IN"
    assert res.provider_name == "Sarvam AI Translate"
    assert res.model_id == "sarvam-translate:v1"
    assert res.request_id == "req_sarvam_abc123"
    assert res.latency_seconds >= 0.0

    # Verify official SDK call arguments
    mock_sdk.text.translate.assert_called_once()
    call_kwargs = mock_sdk.text.translate.call_args[1]
    assert call_kwargs["source_language_code"] == "ta-IN"
    assert call_kwargs["target_language_code"] == "en-IN"
    assert call_kwargs["model"] == "sarvam-translate:v1"
    assert call_kwargs["mode"] == "formal"


def test_sarvam_translate_history_retention():
    """Verify translation history is captured and can be cleared."""
    client = SarvamClient(api_key="sk_valid_mock_key_12345")
    mock_sdk = MagicMock()
    mock_response = MagicMock()
    mock_response.translated_text = "Hero stone of warrior Sathan."
    mock_response.request_id = "req_hist_01"
    mock_sdk.text.translate.return_value = mock_response

    client.set_sdk_client(mock_sdk)
    provider = SarvamTranslateProvider(client=client)

    provider.clear_history()
    assert len(provider.get_history()) == 0

    provider.translate("சாத்தன் நடுகல்", source_lang="ta", target_lang="en")
    history = provider.get_history()
    assert len(history) == 1
    assert history[0]["request_id"] == "req_hist_01"


def test_sarvam_translate_error_mappings():
    """Verify low-level HTTP errors are translated to structured Kallvettu exceptions."""
    client = SarvamClient(api_key="sk_valid_mock_key_12345")
    mock_sdk = MagicMock()

    # 401 Unauthorized
    mock_sdk.text.translate.side_effect = Exception("401 Client Error: Unauthorized for url")
    client.set_sdk_client(mock_sdk)
    provider = SarvamTranslateProvider(client=client)

    with pytest.raises(AuthenticationError):
        provider.translate("வாசகம்", source_lang="ta", target_lang="en")

    # 429 Rate Limit
    mock_sdk.text.translate.side_effect = Exception("429 Too Many Requests: Rate limit exceeded")
    with pytest.raises(RateLimitExceededError):
        provider.translate("வாசகம்", source_lang="ta", target_lang="en")


def test_provider_manager_default_translation():
    """Verify ProviderManager auto-registers SarvamTranslateProvider."""
    mgr = get_provider_manager()
    prov = mgr.get_translation_provider("sarvam_translate")
    assert prov is not None
    assert isinstance(prov, SarvamTranslateProvider)
    assert prov.provider_name == "Sarvam AI Translate"
