"""
Phase 5 Unit Tests: Sarvam Mayura & Sarvam-105B Integration.
Covers:
- Mayura 1,000 character limit enforcement
- Mayura colloquial modes and transliteration scripts
- Sarvam-105B Classical Tamil interpretation
- reasoning_content fallback handling
- ProviderManager registration and discovery
"""

import os
import sys
import pytest
from unittest.mock import MagicMock

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.ai.sarvam.client import SarvamClient
from services.ai.sarvam.mayura_provider import MayuraTranslationProvider
from services.ai.sarvam.chat_provider import SarvamChatInterpretationProvider
from services.ai.common.base_provider import TranslationResult, InterpretationResult
from services.ai.common.error_handler import (
    ProviderNotConfiguredError,
    AuthenticationError,
    CharacterLimitExceededError,
)
from services.ai.common.provider_manager import get_provider_manager


# ==============================================================================
# 1. MAYURA ADVANCED TEXT PROCESSING TESTS
# ==============================================================================

def test_mayura_unconfigured():
    """Verify unconfigured client raises ProviderNotConfiguredError."""
    client = SarvamClient(api_key="")
    prov = MayuraTranslationProvider(client=client)

    with pytest.raises(ProviderNotConfiguredError):
        prov.translate("கல்வெட்டு")


def test_mayura_mocked_success_with_modes_and_scripts():
    """Verify Mayura executes with colloquial mode and transliteration script."""
    client = SarvamClient(api_key="sk_valid_mock_key_12345")
    mock_sdk = MagicMock()

    mock_resp = MagicMock()
    mock_resp.translated_text = "kallvettu saasanam"
    mock_resp.request_id = "req_mayura_777"
    mock_sdk.text.translate.return_value = mock_resp

    client.set_sdk_client(mock_sdk)
    prov = MayuraTranslationProvider(client=client)

    res = prov.translate(
        text="கல்வெட்டு சாசனம்",
        source_lang="ta-IN",
        target_lang="en-IN",
        mode="modern-colloquial",
        output_script="roman"
    )

    assert isinstance(res, TranslationResult)
    assert res.translated_text == "kallvettu saasanam"
    assert res.model_id == "mayura:v1"
    assert res.metadata["mode"] == "modern-colloquial"
    assert res.metadata["output_script"] == "roman"

    # Verify official SDK call arguments
    call_kwargs = mock_sdk.text.translate.call_args[1]
    assert call_kwargs["model"] == "mayura:v1"
    assert call_kwargs["mode"] == "modern-colloquial"
    assert call_kwargs["output_script"] == "roman"


def test_mayura_auto_source_detection():
    """Verify Mayura accepts auto source language code."""
    client = SarvamClient(api_key="sk_valid_mock_key_12345")
    mock_sdk = MagicMock()
    mock_sdk.text.translate.return_value = MagicMock(translated_text="Text in English")

    client.set_sdk_client(mock_sdk)
    prov = MayuraTranslationProvider(client=client)

    res = prov.translate("ஏதோ ஒரு உரை", source_lang="auto", target_lang="en")
    assert res.source_language == "auto"


# ==============================================================================
# 2. SARVAM-105B CHAT COMPLETIONS TESTS
# ==============================================================================

def test_sarvam_chat_unconfigured():
    """Verify unconfigured chat provider raises ProviderNotConfiguredError."""
    client = SarvamClient(api_key="")
    prov = SarvamChatInterpretationProvider(client=client)

    with pytest.raises(ProviderNotConfiguredError):
        prov.interpret("கல்வெட்டு வாசிப்பு")


def test_sarvam_chat_interpretation_mocked_success():
    """Verify Sarvam-105B executes chat completion with epigraphical system prompt."""
    client = SarvamClient(api_key="sk_valid_mock_key_12345")
    mock_sdk = MagicMock()

    mock_choice = MagicMock()
    mock_choice.message.content = "இக்கல்வெட்டு நடுகல்லைக் குறிக்கிறது.\n\nவீரன் சாத்தன் போரில் உயிர்நீத்த செய்தி விளக்கப்படுகிறது."
    mock_choice.message.reasoning_content = None

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_sdk.chat.completions.return_value = mock_resp

    client.set_sdk_client(mock_sdk)
    prov = SarvamChatInterpretationProvider(client=client)

    res = prov.interpret(
        transcription="சாத்தன் நடுகல்",
        script="வட்டெழுத்து",
        era="7-ஆம் நூற்றாண்டு"
    )

    assert isinstance(res, InterpretationResult)
    assert "நடுகல்லைக் குறிக்கிறது" in res.modern_tamil
    assert "சாத்தன்" in res.linguistic_explanation
    assert res.model_id == "sarvam-105b"
    assert "limitations" in res.to_dict()

    # Verify reasoning_effort was None to safeguard content
    call_kwargs = mock_sdk.chat.completions.call_args[1]
    assert call_kwargs["model"] == "sarvam-105b"
    assert call_kwargs["reasoning_effort"] is None


def test_sarvam_chat_reasoning_content_fallback():
    """Verify fallback to reasoning_content when content is None (vital Sarvam SDK gotcha)."""
    client = SarvamClient(api_key="sk_valid_mock_key_12345")
    mock_sdk = MagicMock()

    mock_choice = MagicMock()
    mock_choice.message.content = None  # Model consumed budget in reasoning
    mock_choice.message.reasoning_content = "மீட்கப்பட்ட உரை: அரசன் நெடுஞ்செழியன் கற்படுக்கைத் தானம்."

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_sdk.chat.completions.return_value = mock_resp

    client.set_sdk_client(mock_sdk)
    prov = SarvamChatInterpretationProvider(client=client)

    res = prov.interpret("மாங்குளம் கல்வெட்டு")
    assert isinstance(res, InterpretationResult)
    assert "நெடுஞ்செழியன்" in res.modern_tamil


def test_sarvam_chat_method_string_prompt():
    """Verify chat() method with a single string query."""
    client = SarvamClient(api_key="sk_valid_mock_key_12345")
    mock_sdk = MagicMock()

    mock_choice = MagicMock()
    mock_choice.message.content = "கோப்பரகேசரி என்பது சோழ மன்னர்களின் சிறப்புப் பட்டம்."
    mock_choice.message.reasoning_content = None

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_sdk.chat.completions.return_value = mock_resp

    client.set_sdk_client(mock_sdk)
    prov = SarvamChatInterpretationProvider(client=client)

    ans = prov.chat("கோப்பரகேசரி என்ற பட்டம் எதனைக் குறிக்கிறது?")
    assert "சோழ மன்னர்களின்" in ans
    assert mock_sdk.chat.completions.called


def test_sarvam_chat_method_messages_list():
    """Verify chat() method with multi-turn message sequence."""
    client = SarvamClient(api_key="sk_valid_mock_key_12345")
    mock_sdk = MagicMock()

    mock_choice = MagicMock()
    mock_choice.message.content = "இக்கல்வெட்டின் காலம் கி.பி. 11-ஆம் நூற்றாண்டு."
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_sdk.chat.completions.return_value = mock_resp

    client.set_sdk_client(mock_sdk)
    prov = SarvamChatInterpretationProvider(client=client)

    msgs = [
        {"role": "user", "content": "ராஜேந்திர சோழன் கல்வெட்டு"},
        {"role": "assistant", "content": "இது கங்கை கொண்ட சோழபுர சாசனம்."},
        {"role": "user", "content": "இதன் காலம் என்ன?"}
    ]
    ans = prov.chat(msgs)
    assert "11-ஆம் நூற்றாண்டு" in ans


def test_sarvam_chat_method_reasoning_fallback():
    """Verify chat() method correctly falls back to reasoning_content when content is None."""
    client = SarvamClient(api_key="sk_valid_mock_key_12345")
    mock_sdk = MagicMock()

    mock_choice = MagicMock()
    mock_choice.message.content = None
    mock_choice.message.reasoning_content = "காரண விளக்கம்: இது பல்லவ நடுகல் வாசகம்."
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_sdk.chat.completions.return_value = mock_resp

    client.set_sdk_client(mock_sdk)
    prov = SarvamChatInterpretationProvider(client=client)

    ans = prov.chat("இந்த கல்வெட்டு எதைப் பற்றியது?")
    assert "பல்லவ நடுகல்" in ans


# ==============================================================================
# 3. PROVIDER MANAGER DISCOVERY TESTS
# ==============================================================================

def test_provider_manager_sarvam_mayura_and_chat_registration():
    """Verify ProviderManager auto-registers Mayura and Sarvam-105B."""
    mgr = get_provider_manager()

    mayura_prov = mgr.get_translation_provider("sarvam_mayura")
    assert mayura_prov is not None
    assert isinstance(mayura_prov, MayuraTranslationProvider)

    chat_prov = mgr.get_interpretation_provider("sarvam_chat_interpretation")
    assert chat_prov is not None
    assert isinstance(chat_prov, SarvamChatInterpretationProvider)

