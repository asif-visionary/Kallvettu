"""
Sarvam AI Mayura Translation & Advanced Text Processing Provider for Kallvettu.
Implements BaseTranslationProvider using official SDK model 'mayura:v1'.
Features:
- Strict 1,000 character input ceiling validation
- Multiple linguistic modes: 'formal', 'modern-colloquial', 'classic-colloquial', 'code-mixed'
- Transliteration scripts: 'roman', 'fully-native', 'spoken-form-in-native'
- Automatic source language detection ('auto')
- Sentence-aware chunking for long texts
- Translation history tracking
"""

import time
import uuid
import logging
from typing import Dict, Any, List, Optional, Tuple

from services.ai.common.base_provider import (
    BaseTranslationProvider,
    ProviderType,
    TranslationResult,
)
from services.ai.common.error_handler import (
    ProviderNotConfiguredError,
    AuthenticationError,
    RateLimitExceededError,
    CharacterLimitExceededError,
    InvalidInputError,
    KallvettuAIError,
)
from services.ai.common.response_validator import clean_text_artifacts
from .client import SarvamClient, get_sarvam_client
from .translation_provider import chunk_text_sentence_aware, normalize_sarvam_lang

logger = logging.getLogger("kallvettu.sarvam.mayura")

# Valid modes and scripts supported by Mayura
VALID_MODES = {"formal", "modern-colloquial", "classic-colloquial", "code-mixed"}
VALID_OUTPUT_SCRIPTS = {"roman", "fully-native", "spoken-form-in-native"}


class MayuraTranslationProvider(BaseTranslationProvider):
    """
    Advanced Indic Text Processing & Translation Provider powered by Sarvam Mayura (mayura:v1).
    """

    MODEL_ID = "mayura:v1"
    MAX_CHARS_PER_REQUEST = 1000

    def __init__(self, client: Optional[SarvamClient] = None):
        self._client = client or get_sarvam_client()
        self._history: List[Dict[str, Any]] = []

    @property
    def provider_name(self) -> str:
        return "Sarvam AI Mayura (Advanced Translation & Transliteration)"

    @property
    def provider_id(self) -> str:
        return "sarvam_mayura"

    def is_configured(self) -> bool:
        return self._client.is_configured()

    def check_health(self) -> Tuple[bool, str]:
        return self._client.check_health()

    def get_history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()

    def set_client_for_testing(self, mock_sdk: Any) -> None:
        """Sets internal SDK client for mock testing."""
        self._client.set_sdk_client(mock_sdk)

    def translate(
        self,
        text: str,
        source_lang: str = "ta-IN",
        target_lang: str = "en-IN",
        model: Optional[str] = None,
        **kwargs: Any
    ) -> TranslationResult:
        """
        Translates text using mayura:v1 with support for colloquial modes and transliteration scripts.
        Strictly enforces the 1,000 character limit per request.
        """
        if not self.is_configured():
            raise ProviderNotConfiguredError(
                "Sarvam AI is not configured. Supply SARVAM_API_KEY.",
                provider=self.provider_name
            )

        clean_input = clean_text_artifacts(text)
        if not clean_input:
            raise InvalidInputError("Input text to translate cannot be empty.", provider=self.provider_name)

        # Handle auto source language detection (supported on mayura)
        if source_lang.lower() == "auto":
            src_lang = "auto"
        else:
            src_lang = normalize_sarvam_lang(source_lang, default="ta-IN")

        tgt_lang = normalize_sarvam_lang(target_lang, default="en-IN")

        # Linguistic mode validation
        mode = kwargs.get("mode", "formal")
        if mode not in VALID_MODES:
            mode = "formal"

        # Transliteration script option (mayura only: roman, fully-native, etc.)
        output_script = kwargs.get("output_script")
        if output_script and output_script not in VALID_OUTPUT_SCRIPTS:
            output_script = None

        # Chunk text if exceeding 900 chars (safe threshold for 1000 limit)
        chunks = chunk_text_sentence_aware(clean_input, max_chunk_chars=900)

        sdk = self._client.get_sdk_client()
        t0 = time.perf_counter()

        translated_parts: List[str] = []
        captured_req_ids: List[str] = []

        for idx, chunk in enumerate(chunks):
            if len(chunk) > self.MAX_CHARS_PER_REQUEST:
                raise CharacterLimitExceededError(
                    f"Chunk {idx+1} length ({len(chunk)}) exceeds Mayura {self.MAX_CHARS_PER_REQUEST} character ceiling.",
                    provider=self.provider_name
                )

            call_kwargs: Dict[str, Any] = {
                "input": chunk,
                "source_language_code": src_lang,
                "target_language_code": tgt_lang,
                "model": self.MODEL_ID,
                "mode": mode,
            }
            if output_script:
                call_kwargs["output_script"] = output_script

            try:
                response = sdk.text.translate(**call_kwargs)
                out_text = getattr(response, "translated_text", "")
                if not out_text and isinstance(response, dict):
                    out_text = response.get("translated_text", "")

                req_id = getattr(response, "request_id", None) or f"mayura-{uuid.uuid4().hex[:8]}"
                captured_req_ids.append(str(req_id))
                translated_parts.append(out_text.strip())

            except Exception as e:
                raw_err = str(e)
                if "401" in raw_err or "unauthorized" in raw_err.lower():
                    raise AuthenticationError(
                        "Sarvam API key is unauthorized or invalid.",
                        provider=self.provider_name,
                        raw_error=e
                    ) from e
                elif "429" in raw_err or "rate limit" in raw_err.lower():
                    raise RateLimitExceededError(
                        "Mayura translation rate limit exceeded.",
                        provider=self.provider_name,
                        raw_error=e
                    ) from e
                elif "422" in raw_err or "character" in raw_err.lower():
                    raise CharacterLimitExceededError(
                        f"Mayura character limit violation: {raw_err}",
                        provider=self.provider_name,
                        raw_error=e
                    ) from e
                else:
                    raise KallvettuAIError(
                        f"Mayura translation request failed: {raw_err}",
                        provider=self.provider_name,
                        raw_error=e
                    ) from e

        total_latency = time.perf_counter() - t0
        final_translation = " ".join(translated_parts).strip()
        primary_req_id = captured_req_ids[0] if captured_req_ids else f"mayura-{uuid.uuid4().hex[:8]}"

        result = TranslationResult(
            translated_text=final_translation,
            source_text=clean_input,
            source_language=src_lang,
            target_language=tgt_lang,
            provider_name=self.provider_name,
            model_id=self.MODEL_ID,
            latency_seconds=total_latency,
            request_id=primary_req_id,
            confidence=0.96,
            metadata={
                "mode": mode,
                "output_script": output_script,
                "chunks_count": len(chunks),
                "request_ids": captured_req_ids,
            }
        )

        self._history.append({
            "timestamp": time.time(),
            "source_text": clean_input[:100],
            "translated_text": final_translation[:100],
            "mode": mode,
            "output_script": output_script,
            "request_id": primary_req_id,
        })

        return result
