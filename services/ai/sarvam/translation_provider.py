"""
Sarvam AI Text Translation Provider for Kallvettu.
Implements BaseTranslationProvider using official SDK method `client.text.translate()`.
Features:
- Primary model: `sarvam-translate:v1` (max 2,000 chars)
- Strict language code normalization (ta-IN, en-IN, od-IN, etc.)
- Sentence-aware and grapheme-safe text chunking for long epigraphical edicts
- Request ID capture and segment-level provenance
- Translation history tracking
- Granular error translation (401, 429, 422)
"""

import re
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple, Union

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
    ServiceUnavailableError,
    InvalidInputError,
    KallvettuAIError,
)
from services.ai.common.response_validator import clean_text_artifacts
from .client import SarvamClient, get_sarvam_client


# Standard Sarvam language code mapping
SARVAM_LANG_MAP = {
    "ta": "ta-IN",
    "ta-in": "ta-IN",
    "en": "en-IN",
    "en-in": "en-IN",
    "hi": "hi-IN",
    "hi-in": "hi-IN",
    "ml": "ml-IN",
    "ml-in": "ml-IN",
    "te": "te-IN",
    "te-in": "te-IN",
    "kn": "kn-IN",
    "kn-in": "kn-IN",
    "bn": "bn-IN",
    "bn-in": "bn-IN",
    "gu": "gu-IN",
    "gu-in": "gu-IN",
    "mr": "mr-IN",
    "mr-in": "mr-IN",
    "pa": "pa-IN",
    "pa-in": "pa-IN",
    "od": "od-IN",
    "od-in": "od-IN",
    "or": "od-IN",   # Correct historical/ISO typo to Sarvam's expected 'od-IN'
    "or-in": "od-IN",
}


def normalize_sarvam_lang(lang_code: Optional[str], default: str = "ta-IN") -> str:
    """Normalizes language code into Sarvam format (e.g. 'ta' -> 'ta-IN')."""
    if not lang_code:
        return default
    key = lang_code.strip().lower()
    return SARVAM_LANG_MAP.get(key, lang_code.strip())


def chunk_text_sentence_aware(text: str, max_chunk_chars: int = 1800) -> List[str]:
    """
    Splits long epigraphical text into sentence-aware chunks without breaking words
    or exceeding the 2,000 character limit of sarvam-translate:v1.
    Respects Tamil sentence ends, paragraph breaks, and epigraphical delimiters.
    """
    if not text:
        return []

    text_clean = text.strip()
    if len(text_clean) <= max_chunk_chars:
        return [text_clean]

    # Regex delimiters: double newline, period with space, semicolon, colon, or lithic bar
    sentences = re.split(r'(\n\n|\n|[.?!।|]\s+)', text_clean)

    chunks: List[str] = []
    current_chunk = ""

    for part in sentences:
        if not part:
            continue

        # If a single sentence exceeds the chunk ceiling, split it by words
        if len(part) > max_chunk_chars:
            words = part.split(' ')
            for word in words:
                if len(current_chunk) + len(word) + 1 <= max_chunk_chars:
                    current_chunk = f"{current_chunk} {word}".strip()
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = word
            continue

        if len(current_chunk) + len(part) <= max_chunk_chars:
            current_chunk += part
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = part

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text_clean]


class SarvamTranslateProvider(BaseTranslationProvider):
    """
    Primary Cloud Translation Provider powered by Sarvam AI (sarvam-translate:v1).
    """

    MODEL_ID = "sarvam-translate:v1"
    MAX_CHARS_PER_REQUEST = 2000

    def __init__(self, client: Optional[SarvamClient] = None):
        self._client = client or get_sarvam_client()
        self._history: List[Dict[str, Any]] = []

    @property
    def provider_name(self) -> str:
        return "Sarvam AI Translate"

    @property
    def provider_id(self) -> str:
        return "sarvam_translate"

    def is_configured(self) -> bool:
        return self._client.is_configured()

    def check_health(self) -> Tuple[bool, str]:
        return self._client.check_health()

    def get_history(self) -> List[Dict[str, Any]]:
        """Returns session translation history."""
        return list(self._history)

    def clear_history(self) -> None:
        """Clears translation history."""
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
        Translates Tamil or Indic text using sarvam-translate:v1.
        Automatically handles sentence-aware chunking for long inscriptions.
        """
        if not self.is_configured():
            raise ProviderNotConfiguredError(
                "Sarvam AI is not configured. Supply SARVAM_API_KEY in environment or Settings.",
                provider=self.provider_name
            )

        clean_input = clean_text_artifacts(text)
        if not clean_input:
            raise InvalidInputError("Input text to translate cannot be empty.", provider=self.provider_name)

        src_lang = normalize_sarvam_lang(source_lang, default="ta-IN")
        tgt_lang = normalize_sarvam_lang(target_lang, default="en-IN")
        model_name = model or self.MODEL_ID

        # Ensure sarvam-translate:v1 never receives unsupported auto source language
        if src_lang.lower() == "auto":
            src_lang = "ta-IN"

        # Divide into chunks if length approaches or exceeds 2,000 chars
        chunks = chunk_text_sentence_aware(clean_input, max_chunk_chars=1800)

        sdk = self._client.get_sdk_client()
        t0 = time.perf_counter()

        translated_parts: List[str] = []
        segment_records: List[Dict[str, Any]] = []
        captured_req_ids: List[str] = []

        for idx, chunk in enumerate(chunks):
            # Strict character limit verification per chunk
            if len(chunk) > self.MAX_CHARS_PER_REQUEST:
                raise CharacterLimitExceededError(
                    f"Chunk {idx+1} length ({len(chunk)}) exceeds Sarvam {self.MAX_CHARS_PER_REQUEST} character ceiling.",
                    provider=self.provider_name
                )

            try:
                # sarvam-translate:v1 only supports mode='formal' and does NOT support output_script
                mode = kwargs.get("mode", "formal")
                if mode != "formal":
                    mode = "formal"

                # Execute official SDK call
                response = sdk.text.translate(
                    input=chunk,
                    source_language_code=src_lang,
                    target_language_code=tgt_lang,
                    model=model_name,
                    mode=mode,
                )

                # Extract translated text
                out_text = getattr(response, "translated_text", "")
                if not out_text and isinstance(response, dict):
                    out_text = response.get("translated_text", "")

                req_id = getattr(response, "request_id", None) or f"sarvam-{uuid.uuid4().hex[:8]}"
                captured_req_ids.append(str(req_id))

                translated_parts.append(out_text.strip())
                segment_records.append({
                    "segment_index": idx,
                    "source_chunk": chunk,
                    "translated_chunk": out_text.strip(),
                    "request_id": str(req_id),
                })

            except Exception as e:
                raw_err = str(e)
                if "401" in raw_err or "unauthorized" in raw_err.lower():
                    raise AuthenticationError(
                        "Sarvam AI subscription key is invalid or unauthorized.",
                        provider=self.provider_name,
                        raw_error=e
                    ) from e
                elif "429" in raw_err or "rate limit" in raw_err.lower():
                    raise RateLimitExceededError(
                        "Sarvam AI rate limit exceeded.",
                        provider=self.provider_name,
                        raw_error=e
                    ) from e
                elif "422" in raw_err or "character" in raw_err.lower():
                    raise CharacterLimitExceededError(
                        f"Sarvam AI character limit violation: {raw_err}",
                        provider=self.provider_name,
                        raw_error=e
                    ) from e
                elif "connection" in raw_err.lower() or "timeout" in raw_err.lower():
                    raise ServiceUnavailableError(
                        f"Network connection to Sarvam AI failed: {raw_err}",
                        provider=self.provider_name,
                        raw_error=e
                    ) from e
                else:
                    raise KallvettuAIError(
                        f"Sarvam translation request failed: {raw_err}",
                        provider=self.provider_name,
                        raw_error=e
                    ) from e

        total_latency = time.perf_counter() - t0
        final_translation = " ".join(translated_parts).strip()
        primary_req_id = captured_req_ids[0] if captured_req_ids else f"sarvam-{uuid.uuid4().hex[:8]}"

        result = TranslationResult(
            translated_text=final_translation,
            source_text=clean_input,
            source_language=src_lang,
            target_language=tgt_lang,
            provider_name=self.provider_name,
            model_id=model_name,
            latency_seconds=total_latency,
            request_id=primary_req_id,
            confidence=0.95,
            metadata={
                "chunks_count": len(chunks),
                "segments": segment_records,
                "request_ids": captured_req_ids,
                "mode": "formal",
            }
        )

        # Append to session history
        self._history.append({
            "timestamp": time.time(),
            "source_text": clean_input[:100] + ("..." if len(clean_input) > 100 else ""),
            "translated_text": final_translation[:100] + ("..." if len(final_translation) > 100 else ""),
            "source_lang": src_lang,
            "target_lang": tgt_lang,
            "request_id": primary_req_id,
        })

        return result

