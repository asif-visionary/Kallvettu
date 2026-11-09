"""
Sarvam-105B Classical Tamil Linguistic & Epigraphical Interpretation Provider for Kallvettu.
Implements BaseInterpretationProvider using official SDK client.chat.completions().
Features:
- Models: 'sarvam-105b' (128K context reasoning) and 'sarvam-105b-conversations' (32K real-time)
- Strict epigraphical system prompt preserving names, dates, and lacunae
- Content extraction with reasoning_content fallback protection
- Structured linguistic outputs (modern Tamil, explanation, uncertain words, limitations)
- Scientific honesty disclaimer (AI interpretation is never treated as verified lithic proof)
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Union

from services.ai.common.base_provider import (
    BaseInterpretationProvider,
    ProviderType,
    InterpretationResult,
)
from services.ai.common.error_handler import (
    ProviderNotConfiguredError,
    AuthenticationError,
    RateLimitExceededError,
    InvalidInputError,
    KallvettuAIError,
)
from services.ai.common.response_validator import clean_text_artifacts
from .client import SarvamClient, get_sarvam_client

logger = logging.getLogger("kallvettu.sarvam.chat")

# Standard epigraphical scholar system instruction
EPIGRAPHY_SYSTEM_PROMPT = (
    "You are an expert Classical Tamil linguistic scholar and epigraphist. "
    "Your responsibility is to interpret ancient Tamil stone inscriptions into understandable, "
    "grammatically pure modern Tamil and clear English. "
    "Preserve historical names, dynastic titles, regnal dates, and administrative terms faithfully. "
    "Do not fabricate missing or illegible text; if letters are damaged or inside brackets [...], highlight them. "
    "Provide: 1) Modern Tamil translation, 2) Word-level linguistic notes, 3) Historical context, and 4) Uncertain readings."
)


class SarvamChatInterpretationProvider(BaseInterpretationProvider):
    """
    Classical Tamil Interpretation Provider powered by Sarvam-105B.
    """

    PRIMARY_MODEL = "sarvam-105b"
    FALLBACK_MODEL = "sarvam-105b-conversations"

    def __init__(self, client: Optional[SarvamClient] = None, model: Optional[str] = None):
        self._client = client or get_sarvam_client()
        self._model = model or self.PRIMARY_MODEL

    @property
    def provider_name(self) -> str:
        return f"Sarvam AI Chat ({self._model})"

    @property
    def provider_id(self) -> str:
        return "sarvam_chat_interpretation"

    def is_configured(self) -> bool:
        return self._client.is_configured()

    def check_health(self) -> Tuple[bool, str]:
        return self._client.check_health()

    def set_client_for_testing(self, mock_sdk: Any) -> None:
        """Sets internal SDK client for mock testing."""
        self._client.set_sdk_client(mock_sdk)

    def interpret(
        self,
        transcription: str,
        script: Optional[str] = None,
        era: Optional[str] = None,
        **kwargs: Any
    ) -> InterpretationResult:
        """
        Interprets approved ancient Tamil transcription using Sarvam-105B.
        """
        if not self.is_configured():
            raise ProviderNotConfiguredError(
                "Sarvam AI is not configured. Supply SARVAM_API_KEY.",
                provider=self.provider_name
            )

        clean_text = clean_text_artifacts(transcription)
        if not clean_text:
            raise InvalidInputError("Transcription input cannot be empty.", provider=self.provider_name)

        t0 = time.perf_counter()
        sdk = self._client.get_sdk_client()
        model_name = kwargs.get("model", self._model)

        user_content = f"Interpret this approved inscription transcription:\n\n{clean_text}"
        script_val = script or kwargs.get("script_hint")
        if script_val or era:
            user_content += f"\n\nContext: Script = {script_val or 'Classical Tamil'}, Estimated Era = {era or 'Historical'}"

        messages = [
            {"role": "system", "content": EPIGRAPHY_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

        try:
            # reasoning_effort=None ensures reasoning tokens do not exhaust token budget
            response = sdk.chat.completions(
                model=model_name,
                messages=messages,
                reasoning_effort=None,
                max_tokens=kwargs.get("max_tokens", 2000)
            )

            choice = response.choices[0]
            raw_content = choice.message.content

            # Critical gotcha: if content is None, fallback to reasoning_content
            if not raw_content and hasattr(choice.message, "reasoning_content"):
                raw_content = choice.message.reasoning_content

            final_text = clean_text_artifacts(raw_content or "")

        except Exception as e:
            raw_err = str(e)
            if "401" in raw_err or "unauthorized" in raw_err.lower():
                raise AuthenticationError(
                    "Sarvam API key is unauthorized or invalid for chat.",
                    provider=self.provider_name,
                    raw_error=e
                ) from e
            elif "429" in raw_err or "rate limit" in raw_err.lower():
                raise RateLimitExceededError(
                    "Sarvam-105B rate limit exceeded.",
                    provider=self.provider_name,
                    raw_error=e
                ) from e
            else:
                raise KallvettuAIError(
                    f"Sarvam-105B chat completion failed: {raw_err}",
                    provider=self.provider_name,
                    raw_error=e
                ) from e

        latency = time.perf_counter() - t0

        # Parse sections if present
        modern_tamil_part = final_text
        explanation_part = f"Linguistic and historical interpretation generated by {model_name}."

        if "\n\n" in final_text:
            parts = final_text.split("\n\n", 1)
            modern_tamil_part = parts[0].strip()
            explanation_part = parts[1].strip()

        # Extract structured findings if present in response
        key_takeaways = []
        entities = []
        historical_context = ""

        for line in final_text.splitlines():
            line_str = line.strip()
            if line_str.startswith("- ") or line_str.startswith("* "):
                key_takeaways.append(line_str.lstrip("-* ").strip())
            elif "Historical Context:" in line_str or "வரலாற்றுப் பின்னணி:" in line_str:
                historical_context = line_str.split(":", 1)[1].strip()
            elif "Entities:" in line_str or "பெயர்கள்:" in line_str:
                raw_ents = line_str.split(":", 1)[1].strip()
                entities = [e.strip(" .,") for e in raw_ents.split(",") if e.strip()]

        return InterpretationResult(
            modern_tamil=modern_tamil_part,
            linguistic_explanation=explanation_part,
            uncertain_words=[],
            alternative_interpretations=[],
            limitations=(
                "AI-assisted linguistic interpretation. Generated by Sarvam-105B. "
                "Must not be cited as primary epigraphical evidence without physical squeeze verification."
            ),
            provider_name=self.provider_name,
            model_id=model_name,
            latency_seconds=latency,
            metadata={
                "source_transcription": clean_text,
                "script": script,
                "era": era,
                "full_response": final_text,
                "key_takeaways": key_takeaways,
                "entities_identified": entities,
                "historical_context": historical_context,
            }
        )

    def chat(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 1500,
        **kwargs: Any
    ) -> str:
        """
        Executes interactive multi-turn or single-turn conversational completion using Sarvam-105B.
        Supports raw text queries or OpenAI-style message sequences.
        """
        if not self.is_configured():
            raise ProviderNotConfiguredError(
                "Sarvam AI is not configured. Supply SARVAM_API_KEY in environment or Settings.",
                provider=self.provider_name
            )

        sdk = self._client.get_sdk_client()
        model_name = model or self._model
        sys_msg = system_prompt or EPIGRAPHY_SYSTEM_PROMPT

        if isinstance(prompt, str):
            clean_prompt = clean_text_artifacts(prompt)
            if not clean_prompt:
                raise InvalidInputError("Prompt cannot be empty.", provider=self.provider_name)
            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": clean_prompt}
            ]
        elif isinstance(prompt, list):
            if not prompt:
                raise InvalidInputError("Message list cannot be empty.", provider=self.provider_name)
            has_system = any(m.get("role") == "system" for m in prompt if isinstance(m, dict))
            if not has_system:
                messages = [{"role": "system", "content": sys_msg}] + prompt
            else:
                messages = prompt
        else:
            raise InvalidInputError(f"Unsupported prompt type: {type(prompt)}", provider=self.provider_name)

        try:
            # reasoning_effort=None ensures reasoning tokens do not exhaust token budget
            response = sdk.chat.completions(
                model=model_name,
                messages=messages,
                reasoning_effort=None,
                max_tokens=max_tokens
            )

            choice = response.choices[0]
            raw_content = choice.message.content

            # Critical gotcha: if content is None, fallback to reasoning_content
            if not raw_content and hasattr(choice.message, "reasoning_content"):
                raw_content = choice.message.reasoning_content

            return clean_text_artifacts(raw_content or "")

        except Exception as e:
            raw_err = str(e)
            if "401" in raw_err or "unauthorized" in raw_err.lower():
                raise AuthenticationError(
                    "Sarvam API key is unauthorized or invalid for chat.",
                    provider=self.provider_name,
                    raw_error=e
                ) from e
            elif "429" in raw_err or "rate limit" in raw_err.lower():
                raise RateLimitExceededError(
                    "Sarvam-105B rate limit exceeded.",
                    provider=self.provider_name,
                    raw_error=e
                ) from e
            else:
                raise KallvettuAIError(
                    f"Sarvam-105B chat query failed: {raw_err}",
                    provider=self.provider_name,
                    raw_error=e
                ) from e
