"""
Sarvam AI Client Wrapper for Kallvettu.
Manages authentication, client lifecycle, and SDK access for Sarvam services:
- Text Translation (sarvam-translate:v1, mayura:v1)
- Document AI (Sarvam Vision 3B parse, extract, classify, split)
- Chat Completions (sarvam-105b, sarvam-105b-conversations)
"""

import os
import logging
from typing import Optional, Any
from config import CONFIG
from services.ai.common.error_handler import (
    ProviderNotConfiguredError,
    AuthenticationError,
    RateLimitExceededError,
    ServiceUnavailableError,
    KallvettuAIError,
)

logger = logging.getLogger("kallvettu.sarvam.client")


class SarvamClient:
    """
    Wrapper around official SarvamAI Python SDK.
    Provides singleton caching, credential resolution, and structured error mapping.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._explicit_key = api_key
        self._sdk_client: Optional[Any] = None

    @property
    def api_key(self) -> str:
        """Retrieves active API key from instance or global config."""
        if self._explicit_key is not None:
            return self._explicit_key.strip()
        return CONFIG.sarvam_api_key

    def is_configured(self) -> bool:
        """Returns True if a non-empty API key is present."""
        key = self.api_key
        return bool(key and len(key) > 5)

    def get_sdk_client(self) -> Any:
        """
        Initializes and returns the official sarvamai.SarvamAI SDK client.
        Raises ProviderNotConfiguredError if no key is configured.
        """
        if not self.is_configured():
            raise ProviderNotConfiguredError(
                "Sarvam AI API key is not configured. Set SARVAM_API_KEY to enable.",
                provider="Sarvam AI"
            )

        if self._sdk_client is None:
            try:
                from sarvamai import SarvamAI
                self._sdk_client = SarvamAI(api_subscription_key=self.api_key)
            except ImportError as e:
                raise KallvettuAIError(
                    "sarvamai package is not installed. Run 'pip install sarvamai'.",
                    provider="Sarvam AI",
                    raw_error=e
                ) from e
            except Exception as e:
                raise KallvettuAIError(
                    f"Failed to initialize SarvamAI SDK client: {e}",
                    provider="Sarvam AI",
                    raw_error=e
                ) from e

        return self._sdk_client

    def set_sdk_client(self, client: Any) -> None:
        """Allows injecting a mock SDK client for testing."""
        self._sdk_client = client

    def check_health(self) -> tuple[bool, str]:
        """Checks configuration and basic viability."""
        if not self.is_configured():
            return False, "Sarvam AI API key is not configured."
        return True, "Sarvam AI client configured and ready."


# Global Singleton Client
_GLOBAL_SARVAM_CLIENT: Optional[SarvamClient] = None


def get_sarvam_client() -> SarvamClient:
    """Returns shared singleton instance of SarvamClient."""
    global _GLOBAL_SARVAM_CLIENT
    if _GLOBAL_SARVAM_CLIENT is None:
        _GLOBAL_SARVAM_CLIENT = SarvamClient()
    return _GLOBAL_SARVAM_CLIENT

