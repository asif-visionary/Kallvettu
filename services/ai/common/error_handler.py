"""
Unified Exception Hierarchy & Error Formatter for Kallvettu AI Providers.
Ensures no API credentials leak into logs or UI notifications.
"""

import re
from typing import Optional


class KallvettuAIError(Exception):
    """Base exception for all AI provider integration failures."""
    def __init__(self, message: str, provider: str = "Unknown", raw_error: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.raw_error = raw_error


class ProviderNotConfiguredError(KallvettuAIError):
    """Raised when an API call is attempted on an unconfigured provider."""
    pass


class AuthenticationError(KallvettuAIError):
    """Raised on HTTP 401/403 or invalid API subscription keys."""
    pass


class RateLimitExceededError(KallvettuAIError):
    """Raised on HTTP 429 rate limit or quota exhaustion."""
    pass


class ServiceUnavailableError(KallvettuAIError):
    """Raised on HTTP 500/502/503 or remote connection drops."""
    pass


class InvalidInputError(KallvettuAIError):
    """Raised when input text or image payload violates provider constraints."""
    pass


class CharacterLimitExceededError(InvalidInputError):
    """Raised when text exceeds hard request ceilings (e.g., Sarvam 2000 chars, Mayura 1000 chars)."""
    pass


class ResponseValidationError(KallvettuAIError):
    """Raised when the provider returns unparseable or corrupted payload."""
    pass


def sanitize_error_string(text: str) -> str:
    """Strips API keys, bearer tokens, and internal URL query strings from error messages."""
    if not text:
        return ""
    # Strip potential api keys or tokens
    sanitized = re.sub(r"(AIzaSy|sk-|api-subscription-key[\s:=]+)[A-Za-z0-9_\-]{8,}", "[REDACTED_KEY]", text)
    sanitized = re.sub(r"(token=)[A-Za-z0-9_\-\.]{8,}", r"\1[REDACTED_TOKEN]", sanitized)
    return sanitized


def format_provider_error(error: Exception, provider_name: str = "Provider") -> str:
    """
    Translates low-level HTTP and SDK errors into clear, researcher-facing notifications.
    """
    raw_msg = str(error)
    clean_msg = sanitize_error_string(raw_msg)

    if isinstance(error, ProviderNotConfiguredError):
        return f"⚠️ {provider_name} is not configured. Please supply credentials in Settings or .env."

    if isinstance(error, AuthenticationError) or "401" in raw_msg or "403" in raw_msg or "unauthorized" in raw_msg.lower():
        return f"🔒 {provider_name} authentication failed: API subscription key is invalid or unauthorized."

    if isinstance(error, RateLimitExceededError) or "429" in raw_msg or "rate limit" in raw_msg.lower():
        return f"⏱️ {provider_name} rate limit reached: please pause for a few seconds before retrying."

    if isinstance(error, CharacterLimitExceededError):
        return f"📏 {provider_name} input limit exceeded: {clean_msg}"

    if "timeout" in raw_msg.lower():
        return f"⌛ {provider_name} request timed out after network wait. Provider may be under high load."

    if "connection" in raw_msg.lower() or "enotfound" in raw_msg.lower():
        return f"🌐 Network connection to {provider_name} failed. Check your internet connectivity."

    return f"❌ {provider_name} operation failed: {clean_msg}"

