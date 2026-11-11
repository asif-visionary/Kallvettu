"""
Unified Configuration & Security Hardening Module for Kallvettu.
Provides typed configuration management, environment variable loading,
API key masking, filename sanitization, and path traversal protection.
"""

import os
import re
from dataclasses import dataclass
from typing import Set, Tuple, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ==============================================================================
# 1. SECURITY UTILITIES
# ==============================================================================

def sanitize_filename(filename: str, default: str = "kallvettu_artifact") -> str:
    """
    Sanitizes a user-supplied filename to prevent Path Traversal attacks (CWE-22).
    Strips directory separators, null bytes, and non-whitelisted characters.
    """
    if not filename or not isinstance(filename, str):
        return default

    # Extract base name only (strips any leading path components like ../ or C:\)
    base = os.path.basename(filename.strip())

    # Remove null bytes and control characters
    base = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", base)

    # Allow alphanumeric characters, underscores, hyphens, and single periods
    sanitized = re.sub(r"[^\w\-\.]", "_", base)

    # Disallow leading dots to prevent hidden files or traversal like '..'
    sanitized = sanitized.lstrip(".")

    # Avoid empty string after stripping
    if not sanitized:
        sanitized = default

    return sanitized


def mask_api_key(api_key: Optional[str]) -> str:
    """
    Masks sensitive API credentials for secure UI rendering and logging.
    Example: 'AIzaSyD-1234567890abcdef' -> 'AIzaSy...cdef'
    """
    if not api_key or not isinstance(api_key, str):
        return "Not Configured"

    key_clean = api_key.strip()
    if not key_clean:
        return "Not Configured"

    if len(key_clean) <= 8:
        return "********"

    prefix = key_clean[:6]
    suffix = key_clean[-4:]
    return f"{prefix}...{suffix}"


# ==============================================================================
# 2. CENTRALIZED CONFIGURATION CLASS
# ==============================================================================

@dataclass(frozen=True)
class KallvettuConfig:
    """Production configuration specification for the Kallvettu platform."""

    # Application Metadata
    APP_NAME: str = "Kallvettu"
    APP_TITLE: str = "Kallvettu — Archaeological Epigraphy & NLP Research Workbench"
    APP_VERSION: str = "3.0.0-production"

    # Environment
    ENV: str = os.getenv("KALLVETTU_ENV", "production")
    DEBUG: bool = os.getenv("KALLVETTU_DEBUG", "false").lower() in ("true", "1", "yes")

    # Storage Paths
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    ASSETS_DIR: str = os.path.join(BASE_DIR, "assets")
    SAMPLES_DIR: str = os.path.join(ASSETS_DIR, "samples")
    STYLES_PATH: str = os.path.join(ASSETS_DIR, "styles.css")

    # Ingestion & Security Ceilings
    MAX_UPLOAD_SIZE_BYTES: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50")) * 1024 * 1024
    ALLOWED_IMAGE_FORMATS: Tuple[str, ...] = ("JPEG", "JPG", "PNG", "WEBP", "TIFF", "BMP")
    MIN_IMAGE_DIM: int = 64
    MAX_IMAGE_DIM_DEFAULT: int = int(os.getenv("MAX_IMAGE_DIM_DEFAULT", "1600"))

    # Computer Vision Parameters
    CLAHE_CLIP_LIMIT_DEFAULT: float = float(os.getenv("CLAHE_CLIP_LIMIT_DEFAULT", "2.5"))
    CLAHE_TILE_GRID_DEFAULT: Tuple[int, int] = (8, 8)
    SAUVOLA_K_DEFAULT: float = float(os.getenv("SAUVOLA_K_DEFAULT", "0.20"))
    SAUVOLA_WINDOW_DEFAULT: int = int(os.getenv("SAUVOLA_WINDOW_DEFAULT", "25"))

    # AI & NLP Configuration
    DEFAULT_GEMINI_MODEL: str = os.getenv("DEFAULT_GEMINI_MODEL", "gemini-2.5-flash")
    SUPPORTED_GEMINI_MODELS: Tuple[str, ...] = ("gemini-2.5-flash", "gemini-2.5-pro", "gemini-flash-latest")
    API_TIMEOUT_SECONDS: int = int(os.getenv("API_TIMEOUT_SECONDS", "60"))

    # Sarvam AI Ecosystem Configuration
    SARVAM_BASE_URL: str = os.getenv("SARVAM_BASE_URL", "https://api.sarvam.ai")
    DEFAULT_SARVAM_TRANSLATE_MODEL: str = "sarvam-translate:v1"
    DEFAULT_MAYURA_MODEL: str = "mayura:v1"
    DEFAULT_SARVAM_CHAT_MODEL: str = "sarvam-105b"
    SARVAM_TRANSLATE_MAX_CHARS: int = 2000
    MAYURA_MAX_CHARS: int = 1000

    # PaddleOCR Local Engine Configuration
    PADDLE_CACHE_DIR: str = os.getenv(
        "PADDLE_PDX_CACHE_HOME",
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".paddlex_cache"))
    )
    DEFAULT_PADDLE_REC_MODEL: str = "ta_PP-OCRv5_mobile_rec"

    # Provider Feature Switches
    ENABLE_SARVAM: bool = os.getenv("ENABLE_SARVAM", "true").lower() in ("true", "1", "yes")
    ENABLE_GEMINI: bool = os.getenv("ENABLE_GEMINI", "true").lower() in ("true", "1", "yes")
    ENABLE_PADDLEOCR: bool = os.getenv("ENABLE_PADDLEOCR", "true").lower() in ("true", "1", "yes")

    @property
    def gemini_api_key(self) -> str:
        """Retrieves Gemini API key securely from environment."""
        return os.getenv("GEMINI_API_KEY", "").strip()

    @property
    def sarvam_api_key(self) -> str:
        """Retrieves Sarvam AI API key securely from environment."""
        return os.getenv("SARVAM_API_KEY", "").strip()


# Global Singleton Configuration Instance
CONFIG = KallvettuConfig()
