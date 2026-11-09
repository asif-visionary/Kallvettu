"""Common Provider Architecture & Data Contracts for Kallvettu AI Ecosystem."""

from .base_provider import (
    ProviderType,
    ProviderStatus,
    BoundingBox,
    OCRResult,
    TranslationResult,
    InterpretationResult,
    LinguisticAnalysisResult,
    BaseProvider,
    BaseOCRProvider,
    BaseTranslationProvider,
    BaseInterpretationProvider,
    BaseLinguisticProvider,
)
from .error_handler import (
    KallvettuAIError,
    ProviderNotConfiguredError,
    AuthenticationError,
    RateLimitExceededError,
    ServiceUnavailableError,
    InvalidInputError,
    CharacterLimitExceededError,
    ResponseValidationError,
    format_provider_error,
)
from .response_validator import (
    validate_tamil_unicode,
    calculate_tamil_character_ratio,
    normalize_bounding_box,
    clamp_confidence,
    clean_text_artifacts,
)
from .provider_manager import ProviderManager, get_provider_manager
from .comparison_engine import (
    MultiEngineOCRComparator,
    MultiOCRComparisonResult,
    DisputedToken,
    get_ocr_comparator,
)
from .source_identity import (
    ProvenanceMode,
    SourceIdentity,
    compute_image_sha256,
    create_source_identity,
)

__all__ = [
    "ProviderType",
    "ProviderStatus",
    "BoundingBox",
    "OCRResult",
    "TranslationResult",
    "InterpretationResult",
    "LinguisticAnalysisResult",
    "BaseProvider",
    "BaseOCRProvider",
    "BaseTranslationProvider",
    "BaseInterpretationProvider",
    "BaseLinguisticProvider",
    "KallvettuAIError",
    "ProviderNotConfiguredError",
    "AuthenticationError",
    "RateLimitExceededError",
    "ServiceUnavailableError",
    "InvalidInputError",
    "CharacterLimitExceededError",
    "ResponseValidationError",
    "format_provider_error",
    "validate_tamil_unicode",
    "calculate_tamil_character_ratio",
    "normalize_bounding_box",
    "clamp_confidence",
    "clean_text_artifacts",
    "ProviderManager",
    "get_provider_manager",
    "MultiEngineOCRComparator",
    "MultiOCRComparisonResult",
    "DisputedToken",
    "get_ocr_comparator",
    "ProvenanceMode",
    "SourceIdentity",
    "compute_image_sha256",
    "create_source_identity",
]

