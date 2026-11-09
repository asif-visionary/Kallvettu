"""
Base Provider Contracts & Data Models for Kallvettu Multi-Provider AI Architecture.
Defines typed interfaces and result objects for OCR, Translation, Interpretation, and Linguistics.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from PIL import Image


class ProviderType(str, Enum):
    """Categories of AI services available in the Kallvettu platform."""
    OCR = "ocr"
    TRANSLATION = "translation"
    INTERPRETATION = "interpretation"
    LINGUISTIC = "linguistic"


class ProviderStatus(str, Enum):
    """Operational status of an integrated provider."""
    CONFIGURED = "configured"      # Credentials present, ready for execution
    AVAILABLE = "available"        # Verified responsive
    UNCONFIGURED = "unconfigured"  # Missing credentials (safe offline mode)
    DEGRADED = "degraded"          # Partial functionality or rate limited
    ERROR = "error"                # Unrecoverable error or invalid credentials


@dataclass
class BoundingBox:
    """Normalized spatial bounding box for recognized lithic characters/lines."""
    ymin: float  # Top coordinate (0.0 to 1.0)
    xmin: float  # Left coordinate (0.0 to 1.0)
    ymax: float  # Bottom coordinate (0.0 to 1.0)
    xmax: float  # Right coordinate (0.0 to 1.0)
    confidence: Optional[float] = None
    text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ymin": round(self.ymin, 4),
            "xmin": round(self.xmin, 4),
            "ymax": round(self.ymax, 4),
            "xmax": round(self.xmax, 4),
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "text": self.text,
        }


@dataclass
class OCRResult:
    """Unified structured response contract for all lithic OCR engines."""
    text: str
    provider_name: str
    model_id: str
    confidence: Optional[float] = None
    latency_seconds: float = 0.0
    bounding_boxes: List[BoundingBox] = field(default_factory=list)
    raw_response: Optional[Dict[str, Any]] = None
    is_cloud: bool = True
    is_historical_script: bool = False
    provenance_notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "provider_name": self.provider_name,
            "model_id": self.model_id,
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "latency_seconds": round(self.latency_seconds, 3),
            "bounding_boxes": [b.to_dict() for b in self.bounding_boxes],
            "is_cloud": self.is_cloud,
            "is_historical_script": self.is_historical_script,
            "provenance_notes": self.provenance_notes,
            "metadata": self.metadata,
        }


@dataclass
class TranslationResult:
    """Unified translation contract for Sarvam AI, Gemini, and Classical NLP."""
    translated_text: str
    source_text: str
    source_language: str
    target_language: str
    provider_name: str
    model_id: str
    latency_seconds: float = 0.0
    request_id: Optional[str] = None
    confidence: Optional[float] = None
    raw_response: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "translated_text": self.translated_text,
            "source_text": self.source_text,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "provider_name": self.provider_name,
            "model_id": self.model_id,
            "latency_seconds": round(self.latency_seconds, 3),
            "request_id": self.request_id,
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "metadata": self.metadata,
        }


@dataclass
class InterpretationResult:
    """Unified Classical Tamil linguistic and contextual interpretation contract."""
    modern_tamil: str
    linguistic_explanation: str
    uncertain_words: List[str] = field(default_factory=list)
    alternative_interpretations: List[str] = field(default_factory=list)
    limitations: str = ""
    provider_name: str = ""
    model_id: str = ""
    latency_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def interpretation_text(self) -> str:
        return str(self.metadata.get("full_response") or self.linguistic_explanation or self.modern_tamil)

    @property
    def historical_context(self) -> str:
        return str(self.metadata.get("historical_context", ""))

    @property
    def key_takeaways(self) -> List[str]:
        return list(self.metadata.get("key_takeaways", []))

    @property
    def entities_identified(self) -> List[str]:
        return list(self.metadata.get("entities_identified", []))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modern_tamil": self.modern_tamil,
            "linguistic_explanation": self.linguistic_explanation,
            "uncertain_words": self.uncertain_words,
            "alternative_interpretations": self.alternative_interpretations,
            "limitations": self.limitations,
            "provider_name": self.provider_name,
            "model_id": self.model_id,
            "latency_seconds": round(self.latency_seconds, 3),
            "metadata": self.metadata,
        }


@dataclass
class LinguisticAnalysisResult:
    """Unified result for language detection, transliteration, glossary, and NER."""
    detected_language: str = ""
    transliteration: str = ""
    named_entities: List[Dict[str, Any]] = field(default_factory=list)
    glossary_matches: List[Dict[str, Any]] = field(default_factory=list)
    normalized_text: str = ""
    provider_name: str = ""
    model_id: str = ""
    latency_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def iso15919_transliteration(self) -> str:
        return self.transliteration

    @property
    def language_confidence(self) -> float:
        return float(self.metadata.get("language_confidence", 1.0))

    @property
    def historical_glossary_matches(self) -> List[Dict[str, Any]]:
        return self.glossary_matches

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected_language": self.detected_language,
            "transliteration": self.transliteration,
            "named_entities": self.named_entities,
            "glossary_matches": self.glossary_matches,
            "normalized_text": self.normalized_text,
            "provider_name": self.provider_name,
            "model_id": self.model_id,
            "latency_seconds": round(self.latency_seconds, 3),
            "metadata": self.metadata,
        }


# ==============================================================================
# ABSTRACT BASE PROVIDER CLASSES
# ==============================================================================

class BaseProvider(ABC):
    """Abstract base class for all external and local AI providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider brand name (e.g., 'Sarvam AI', 'Google Gemini Vision')."""
        pass

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique identifier key (e.g., 'sarvam_vision', 'paddleocr_local')."""
        pass

    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Functional capability category."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Returns True if required credentials or models are present."""
        pass

    @abstractmethod
    def check_health(self) -> Tuple[bool, str]:
        """Verifies responsiveness or configuration validity. Returns (is_healthy, status_message)."""
        pass


class BaseOCRProvider(BaseProvider):
    """Abstract contract for Optical Character Recognition engines."""

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.OCR

    @abstractmethod
    def recognize(
        self,
        image: Union[bytes, Image.Image, np.ndarray, str],
        script_hint: Optional[str] = None,
        **kwargs: Any
    ) -> OCRResult:
        """
        Executes character recognition on the given image.
        Accepts raw image bytes, PIL Image, NumPy array, or local file path.
        """
        pass


class BaseTranslationProvider(BaseProvider):
    """Abstract contract for Neural Machine Translation engines."""

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.TRANSLATION

    @abstractmethod
    def translate(
        self,
        text: str,
        source_lang: str = "ta",
        target_lang: str = "en",
        model: Optional[str] = None,
        **kwargs: Any
    ) -> TranslationResult:
        """Translates text between supported language pairs."""
        pass


class BaseInterpretationProvider(BaseProvider):
    """Abstract contract for Classical Tamil LLM contextual interpretation engines."""

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.INTERPRETATION

    @abstractmethod
    def interpret(
        self,
        transcription: str,
        script: Optional[str] = None,
        era: Optional[str] = None,
        **kwargs: Any
    ) -> InterpretationResult:
        """Produces modern Tamil interpretation and linguistic breakdown."""
        pass

    def chat(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        **kwargs: Any
    ) -> str:
        """Executes conversational or interactive Q&A completion."""
        raise NotImplementedError("Conversational chat is not implemented for this provider.")


class BaseLinguisticProvider(BaseProvider):
    """Abstract contract for auxiliary NLP services (NER, Transliteration, Glossary)."""

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.LINGUISTIC

    @abstractmethod
    def analyze(
        self,
        text: str,
        operations: Optional[List[str]] = None,
        **kwargs: Any
    ) -> LinguisticAnalysisResult:
        """Executes linguistic intelligence services on Tamil text."""
        pass

