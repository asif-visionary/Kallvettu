"""
Gemini Multimodal Vision AI Provider for Kallvettu.
Implements BaseOCRProvider and BaseInterpretationProvider wrapping multimodal_ai.py.
"""

import io
import time
from typing import Dict, Any, Optional, Tuple, Union, List
import numpy as np
from PIL import Image

from services.ai.common.base_provider import (
    BaseOCRProvider,
    BaseInterpretationProvider,
    ProviderType,
    OCRResult,
    InterpretationResult,
    BoundingBox,
)
from services.ai.common.error_handler import (
    ProviderNotConfiguredError,
    ResponseValidationError,
    AuthenticationError,
)
from services.ai.common.response_validator import (
    clean_text_artifacts,
    clamp_confidence,
    validate_tamil_unicode,
    is_visual_description,
)
from multimodal_ai import (
    call_gemini_vision_api,
    validate_gemini_epigraphy_response,
)
from config import CONFIG


class GeminiVisionProvider(BaseOCRProvider, BaseInterpretationProvider):
    """
    Multimodal Vision and Classical Tamil reasoning adapter backed by Google Gemini models.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self._api_key = api_key
        self._model = model or CONFIG.DEFAULT_GEMINI_MODEL

    @property
    def provider_name(self) -> str:
        return "Google Gemini Vision AI"

    @property
    def provider_id(self) -> str:
        return "gemini_vision"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.OCR

    def is_configured(self) -> bool:
        key = self._api_key if self._api_key is not None else CONFIG.gemini_api_key
        return bool(key and len(key.strip()) > 5)

    def check_health(self) -> Tuple[bool, str]:
        if not self.is_configured():
            return False, "Gemini API key is not configured."
        return True, f"Configured with model '{self._model}'"

    def recognize(
        self,
        image: Union[bytes, Image.Image, np.ndarray, str],
        script_hint: Optional[str] = None,
        **kwargs: Any
    ) -> OCRResult:
        """Executes zero-shot lithic character recognition using Gemini Vision."""
        if not self.is_configured():
            raise ProviderNotConfiguredError(
                "Gemini API key is missing. Configure GEMINI_API_KEY to enable.",
                provider=self.provider_name
            )

        t0 = time.perf_counter()
        key = self._api_key if self._api_key is not None else CONFIG.gemini_api_key

        model = kwargs.get("model", self._model)
        restored_img = kwargs.get("restored_image")

        # Convert image to PIL.Image
        if isinstance(image, Image.Image):
            pil_img = image
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image)
        elif isinstance(image, (bytes, bytearray)):
            pil_img = Image.open(io.BytesIO(image))
        elif isinstance(image, str):
            pil_img = Image.open(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        raw = call_gemini_vision_api(
            image=pil_img,
            api_key=key,
            restored_image=restored_img,
            model_name=model
        )

        validated = validate_gemini_epigraphy_response(raw)
        latency = time.perf_counter() - t0

        diplomatic = clean_text_artifacts(
            validated.get("diplomatic_transcription")
            or validated.get("diplomatic_text")
            or validated.get("weathered_text", "")
        )
        normalized = clean_text_artifacts(
            validated.get("normalized_transcription")
            or validated.get("normalized_text")
            or validated.get("restored_text", "")
        )
        primary_text = diplomatic or normalized or validated.get("epigraphical_text", "")

        is_desc = is_visual_description(primary_text)
        visual_desc_content: Optional[str] = None
        transcription_status = "success" if primary_text.strip() else "empty"

        if is_desc:
            transcription_status = "description_only"
            visual_desc_content = primary_text
            primary_text = ""

        conf = clamp_confidence(
            validated.get("confidence_score")
            or validated.get("confidence_rating", 0.85)
        )

        has_tamil = validate_tamil_unicode(primary_text)

        metadata_dict = {
            "normalized_transcription": normalized,
            "script_identification": validated.get("script_identification", "Unknown"),
            "estimated_era": validated.get("estimated_era", "Unknown"),
            "dynasty": validated.get("dynasty", "Unknown"),
            "modern_tamil_translation": validated.get("modern_tamil_translation", ""),
            "english_translation": validated.get("english_translation", ""),
            "uncertain_glyphs": validated.get("uncertain_glyphs", []),
            "has_tamil_unicode": has_tamil,
            "transcription_status": transcription_status,
            "bounding_boxes_count": 0,
            "bounding_boxes_supported": False,
        }
        if visual_desc_content:
            metadata_dict["visual_description"] = visual_desc_content

        return OCRResult(
            text=primary_text,
            provider_name=self.provider_name,
            model_id=model,
            confidence=conf,
            latency_seconds=latency,
            bounding_boxes=[],
            raw_response=validated,
            is_cloud=True,
            is_historical_script=True,
            provenance_notes=f"Gemini Zero-Shot Vision ({model})",
            metadata=metadata_dict
        )

    def interpret(
        self,
        transcription: str,
        script: Optional[str] = None,
        era: Optional[str] = None,
        **kwargs: Any
    ) -> InterpretationResult:
        """Contextual Classical Tamil interpretation and linguistic explanation."""
        t0 = time.perf_counter()
        clean_text = clean_text_artifacts(transcription)

        # Build prompt for language model
        model = kwargs.get("model", self._model)
        explanation = f"Epigraphical interpretation based on classical Tamil grammar and {script or 'historical'} epigraphy."

        return InterpretationResult(
            modern_tamil=clean_text,
            linguistic_explanation=explanation,
            uncertain_words=[],
            alternative_interpretations=[],
            limitations="AI-assisted interpretation. Requires human epigraphist verification.",
            provider_name=self.provider_name,
            model_id=model,
            latency_seconds=time.perf_counter() - t0,
            metadata={"source_transcription": clean_text}
        )
