"""
Local Catalog & Offline Epigraphical OCR Provider for Kallvettu.
Provides zero-dependency fallback for curated inscriptions and manual researcher transcriptions.
"""

import time
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image

from services.ai.common.base_provider import (
    BaseOCRProvider,
    ProviderType,
    OCRResult,
    BoundingBox,
)
from nlp_translation_engine import find_preset_by_id, KNOWN_INSCRIPTIONS_CATALOG


class LocalCatalogOCRProvider(BaseOCRProvider):
    """
    Offline OCR provider leveraging curated archaeological catalog records.
    Requires no cloud API keys, network access, or external weights.
    """

    @property
    def provider_name(self) -> str:
        return "Archaeological Catalog Reference (Offline)"

    @property
    def provider_id(self) -> str:
        return "local_catalog_ocr"

    def is_configured(self) -> bool:
        return True

    def check_health(self) -> Tuple[bool, str]:
        return True, f"Catalog loaded with {len(KNOWN_INSCRIPTIONS_CATALOG)} verified records"

    def recognize(
        self,
        image: Union[bytes, Image.Image, np.ndarray, str],
        script_hint: Optional[str] = None,
        **kwargs: Any
    ) -> OCRResult:
        """Looks up curated ground-truth reading for preset inscriptions."""
        t0 = time.perf_counter()
        preset_id = kwargs.get("preset_id")
        text = kwargs.get("manual_text", "")

        if preset_id:
            record = find_preset_by_id(preset_id)
            if record:
                return OCRResult(
                    text=record.get("weathered_text", ""),
                    provider_name=self.provider_name,
                    model_id="ground-truth-epigraphy-catalog",
                    confidence=1.0,
                    latency_seconds=time.perf_counter() - t0,
                    bounding_boxes=[],
                    is_cloud=False,
                    is_historical_script=True,
                    provenance_notes=f"Curated Inscription ({record.get('title_en', '')})",
                    metadata=dict(record)
                )

        # Fallback to manual text or diagnostic notification
        return OCRResult(
            text=text,
            provider_name=self.provider_name,
            model_id="manual-transcription-fallback",
            confidence=1.0 if text else None,
            latency_seconds=time.perf_counter() - t0,
            bounding_boxes=[],
            is_cloud=False,
            is_historical_script=True,
            provenance_notes="Researcher Manual Transcription",
            metadata={"status": "manual_input"}
        )

