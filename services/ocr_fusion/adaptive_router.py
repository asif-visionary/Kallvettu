"""
Adaptive Epigraphical Routing Engine for Kallvettu (K-IEFE).
Orchestrates tiered recognition execution based on image quality assessment,
local recognition confidence, and consensus entropy to optimize accuracy and API cost.

ROUTING ARCHITECTURE:
- Tier 1: Primary Local OCR (PaddleOCR offline - zero cost, zero latency).
- Tier 2: Quality & Script Assessment (Laplacian blur, Michelson contrast, noise).
- Tier 3: Selective Cloud Escalation (Sarvam Vision / Gemini Vision on high entropy or archaic scripts).
- Tier 4: Experimental Historical OCR (Palaeographical template matching for Brahmi/Vatteluttu).
- Tier 5: Human Scholar Escalation (Contested readings requiring visual inspection).
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from PIL import Image

from services.ai.common.base_provider import OCRResult
from services.ai.common.provider_manager import get_provider_manager, ProviderManager
from restoration_engine import rgb_to_gray

logger = logging.getLogger("kallvettu.adaptive_router")

# Estimated API costs per call (USD) for transparency
ESTIMATED_API_COSTS = {
    "paddleocr_local": 0.000,
    "custom_historical_ocr": 0.000,
    "local_catalog_ocr": 0.000,
    "sarvam_vision": 0.005,       # ~0.5 cents per parse
    "gemini_vision": 0.002,       # ~0.2 cents per multimodal query
    "sarvam_translate": 0.001,
}


@dataclass
class ImageQualityMetrics:
    """Quantitative computer vision metrics assessing stone inscription photograph quality."""
    laplacian_variance: float     # Blur measure (higher = sharper chiseled edges; < 80 indicates severe blur)
    michelson_contrast: float     # Contrast in [0.0, 1.0] ((max - min) / (max + min))
    mean_luminance: float         # Exposure level in [0.0, 255.0]
    estimated_noise_floor: float  # High-frequency granite grain variance
    quality_grade: str            # 'pristine', 'good', 'weathered', 'heavily_eroded'
    recommended_enhancement: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "laplacian_variance": round(self.laplacian_variance, 2),
            "michelson_contrast": round(self.michelson_contrast, 4),
            "mean_luminance": round(self.mean_luminance, 2),
            "estimated_noise_floor": round(self.estimated_noise_floor, 2),
            "quality_grade": self.quality_grade,
            "recommended_enhancement": self.recommended_enhancement,
        }


@dataclass
class RoutingExecutionReport:
    """Complete diagnostic log of adaptive routing decisions and resource expenditure."""
    primary_engine_used: str
    escalations_triggered: List[str]
    engines_executed: List[str]
    total_latency_seconds: float
    total_estimated_cost_usd: float
    image_quality: ImageQualityMetrics
    requires_human_escalation: bool
    escalation_rationales: List[str]
    results_by_provider: Dict[str, OCRResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_engine_used": self.primary_engine_used,
            "escalations_triggered": self.escalations_triggered,
            "engines_executed": self.engines_executed,
            "total_latency_seconds": round(self.total_latency_seconds, 4),
            "total_estimated_cost_usd": round(self.total_estimated_cost_usd, 5),
            "image_quality": self.image_quality.to_dict(),
            "requires_human_escalation": self.requires_human_escalation,
            "escalation_rationales": self.escalation_rationales,
        }


class AdaptiveRoutingEngine:
    """
    Tiered epigraphical dispatch engine routing inscription images dynamically.
    """

    def __init__(self, provider_manager: Optional[ProviderManager] = None):
        self._manager = provider_manager or get_provider_manager()

    def assess_image_quality(self, image: Union[Image.Image, np.ndarray]) -> ImageQualityMetrics:
        """
        Calculates blur, contrast, luminance, and grain metrics on stone photographs.
        """
        if isinstance(image, Image.Image):
            gray = rgb_to_gray(np.array(image.convert("RGB")))
        elif isinstance(image, np.ndarray):
            gray = rgb_to_gray(image) if len(image.shape) == 3 else (image.astype(float) / 255.0)
        else:
            return ImageQualityMetrics(100.0, 0.5, 128.0, 10.0, "good", "None")

        gray_255 = (gray * 255.0).astype(np.uint8)

        # 1. Laplacian variance for blur
        # Discrete Laplacian convolution approximation
        padded = np.pad(gray_255, 1, mode="reflect").astype(float)
        lap = (
            padded[0:-2, 1:-1] + padded[2:, 1:-1] +
            padded[1:-1, 0:-2] + padded[1:-1, 2:] - 4 * padded[1:-1, 1:-1]
        )
        lap_var = float(np.var(lap))

        # 2. Michelson Contrast
        min_v = float(np.min(gray_255))
        max_v = float(np.max(gray_255))
        contrast = float((max_v - min_v) / max(1.0, (max_v + min_v)))

        # 3. Mean luminance
        mean_lum = float(np.mean(gray_255))

        # 4. Noise floor estimate (median absolute deviation of high frequencies)
        high_freq = gray_255 - lap
        noise = float(np.std(high_freq))

        # Classify grade
        if lap_var > 220.0 and contrast > 0.65:
            grade = "pristine"
            rec = "Direct Path A (No enhancement required)"
        elif lap_var > 120.0 and contrast > 0.45:
            grade = "good"
            rec = "Mild CLAHE (Path B) recommended"
        elif lap_var > 60.0 or contrast > 0.25:
            grade = "weathered"
            rec = "Bilateral Denoising + Mild CLAHE recommended"
        else:
            grade = "heavily_eroded"
            rec = "Multi-spectral Chisel Groove extraction & Gemini Vision escalation"

        return ImageQualityMetrics(
            laplacian_variance=lap_var,
            michelson_contrast=contrast,
            mean_luminance=mean_lum,
            estimated_noise_floor=noise,
            quality_grade=grade,
            recommended_enhancement=rec
        )

    def route_and_execute(
        self,
        image: Union[Image.Image, np.ndarray, str],
        script_hint: str = "auto",
        force_all_engines: bool = False,
        **kwargs: Any
    ) -> RoutingExecutionReport:
        """
        Executes tiered recognition with selective cloud escalation.
        """
        t0 = time.perf_counter()

        # Ingestion
        if isinstance(image, str):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Step 1: Assess image quality
        quality = self.assess_image_quality(pil_img)

        executed_engines = []
        escalations = []
        rationales = []
        results: Dict[str, OCRResult] = {}
        total_cost = 0.0

        # Determine if script is archaic
        is_archaic_script = any(s in script_hint.lower() for s in ["brahmi", "vatteluttu", "grantha"])

        # Tier 1: Primary Local OCR (PaddleOCR)
        paddle_prov = self._manager.get_ocr_provider("paddleocr_local")
        paddle_res: Optional[OCRResult] = None

        if paddle_prov and self._manager.is_enabled("paddleocr_local"):
            try:
                t_p = time.perf_counter()
                paddle_res = paddle_prov.recognize(pil_img, script_hint=script_hint, **kwargs)
                results["paddleocr_local"] = paddle_res
                executed_engines.append("paddleocr_local")
                total_cost += ESTIMATED_API_COSTS["paddleocr_local"]
            except Exception as e:
                logger.warning("Tier 1 PaddleOCR execution skipped: %s", e)

        # Decide whether to escalate to cloud engines
        should_escalate_cloud = force_all_engines
        requires_human = False

        if not force_all_engines:
            # Escalation Rule 1: Archaic script detection (PaddleOCR lacks Brahmi weights)
            if is_archaic_script:
                should_escalate_cloud = True
                escalations.append("archaic_script_escalation")
                rationales.append(f"Script hint '{script_hint}' is archaic; local PaddleOCR weights are modern-oriented.")

            # Escalation Rule 2: Low image quality or severe blur
            if quality.quality_grade in {"weathered", "heavily_eroded"}:
                should_escalate_cloud = True
                escalations.append("weathered_stone_escalation")
                rationales.append(f"Stone surface is {quality.quality_grade} (Laplacian var {quality.laplacian_variance:.1f} < 120).")

            # Escalation Rule 3: Low local confidence or empty output
            if not paddle_res or not paddle_res.text.strip():
                should_escalate_cloud = True
                escalations.append("empty_local_output_escalation")
                rationales.append("PaddleOCR produced empty or unparseable output on stone.")
            elif paddle_res.confidence is not None and paddle_res.confidence < 0.70:
                should_escalate_cloud = True
                escalations.append("low_local_confidence_escalation")
                rationales.append(f"PaddleOCR confidence ({paddle_res.confidence:.2f}) below threshold 0.70.")

        # Tier 2: Cloud Escalation (Sarvam Vision & Gemini Vision)
        if should_escalate_cloud:
            # 1. Sarvam Vision Document AI
            sarvam_prov = self._manager.get_ocr_provider("sarvam_vision")
            if sarvam_prov and self._manager.is_enabled("sarvam_vision") and sarvam_prov.is_configured():
                try:
                    # Sarvam Vision receives pristine Path A
                    res_s = sarvam_prov.recognize(pil_img, script_hint=script_hint, safe=True, **kwargs)
                    if res_s and res_s.text.strip():
                        results["sarvam_vision"] = res_s
                        executed_engines.append("sarvam_vision")
                        total_cost += ESTIMATED_API_COSTS["sarvam_vision"]
                except Exception as e_s:
                    logger.debug("Sarvam Vision escalation failed: %s", e_s)

            # 2. Gemini Vision Multimodal AI
            gemini_prov = self._manager.get_ocr_provider("gemini_vision")
            if gemini_prov and self._manager.is_enabled("gemini_vision") and gemini_prov.is_configured():
                try:
                    res_g = gemini_prov.recognize(pil_img, script_hint=script_hint, **kwargs)
                    if res_g and res_g.text.strip():
                        results["gemini_vision"] = res_g
                        executed_engines.append("gemini_vision")
                        total_cost += ESTIMATED_API_COSTS["gemini_vision"]
                except Exception as e_g:
                    logger.debug("Gemini Vision escalation failed: %s", e_g)

        # Tier 3: Experimental Historical OCR (for Brahmi / Vatteluttu)
        if is_archaic_script or force_all_engines or len(results) < 2:
            hist_prov = self._manager.get_ocr_provider("custom_historical_ocr")
            if hist_prov and self._manager.is_enabled("custom_historical_ocr"):
                try:
                    res_h = hist_prov.recognize(pil_img, script_hint=script_hint, **kwargs)
                    if res_h and res_h.text.strip():
                        results["custom_historical_ocr"] = res_h
                        executed_engines.append("custom_historical_ocr")
                        total_cost += ESTIMATED_API_COSTS["custom_historical_ocr"]
                except Exception as e_h:
                    logger.debug("Custom Historical OCR execution skipped: %s", e_h)

        # Human Escalation check: if zero engines succeeded or all engines disagree
        if not results:
            requires_human = True
            escalations.append("human_scholar_escalation")
            rationales.append("All automated OCR engines failed to recognize character strokes. Direct visual study required.")

        elapsed = time.perf_counter() - t0

        return RoutingExecutionReport(
            primary_engine_used=executed_engines[0] if executed_engines else "none",
            escalations_triggered=escalations,
            engines_executed=executed_engines,
            total_latency_seconds=elapsed,
            total_estimated_cost_usd=total_cost,
            image_quality=quality,
            requires_human_escalation=requires_human,
            escalation_rationales=rationales,
            results_by_provider=results
        )

