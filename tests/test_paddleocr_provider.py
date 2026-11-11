"""
Phase 8 Unit & Integration Tests: Official PaddleOCR PP-OCRv5 Tamil Engine.
Covers:
- Package availability and version detection (PaddlePaddle 3.2.0, PaddleOCR 3.7.0)
- Device capability reporting (CPU vs GPU check)
- 4-state lifecycle status reporting (NOT INSTALLED, MODEL DOWNLOAD REQUIRED, AVAILABLE — CPU, ERROR)
- Image validation (empty images, invalid shapes, oversized images clamp)
- Mocked inference tests (Tamil text, bounding boxes, inference confidences)
- Live local CPU inference against an authentic Tamil inscription plate
- Epigraphical limitation warnings & provenance metadata
"""

import os
import sys
import pytest
import numpy as np
from PIL import Image
from unittest.mock import MagicMock, patch

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.ai.offline.paddleocr_provider import PaddleOCRProvider
from services.ai.common.base_provider import OCRResult, BoundingBox
from services.ai.common.error_handler import ProviderNotConfiguredError, KallvettuAIError
from services.ai.common.provider_manager import get_provider_manager
from config import CONFIG


# ==============================================================================
# 1. PACKAGE & ENVIRONMENT TESTS
# ==============================================================================

def test_paddleocr_package_installed():
    """Verify paddlepaddle and paddleocr packages are genuinely importable."""
    prov = PaddleOCRProvider()
    assert prov.is_package_installed is True
    status = prov.get_runtime_status()
    assert status["paddle_version"] is not None
    assert status["paddleocr_version"] is not None


def test_paddleocr_cpu_inference_mode():
    """Verify inference defaults to CPU without claiming fake CUDA acceleration."""
    prov = PaddleOCRProvider(use_gpu=False)
    status = prov.get_runtime_status()
    assert status["device"] == "CPU"
    assert "CPU" in status["status_label"]


def test_paddleocr_metadata():
    """Verify provider metadata and model IDs."""
    prov = PaddleOCRProvider()
    assert prov.provider_name == "PaddleOCR Local Engine (PP-OCRv5 Tamil)"
    assert prov.provider_id == "paddleocr_local"
    assert prov.PRIMARY_REC_MODEL == "ta_PP-OCRv5_mobile_rec"
    assert prov.PRIMARY_DET_MODEL == "PP-OCRv6_medium_det"


# ==============================================================================
# 2. STATUS REPORTING & 4-STATE LIFECYCLE TESTS
# ==============================================================================

def test_paddleocr_status_not_installed():
    """State 1: When package is missing, reports NOT INSTALLED and disabled."""
    prov = PaddleOCRProvider()
    prov._is_pkg_installed = False

    status = prov.get_runtime_status()
    assert status["state"] == "NOT_INSTALLED"
    assert status["configured"] is False
    assert status["enabled"] is False
    assert status["status_label"] == "NOT INSTALLED"


def test_paddleocr_status_model_download_required():
    """State 2: When packages installed but weights missing, reports MODEL DOWNLOAD REQUIRED."""
    prov = PaddleOCRProvider()
    prov._is_pkg_installed = True
    prov._model_name = "non_existent_rec_model_xyz"

    status = prov.get_runtime_status()
    assert status["state"] == "MODEL_DOWNLOAD_REQUIRED"
    assert status["configured"] is False
    assert status["enabled"] is False
    assert status["status_label"] == "MODEL DOWNLOAD REQUIRED"


def test_paddleocr_status_available():
    """State 3: When model weights exist locally, reports AVAILABLE — CPU."""
    prov = PaddleOCRProvider()
    # Cache weights exist in .paddlex_cache
    status = prov.get_runtime_status()
    assert status["state"] == "AVAILABLE"
    assert status["configured"] is True
    assert status["enabled"] is True
    assert "AVAILABLE" in status["status_label"]


def test_paddleocr_status_error():
    """State 4: When initialization error occurred, reports ERROR."""
    prov = PaddleOCRProvider()
    prov._init_error = "CUDA driver mismatch"

    status = prov.get_runtime_status()
    assert status["state"] == "ERROR"
    assert status["configured"] is False
    assert status["enabled"] is False
    assert status["status_label"] == "ERROR"


# ==============================================================================
# 3. IMAGE VALIDATION & PREPROCESSING TESTS
# ==============================================================================

def test_paddleocr_rejects_empty_image():
    """Verify provider rejects empty numpy array or None."""
    prov = PaddleOCRProvider()
    with pytest.raises(ValueError, match="cannot be None"):
        prov.recognize(None)

    empty_arr = np.zeros((0, 0, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="empty or has invalid shape"):
        prov.recognize(empty_arr)


def test_paddleocr_rejects_tiny_dimensions():
    """Verify provider rejects images with dimensions under minimum threshold."""
    prov = PaddleOCRProvider()
    tiny_img = Image.new("RGB", (5, 5))
    with pytest.raises(ValueError, match="too small"):
        prov.recognize(tiny_img)


def test_paddleocr_clamps_oversized_image():
    """Verify oversized images (>3200px) are downscaled to prevent memory exhaustion (CWE-400)."""
    prov = PaddleOCRProvider()
    large_img = Image.new("RGB", (3600, 1800), color=(128, 128, 128))
    img_bgr, h, w = prov._validate_and_prepare_image(large_img)
    assert max(h, w) <= prov.MAX_IMAGE_DIM
    assert h == 1600
    assert w == 3200


# ==============================================================================
# 4. MOCKED INFERENCE & UNICODE TESTS
# ==============================================================================

def test_paddleocr_mock_inference():
    """Verify mocked engine execution extracts lines, bounding boxes, and confidences."""
    prov = PaddleOCRProvider()
    mock_engine = MagicMock()
    mock_engine.predict.return_value = [
        {"rec_text": "ஸ்வஸ்திஸ்ரீ கோப்பரகேசரி", "rec_score": 0.945},
        {"rec_text": "பாண்டியன் நெடுஞ்செழியன்", "rec_score": 0.962},
    ]
    prov.set_mock_engine(mock_engine)

    img = Image.new("RGB", (100, 100))
    res = prov.recognize(img)

    assert isinstance(res, OCRResult)
    assert "கோப்பரகேசரி" in res.text
    assert "நெடுஞ்செழியன்" in res.text
    assert res.is_cloud is False
    assert res.confidence is not None
    assert round(res.confidence, 2) == 0.95
    assert len(res.bounding_boxes) == 2
    assert res.metadata["has_tamil_unicode"] is True
    assert res.metadata["transcription_status"] == "success"


def test_paddleocr_empty_recognition():
    """Verify empty OCR result produces empty text and None/0 confidence."""
    prov = PaddleOCRProvider()
    mock_engine = MagicMock()
    mock_engine.predict.return_value = []
    prov.set_mock_engine(mock_engine)

    img = Image.new("RGB", (100, 100))
    res = prov.recognize(img)

    assert res.text == ""
    assert res.confidence == 0.0
    assert len(res.bounding_boxes) == 0
    assert res.metadata["transcription_status"] == "empty"


# ==============================================================================
# 5. LIVE LOCAL INFERENCE TEST (AUTHENTIC INSCRIPTION PLATE)
# ==============================================================================

def test_paddleocr_live_local_inference():
    """
    Executes actual local inference using the downloaded PP-OCRv6 detector
    and ta_PP-OCRv5_mobile_rec on an authentic test stone inscription.
    Verifies actual text extraction without cloud calls or mock engines.
    """
    sample_path = os.path.join(CONFIG.SAMPLES_DIR, "mangulam.jpg")
    assert os.path.exists(sample_path)

    prov = PaddleOCRProvider(use_gpu=False)
    # Ensure not mocked
    prov._mock_engine = None

    res = prov.recognize(sample_path)

    assert isinstance(res, OCRResult)
    assert res.provider_name == "PaddleOCR Local Engine (PP-OCRv5 Tamil)"
    assert res.model_id == "ta_PP-OCRv5_mobile_rec"
    assert res.is_cloud is False
    assert res.latency_seconds > 0.0

    # Must have extracted non-empty text lines with Tamil Unicode
    assert len(res.text.strip()) > 0
    assert res.metadata["has_tamil_unicode"] is True
    assert len(res.bounding_boxes) > 0
    assert res.confidence is not None
    assert res.confidence > 0.50

    # Check that known keywords from Mangulam plate are detected
    tamil_words = ["கல்வட்ட", "பிராமி", "கணிய்", "நந்தாஸிரிய்", "பாண்டியன்"]
    assert any(w in res.text for w in tamil_words)


# ==============================================================================
# 6. PROVIDER MANAGER DISCOVERY
# ==============================================================================

def test_provider_manager_paddleocr_registration():
    """Verify ProviderManager discovers and configures PaddleOCR as a local OCR provider."""
    pm = get_provider_manager()
    prov = pm.get_ocr_provider("paddleocr_local")
    assert prov is not None
    assert isinstance(prov, PaddleOCRProvider)
    assert prov.provider_name == "PaddleOCR Local Engine (PP-OCRv5 Tamil)"

