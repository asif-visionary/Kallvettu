"""
Phase 3 Unit Tests: Sarvam Vision Document AI Integration.
Covers:
- Image byte serialization from PIL, NumPy array, and raw bytes
- Document AI parse job lifecycle (submit -> poll -> get_results)
- Terminal status handling (completed, failed, timeout)
- Historical script limitation metadata retention
- ProviderManager discovery and registration
"""

import os
import sys
import pytest
import numpy as np
from PIL import Image
from unittest.mock import MagicMock

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.ai.sarvam.client import SarvamClient
from services.ai.sarvam.vision_provider import (
    SarvamVisionProvider,
    prepare_image_bytes,
    extract_content_from_zip,
)
from services.ai.common.base_provider import OCRResult
from services.ai.common.error_handler import (
    ProviderNotConfiguredError,
    ResponseValidationError,
    ServiceUnavailableError,
)
from services.ai.common.provider_manager import get_provider_manager


# ==============================================================================
# 1. IMAGE SERIALIZATION TESTS
# ==============================================================================

def test_prepare_image_bytes_pil():
    """Verify PIL Image converts to valid JPEG bytes."""
    img = Image.new("RGB", (100, 100), color=(200, 180, 140))
    raw, name = prepare_image_bytes(img)
    assert isinstance(raw, bytes)
    assert len(raw) > 0
    assert name.endswith(".jpg")


def test_prepare_image_bytes_numpy():
    """Verify NumPy array converts to JPEG bytes."""
    arr = np.random.randint(0, 255, size=(80, 80, 3), dtype=np.uint8)
    raw, name = prepare_image_bytes(arr)
    assert isinstance(raw, bytes)
    assert len(raw) > 0


def test_prepare_image_bytes_raw():
    """Verify raw bytes passthrough."""
    dummy_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF"
    raw, name = prepare_image_bytes(dummy_bytes)
    assert raw == dummy_bytes


# ==============================================================================
# 2. DOCUMENT AI LIFECYCLE TESTS (MOCKED)
# ==============================================================================

def test_sarvam_vision_unconfigured():
    """Verify unconfigured client raises ProviderNotConfiguredError."""
    client = SarvamClient(api_key="")
    provider = SarvamVisionProvider(client=client)

    dummy_img = Image.new("RGB", (50, 50))
    with pytest.raises(ProviderNotConfiguredError):
        provider.recognize(dummy_img)


def test_sarvam_vision_parse_lifecycle_success():
    """Verify full submit -> poll -> get_results lifecycle returns structured OCRResult."""
    client = SarvamClient(api_key="sk_valid_mock_key_12345")
    mock_sdk = MagicMock()

    # 1. Mock parse submission
    mock_job_resp = MagicMock()
    mock_job_resp.job_id = "job_sarvam_parse_999"
    mock_sdk.doc_ai.parse.return_value = mock_job_resp

    # 2. Mock status polling (returns completed)
    mock_status_resp = MagicMock()
    mock_status_resp.status = "completed"
    mock_sdk.doc_ai.get_status.return_value = mock_status_resp

    # 3. Mock get_results
    mock_results = MagicMock()
    mock_results.content = "## கல்வெட்டு உரை\nஸ்வஸ்திஸ்ரீ கோப்பரகேசரி"
    mock_sdk.doc_ai.get_results.return_value = mock_results

    client.set_sdk_client(mock_sdk)
    provider = SarvamVisionProvider(client=client)

    dummy_img = Image.new("RGB", (100, 100), color=(150, 150, 150))
    res = provider.recognize(dummy_img, language="ta-IN")

    assert isinstance(res, OCRResult)
    assert "கோப்பரகேசரி" in res.text
    assert res.provider_name == "Sarvam Vision (Document AI)"
    assert res.model_id == "sarvam-vision"
    assert res.is_cloud is True
    # Scientific honesty check: Modern Indic model limitation
    assert res.is_historical_script is False
    assert "job_id" in res.metadata
    assert res.metadata["job_id"] == "job_sarvam_parse_999"
    assert "script_limitation_note" in res.metadata


def test_sarvam_vision_job_failed_status():
    """Verify job failure triggers ResponseValidationError."""
    client = SarvamClient(api_key="sk_valid_mock_key_12345")
    mock_sdk = MagicMock()

    mock_job_resp = MagicMock()
    mock_job_resp.job_id = "job_failed_001"
    mock_sdk.doc_ai.parse.return_value = mock_job_resp

    mock_status_resp = MagicMock()
    mock_status_resp.status = "failed"
    mock_sdk.doc_ai.get_status.return_value = mock_status_resp

    client.set_sdk_client(mock_sdk)
    provider = SarvamVisionProvider(client=client)

    dummy_img = Image.new("RGB", (50, 50))
    with pytest.raises(ResponseValidationError):
        provider.recognize(dummy_img)


def test_sarvam_vision_job_timeout():
    """Verify polling loop terminates and raises on timeout."""
    client = SarvamClient(api_key="sk_valid_mock_key_12345")
    mock_sdk = MagicMock()

    mock_job_resp = MagicMock()
    mock_job_resp.job_id = "job_hanging_002"
    mock_sdk.doc_ai.parse.return_value = mock_job_resp

    mock_status_resp = MagicMock()
    mock_status_resp.status = "processing"
    mock_sdk.doc_ai.get_status.return_value = mock_status_resp

    client.set_sdk_client(mock_sdk)
    provider = SarvamVisionProvider(client=client)
    # Configure tiny polling interval for rapid test
    provider.POLL_INTERVAL_SECONDS = 0.05

    dummy_img = Image.new("RGB", (50, 50))
    with pytest.raises(ServiceUnavailableError):
        provider.recognize(dummy_img, timeout=0.15)


def test_provider_manager_sarvam_vision_registration():
    """Verify ProviderManager discovers SarvamVisionProvider."""
    mgr = get_provider_manager()
    prov = mgr.get_ocr_provider("sarvam_vision")
    assert prov is not None
    assert isinstance(prov, SarvamVisionProvider)
    assert prov.provider_name == "Sarvam Vision (Document AI)"


# ==============================================================================
# 3. ZIP ARCHIVE PARSING & DOWNLOAD URL WORKFLOW TESTS
# ==============================================================================

def test_extract_content_from_zip_markdown():
    """Verify zip extractor handles markdown output format."""
    import io, zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("page.md", "# சாஸனம்\nகோப்பரகேசரிவர்மரான ராஜேந்திர சோழன்")
    
    text, boxes, meta = extract_content_from_zip(buf.getvalue())
    assert "ராஜேந்திர சோழன்" in text
    assert meta["format"] == "markdown"


def test_extract_content_from_zip_html():
    """Verify zip extractor parses html output skipping styles and scripts."""
    import io, zipfile
    buf = io.BytesIO()
    sample_html = """
    <html><head><style>body { color: red; }</style></head>
    <body><div class="page-body-container"><p>சுவஸ்திஸ்ரீ திருபுவன வீரதேவர்</p></div></body>
    </html>
    """
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("doc/page.html", sample_html)

    text, boxes, meta = extract_content_from_zip(buf.getvalue())
    assert "சுவஸ்திஸ்ரீ திருபுவன வீரதேவர்" in text
    assert "color: red" not in text
    assert meta["format"] == "html"


def test_extract_content_from_zip_empty_html():
    """Verify empty html from stone inscription yields empty text without error."""
    import io, zipfile
    buf = io.BytesIO()
    empty_html = """
    <html><head><style>body { font-family: serif; }</style></head>
    <body><div class="page-body-container"></div></body></html>
    """
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("inscription.jpg/inscription.html", empty_html)

    text, boxes, meta = extract_content_from_zip(buf.getvalue())
    assert text == ""
    assert len(boxes) == 0


def test_sarvam_vision_download_url_zip_workflow(monkeypatch):
    """Verify that get_download_url downloads zip and never pollutes text with URL."""
    import io, zipfile
    client = SarvamClient(api_key="sk_valid_mock_key_12345")
    mock_sdk = MagicMock()

    mock_job_resp = MagicMock()
    mock_job_resp.job_id = "job_dl_url_test_456"
    mock_sdk.doc_ai.parse.return_value = mock_job_resp

    mock_status_resp = MagicMock()
    mock_status_resp.status = "completed"
    mock_sdk.doc_ai.get_status.return_value = mock_status_resp

    # Make get_results fail (as it does in real Sarvam parse jobs due to ValidationError)
    mock_sdk.doc_ai.get_results.side_effect = Exception("ValidationError: type digitise not expected")

    # Mock get_download_url
    mock_dl = MagicMock()
    mock_dl.url = "https://azure-blob.example.com/inscription.zip?sas=token123"
    mock_sdk.doc_ai.get_download_url.return_value = mock_dl

    # Create mock zip content
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("inscription.jpg/page.md", "ஸ்வஸ்திஸ்ரீ மாங்குளம் தமிழ்ப் பிராமி")
    zip_bytes = buf.getvalue()

    # Mock requests.get
    class MockResp:
        status_code = 200
        content = zip_bytes

    import requests
    monkeypatch.setattr(requests, "get", lambda url, timeout: MockResp())

    client.set_sdk_client(mock_sdk)
    provider = SarvamVisionProvider(client=client)

    dummy_img = Image.new("RGB", (64, 64))
    res = provider.recognize(dummy_img)

    assert isinstance(res, OCRResult)
    # The actual text must be the inscription, NOT the download URL!
    assert "மாங்குளம்" in res.text
    assert "https://" not in res.text
    assert "Document parsed successfully" not in res.text
    # The download URL is preserved safely in metadata
    assert res.metadata["download_url"] == "https://azure-blob.example.com/inscription.zip?sas=token123"
    assert res.metadata["has_tamil_unicode"] is True


def test_sarvam_vision_visual_description_detection():
    """Verify that generic visual descriptions are detected and not treated as transcription."""
    client = SarvamClient(api_key="sk_valid_mock_key_12345")
    mock_sdk = MagicMock()

    mock_job_resp = MagicMock()
    mock_job_resp.job_id = "job_desc_only_101"
    mock_sdk.doc_ai.parse.return_value = mock_job_resp

    mock_status_resp = MagicMock()
    mock_status_resp.status = "completed"
    mock_sdk.doc_ai.get_status.return_value = mock_status_resp

    # Provider outputs a visual description in English instead of transcribing Tamil glyphs
    mock_results = MagicMock()
    mock_results.content = "This photograph shows an ancient weathered stone slab with carved characters on the rock surface."
    mock_sdk.doc_ai.get_results.return_value = mock_results

    client.set_sdk_client(mock_sdk)
    provider = SarvamVisionProvider(client=client)

    dummy_img = Image.new("RGB", (64, 64))
    res = provider.recognize(dummy_img)

    # Transcription text must be blanked out
    assert res.text == ""
    assert res.confidence is None
    assert res.metadata["transcription_status"] == "description_only"
    assert "weathered stone slab" in res.metadata["visual_description"]
    assert "warning" in res.metadata


def test_sarvam_vision_empty_inscription_zero_confidence(monkeypatch):
    """Verify that an inscription yielding no modern text returns empty text and 0 confidence."""
    import io, zipfile
    client = SarvamClient(api_key="sk_valid_mock_key_12345")
    mock_sdk = MagicMock()

    mock_job_resp = MagicMock()
    mock_job_resp.job_id = "job_empty_stone_789"
    mock_sdk.doc_ai.parse.return_value = mock_job_resp

    mock_status_resp = MagicMock()
    mock_status_resp.status = "completed"
    mock_sdk.doc_ai.get_status.return_value = mock_status_resp

    mock_sdk.doc_ai.get_results.side_effect = Exception("ValidationError")

    mock_dl = MagicMock()
    mock_dl.url = "https://azure-blob.example.com/empty.zip"
    mock_sdk.doc_ai.get_download_url.return_value = mock_dl

    # Create mock zip with empty HTML
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("inscription.jpg/inscription.html", "<div class='page-body-container'></div>")
    zip_bytes = buf.getvalue()

    class MockResp:
        status_code = 200
        content = zip_bytes

    import requests
    monkeypatch.setattr(requests, "get", lambda url, timeout: MockResp())

    client.set_sdk_client(mock_sdk)
    provider = SarvamVisionProvider(client=client)

    dummy_img = Image.new("RGB", (64, 64))
    res = provider.recognize(dummy_img)

    assert res.text == ""
    assert res.confidence is None or res.confidence == 0.0
    assert "download_url" in res.metadata


def test_extract_content_from_zip_json_blocks():
    """Verify zip extractor parses structured JSON with blocks, tags, and bounding boxes."""
    import io, json, zipfile
    buf = io.BytesIO()
    sample_json = {
        "image_width": 800,
        "image_height": 600,
        "blocks": [
            {
                "text": "ஸ்வஸ்திஸ்ரீ ராஜராஜ",
                "tag": "heading",
                "bbox": [10, 20, 80, 400],
                "confidence": 0.94
            },
            {
                "text": "தேவர்க்கு யாண்டு இருபத்தொன்றாவது",
                "tag": "paragraph",
                "bbox": [90, 20, 160, 500],
                "confidence": 0.88
            }
        ]
    }
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("document.json", json.dumps(sample_json))

    text, boxes, meta = extract_content_from_zip(buf.getvalue())
    assert "ராஜராஜ" in text
    assert "இருபத்தொன்றாவது" in text
    assert len(boxes) == 2
    assert meta["format"] == "json"
    assert "heading" in meta["tags"]
    assert "paragraph" in meta["tags"]
    assert meta["blocks_count"] == 2
    assert boxes[0].text == "ஸ்வஸ்திஸ்ரீ ராஜராஜ"


def test_sarvam_vision_api_failure_safe_mode():
    """Verify safe=True captures exceptions and returns OCRResult with transcription_status='api_failure'."""
    client = SarvamClient(api_key="sk_valid_mock_key_12345")
    mock_sdk = MagicMock()
    mock_sdk.doc_ai.parse.side_effect = Exception("503 Service Unavailable: Remote cluster unreachable")

    client.set_sdk_client(mock_sdk)
    provider = SarvamVisionProvider(client=client)

    dummy_img = Image.new("RGB", (64, 64))
    res = provider.recognize(dummy_img, safe=True)

    assert isinstance(res, OCRResult)
    assert res.text == ""
    assert res.metadata["transcription_status"] == "api_failure"
    assert "Remote cluster unreachable" in res.metadata["error"]


