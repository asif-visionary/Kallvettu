"""
Sarvam Vision Document AI Provider for Kallvettu.
Implements BaseOCRProvider using official SDK client.doc_ai.parse() lifecycle:
1. Submit Parse Job with image payload (JPEG/PNG)
2. Asynchronous status polling (get_status)
3. Fetch results (get_results / get_download_url)
4. Extraction of recognized text, bounding boxes, and layout structures
"""

import io
import time
import logging
import zipfile
import json
from html.parser import HTMLParser
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import requests
from PIL import Image

from services.ai.common.base_provider import (
    BaseOCRProvider,
    ProviderType,
    OCRResult,
    BoundingBox,
)
from services.ai.common.error_handler import (
    ProviderNotConfiguredError,
    AuthenticationError,
    RateLimitExceededError,
    ServiceUnavailableError,
    ResponseValidationError,
    KallvettuAIError,
)
from services.ai.common.response_validator import (
    clean_text_artifacts,
    clamp_confidence,
    normalize_bounding_box,
    validate_tamil_unicode,
    calculate_tamil_character_ratio,
    is_visual_description,
)
from .client import SarvamClient, get_sarvam_client

logger = logging.getLogger("kallvettu.sarvam.vision")

# Terminal status states for Sarvam Document AI
TERMINAL_STATUSES = {"completed", "partially_completed", "failed", "rejected"}


def prepare_image_bytes(image: Union[bytes, Image.Image, np.ndarray, str]) -> Tuple[bytes, str]:
    """
    Normalizes any image input (bytes, PIL Image, numpy array, or file path)
    into a JPEG byte buffer and filename.
    """
    if isinstance(image, str):
        with open(image, "rb") as f:
            return f.read(), "inscription.jpg"
    elif isinstance(image, (bytes, bytearray)):
        return bytes(image), "inscription.jpg"
    elif isinstance(image, Image.Image):
        buf = io.BytesIO()
        rgb_img = image.convert("RGB")
        rgb_img.save(buf, format="JPEG", quality=95)
        return buf.getvalue(), "inscription.jpg"
    elif isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image.astype(np.uint8))
        buf = io.BytesIO()
        rgb_img = pil_img.convert("RGB")
        rgb_img.save(buf, format="JPEG", quality=95)
        return buf.getvalue(), "inscription.jpg"
    elif hasattr(image, "read"):
        raw = image.read()
        if hasattr(image, "seek"):
            image.seek(0)
        return raw, "inscription.jpg"
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")


class SimpleHTMLTextExtractor(HTMLParser):
    """
    Extracts text tokens from HTML body, cleanly skipping scripts, styles, and headers.
    """
    def __init__(self):
        super().__init__()
        self.texts: List[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag.lower() in ("script", "style", "head", "meta", "link", "title"):
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in ("script", "style", "head", "meta", "link", "title"):
            self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip:
            cleaned = data.strip()
            if cleaned:
                self.texts.append(cleaned)


def extract_content_from_zip(zip_bytes: bytes) -> Tuple[str, List[BoundingBox], Dict[str, Any]]:
    """
    Safely unpacks the Sarvam Document AI ZIP archive in memory, extracts
    the transcribed JSON/markdown/html content, structured layout tags, and bounding boxes.
    """
    extracted_text = ""
    bounding_boxes: List[BoundingBox] = []
    metadata_summary: Dict[str, Any] = {"tags": []}
    discovered_tags = set()
    block_texts: List[str] = []

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            file_names = zf.namelist()
            metadata_summary["zip_files"] = file_names

            # 1. Search for JSON outputs (when output_format="json")
            json_files = [f for f in file_names if f.endswith(".json")]
            for jf in json_files:
                try:
                    raw_data = zf.read(jf).decode("utf-8", errors="replace")
                    data = json.loads(raw_data)
                    metadata_summary["format"] = "json"

                    def process_blocks(blocks_list, page_w, page_h):
                        for b in blocks_list:
                            if not isinstance(b, dict):
                                continue
                            tag = b.get("tag") or b.get("type") or b.get("label")
                            if tag:
                                discovered_tags.add(str(tag))
                            txt = b.get("text") or b.get("content") or ""
                            if isinstance(txt, str) and txt.strip():
                                block_texts.append(txt.strip())
                            bbox = b.get("bbox") or b.get("polygon") or b.get("box") or b.get("coordinates")
                            conf = b.get("confidence")
                            if conf is not None:
                                try:
                                    conf = float(conf)
                                except (ValueError, TypeError):
                                    conf = None
                            if bbox:
                                norm_box = normalize_bounding_box(bbox, page_w, page_h, confidence=conf, text=txt if isinstance(txt, str) else None)
                                if norm_box is not None:
                                    bounding_boxes.append(norm_box)

                    if isinstance(data, dict):
                        if isinstance(data.get("content"), str) and data["content"].strip():
                            if not extracted_text:
                                extracted_text = data["content"].strip()
                        elif isinstance(data.get("text"), str) and data["text"].strip():
                            if not extracted_text:
                                extracted_text = data["text"].strip()
                        elif isinstance(data.get("markdown"), str) and data["markdown"].strip():
                            if not extracted_text:
                                extracted_text = data["markdown"].strip()

                        img_w = float(data.get("image_width") or data.get("width") or 1000)
                        img_h = float(data.get("image_height") or data.get("height") or 1000)

                        if "blocks" in data and isinstance(data["blocks"], list):
                            process_blocks(data["blocks"], img_w, img_h)

                        if "pages" in data and isinstance(data["pages"], list):
                            for page in data["pages"]:
                                if isinstance(page, dict):
                                    pw = float(page.get("page_width") or page.get("width") or img_w)
                                    ph = float(page.get("page_height") or page.get("height") or img_h)
                                    if "blocks" in page and isinstance(page["blocks"], list):
                                        process_blocks(page["blocks"], pw, ph)
                except Exception as json_err:
                    logger.debug("Could not parse json '%s': %s", jf, json_err)

            # If JSON parsed block texts and no single content string was set:
            if not extracted_text and block_texts:
                extracted_text = "\n\n".join(block_texts).strip()

            # 2. Look for Markdown or HTML outputs if JSON didn't yield text
            if not extracted_text:
                md_files = [f for f in file_names if f.endswith((".md", ".txt"))]
                html_files = [f for f in file_names if f.endswith((".html", ".htm"))]

                if md_files:
                    raw_md = zf.read(md_files[0]).decode("utf-8", errors="replace")
                    extracted_text = raw_md.strip()
                    metadata_summary["format"] = "markdown"
                elif html_files:
                    raw_html = zf.read(html_files[0]).decode("utf-8", errors="replace")
                    parser = SimpleHTMLTextExtractor()
                    parser.feed(raw_html)
                    extracted_text = "\n".join(parser.texts).strip()
                    metadata_summary["format"] = "html"

    except Exception as zerr:
        logger.warning("Error unpacking Sarvam zip archive: %s", zerr)
        metadata_summary["zip_error"] = str(zerr)

    metadata_summary["tags"] = sorted(list(discovered_tags))
    metadata_summary["blocks_count"] = len(block_texts) if block_texts else len(bounding_boxes)
    return extracted_text, bounding_boxes, metadata_summary


class SarvamVisionProvider(BaseOCRProvider):
    """
    Document Intelligence & Vision OCR provider powered by Sarvam AI (Sarvam Vision 3B).
    """

    DEFAULT_LANGUAGE = "ta-IN"
    POLL_INTERVAL_SECONDS = 2.0
    DEFAULT_TIMEOUT_SECONDS = 60.0

    def __init__(self, client: Optional[SarvamClient] = None):
        self._client = client or get_sarvam_client()

    @property
    def provider_name(self) -> str:
        return "Sarvam Vision (Document AI)"

    @property
    def provider_id(self) -> str:
        return "sarvam_vision"

    def is_configured(self) -> bool:
        return self._client.is_configured()

    def check_health(self) -> Tuple[bool, str]:
        return self._client.check_health()

    def recognize(
        self,
        image: Union[bytes, Image.Image, np.ndarray, str],
        script_hint: Optional[str] = None,
        safe: bool = False,
        **kwargs: Any
    ) -> OCRResult:
        """
        Executes cloud document intelligence OCR on the stone inscription photograph.
        Implements async submit -> poll -> fetch lifecycle.
        If safe=True, returns an OCRResult with transcription_status='api_failure' on errors.
        """
        if safe:
            try:
                return self._recognize_impl(image, script_hint=script_hint, **kwargs)
            except Exception as e:
                logger.error("Sarvam Vision API execution failed: %s", e)
                return OCRResult(
                    text="",
                    provider_name=self.provider_name,
                    model_id="sarvam-vision",
                    confidence=0.0,
                    latency_seconds=0.0,
                    bounding_boxes=[],
                    raw_response={"error": str(e)},
                    is_cloud=True,
                    is_historical_script=False,
                    provenance_notes="Sarvam Document AI execution failure",
                    metadata={"transcription_status": "api_failure", "error": str(e)}
                )
        return self._recognize_impl(image, script_hint=script_hint, **kwargs)

    def _recognize_impl(
        self,
        image: Union[bytes, Image.Image, np.ndarray, str],
        script_hint: Optional[str] = None,
        **kwargs: Any
    ) -> OCRResult:
        if not self.is_configured():
            raise ProviderNotConfiguredError(
                "Sarvam AI is not configured. Supply SARVAM_API_KEY in environment or Settings.",
                provider=self.provider_name
            )

        t0 = time.perf_counter()
        img_bytes, filename = prepare_image_bytes(image)

        # Inspect image dimensions for bounding box normalization
        img_w, img_h = 1000, 1000
        try:
            with Image.open(io.BytesIO(img_bytes)) as pil_test:
                img_w, img_h = pil_test.width, pil_test.height
        except Exception:
            pass

        sdk = self._client.get_sdk_client()
        lang = kwargs.get("language", self.DEFAULT_LANGUAGE)
        timeout = float(kwargs.get("timeout", self.DEFAULT_TIMEOUT_SECONDS))

        # 1. Submit Document AI Parse Job
        try:
            file_tuple = [(filename, img_bytes, "image/jpeg")]
            job_resp = sdk.doc_ai.parse(
                file=file_tuple,
                language=lang,
                request_options={"additional_body_parameters": {"output_format": "json"}},
            )
            job_id = getattr(job_resp, "job_id", None) or getattr(job_resp, "id", None)
            if not job_id and isinstance(job_resp, dict):
                job_id = job_resp.get("job_id") or job_resp.get("id")

            if not job_id:
                raise ResponseValidationError(
                    "Sarvam doc_ai.parse did not return a valid job_id.",
                    provider=self.provider_name
                )
        except Exception as e:
            raw_err = str(e)
            if "401" in raw_err or "unauthorized" in raw_err.lower():
                raise AuthenticationError(
                    "Sarvam API key is unauthorized or invalid for Document AI.",
                    provider=self.provider_name,
                    raw_error=e
                ) from e
            elif "429" in raw_err or "rate limit" in raw_err.lower():
                raise RateLimitExceededError(
                    "Sarvam Document AI rate limit exceeded.",
                    provider=self.provider_name,
                    raw_error=e
                ) from e
            else:
                raise KallvettuAIError(
                    f"Failed to submit Sarvam Document AI parse job: {raw_err}",
                    provider=self.provider_name,
                    raw_error=e
                ) from e

        # 2. Async Status Polling Loop
        poll_start = time.perf_counter()
        final_status = "unknown"

        while True:
            if time.perf_counter() - poll_start > timeout:
                raise ServiceUnavailableError(
                    f"Sarvam Document AI job {job_id} timed out after {timeout}s.",
                    provider=self.provider_name
                )

            try:
                status_resp = sdk.doc_ai.get_status(job_id=job_id)
                current_status = getattr(status_resp, "status", "unknown").lower()
                final_status = current_status

                if current_status in TERMINAL_STATUSES:
                    break
            except Exception as e:
                logger.warning("Sarvam doc_ai polling error: %s", e)

            time.sleep(self.POLL_INTERVAL_SECONDS)

        if final_status in {"failed", "rejected"}:
            raise ResponseValidationError(
                f"Sarvam Document AI job {job_id} terminated with status: {final_status}",
                provider=self.provider_name
            )

        # 3. Retrieve Parsed Results
        extracted_text = ""
        bounding_boxes: List[BoundingBox] = []
        raw_result_data: Dict[str, Any] = {}
        dl_url = ""

        # Check get_results first (useful in mocked tests or direct endpoint updates)
        try:
            results_obj = sdk.doc_ai.get_results(job_id=job_id, format="json")
            res_content = getattr(results_obj, "content", None)
            res_md = getattr(results_obj, "markdown", None)
            res_txt = getattr(results_obj, "text", None)

            if isinstance(res_content, str) and res_content.strip():
                extracted_text = res_content
            elif isinstance(res_md, str) and res_md.strip():
                extracted_text = res_md
            elif isinstance(res_txt, str) and res_txt.strip():
                extracted_text = res_txt
            elif isinstance(results_obj, dict):
                extracted_text = (
                    results_obj.get("content")
                    or results_obj.get("markdown")
                    or results_obj.get("text", "")
                )
            if extracted_text:
                raw_result_data["results"] = str(results_obj)[:500]
        except Exception as e:
            logger.debug("Sarvam get_results skipped or failed (%s), fetching via get_download_url", e)

        # Download ZIP archive if direct text was not returned
        if not extracted_text.strip():
            try:
                dl = sdk.doc_ai.get_download_url(job_id=job_id)
                dl_url = getattr(dl, "url", "")
                if not dl_url and isinstance(dl, dict):
                    dl_url = dl.get("url", "")
                raw_result_data["download_url"] = dl_url

                if dl_url:
                    # Fetch zip directly without sending custom headers that conflict with Azure Blob SAS
                    resp = requests.get(dl_url, timeout=30)
                    if resp.status_code == 200:
                        zip_text, zip_boxes, zip_meta = extract_content_from_zip(resp.content)
                        if zip_text.strip():
                            extracted_text = zip_text
                        if zip_boxes:
                            bounding_boxes.extend(zip_boxes)
                        raw_result_data.update(zip_meta)
                    else:
                        logger.warning("Sarvam ZIP download returned status %s", resp.status_code)
                        raw_result_data["download_http_status"] = resp.status_code
            except Exception as e2:
                logger.warning("Sarvam download url retrieval or zip parsing failed: %s", e2)
                raw_result_data["download_error"] = str(e2)

        cleaned_text = clean_text_artifacts(extracted_text)
        latency = time.perf_counter() - t0

        # Check classification: description_only vs partial vs success vs empty
        is_desc = is_visual_description(cleaned_text)
        has_tamil = validate_tamil_unicode(cleaned_text)
        tamil_ratio = calculate_tamil_character_ratio(cleaned_text) if cleaned_text else 0.0

        visual_desc_content: Optional[str] = None
        transcription_status = "empty"

        if not cleaned_text:
            transcription_status = "empty"
        elif is_desc:
            transcription_status = "description_only"
            visual_desc_content = cleaned_text
            # Never treat generic image descriptions as authentic inscription text
            cleaned_text = ""
            raw_result_data["visual_description"] = visual_desc_content
        elif has_tamil and tamil_ratio < 0.60:
            # Contains both Tamil glyphs and surrounding English narrative/descriptions
            transcription_status = "partial"
            tamil_tokens = [tok for tok in cleaned_text.split() if validate_tamil_unicode(tok)]
            raw_result_data["raw_mixed_content"] = cleaned_text
            cleaned_text = " ".join(tamil_tokens).strip()
        elif has_tamil:
            transcription_status = "success"
        else:
            # Non-empty but non-Tamil text
            transcription_status = "partial"

        # Extract true provider confidence if available in bounding boxes or metadata (NEVER hardcode)
        box_confs = [b.confidence for b in bounding_boxes if b.confidence is not None]
        if box_confs:
            confidence = float(np.mean(box_confs))
        elif isinstance(raw_result_data.get("results"), dict) and "confidence" in raw_result_data["results"]:
            confidence = clamp_confidence(raw_result_data["results"]["confidence"])
        else:
            # Genuine null confidence when provider does not supply confidence score
            confidence = None

        has_tamil = validate_tamil_unicode(cleaned_text)

        metadata_dict = {
            "job_id": job_id,
            "terminal_status": final_status,
            "language": lang,
            "download_url": dl_url,
            "has_tamil_unicode": has_tamil,
            "detected_text_length": len(cleaned_text),
            "transcription_status": transcription_status,
            "bounding_boxes_count": len(bounding_boxes),
            "bounding_boxes_supported": len(bounding_boxes) > 0,
            "tags": raw_result_data.get("tags", []),
            "script_limitation_note": (
                "Sarvam Vision is trained on modern Indic document corpora. "
                "Lithic character readings for Tamil-Brahmi or Vatteluttu should be verified by an epigraphist."
            ),
        }
        if visual_desc_content:
            metadata_dict["visual_description"] = visual_desc_content
            metadata_dict["warning"] = "Sarvam Vision returned only a visual description of the inscription image, not an inscription transcription."

        return OCRResult(
            text=cleaned_text,
            provider_name=self.provider_name,
            model_id="sarvam-vision",
            confidence=confidence,
            latency_seconds=latency,
            bounding_boxes=bounding_boxes,
            raw_response=raw_result_data,
            is_cloud=True,
            is_historical_script=False,
            provenance_notes="Sarvam Document AI (Modern Indic OCR Model)",
            metadata=metadata_dict
        )

