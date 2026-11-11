"""
Source Identity & Epigraphical Provenance Tracking for Kallvettu.

Guarantees data isolation across:
1. Source image identification and cryptographic SHA-256 integrity.
2. Processing session IDs and distinct OCR execution runs.
3. Separation of curated preset catalog ground truth from live OCR predictions.
4. Prevention of invalid comparisons across mismatched image sources.
"""

import hashlib
import io
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Union, Dict, Any, Tuple
import numpy as np
from PIL import Image


class ProvenanceMode(str, Enum):
    """Provenance origin of an inscriptional transcription."""
    CURATED_PRESET = "curated_preset"      # Verified catalog ground-truth (e.g., Mahadevan 2003 / EI)
    LIVE_OCR = "live_ocr"                  # Raw computational prediction from vision/OCR model
    MANUAL_RESEARCHER = "manual_researcher" # Scholar-edited diplomatic revision


def compute_image_sha256(image: Union[bytes, Image.Image, np.ndarray, str]) -> str:
    """
    Computes deterministic SHA-256 hex digest for an image input.
    Operates on raw byte stream or normalized pixel bytes.
    """
    if isinstance(image, bytes):
        return hashlib.sha256(image).hexdigest()

    if isinstance(image, str):
        # File path
        hasher = hashlib.sha256()
        with open(image, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    if isinstance(image, Image.Image):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return hashlib.sha256(buf.getvalue()).hexdigest()

    if isinstance(image, np.ndarray):
        # Continuous C-order bytes
        return hashlib.sha256(image.tobytes()).hexdigest()

    raise ValueError(f"Unsupported image type for SHA-256 calculation: {type(image)}")


@dataclass
class SourceIdentity:
    """
    Cryptographic and metadata identity tracking for an inscription specimen.
    """
    source_image_id: str
    image_sha256: str
    original_filename: str
    session_id: str
    preset_id: Optional[str] = None
    ocr_execution_id: Optional[str] = None
    provenance_mode: str = ProvenanceMode.LIVE_OCR.value
    dimensions: Optional[Tuple[int, int]] = None
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def filename(self) -> str:
        """Alias for original_filename."""
        return self.original_filename

    def is_curated_preset(self) -> bool:
        """Returns True if the transcription originates from a curated catalog reference."""
        return self.provenance_mode == ProvenanceMode.CURATED_PRESET.value

    def is_live_ocr(self) -> bool:
        """Returns True if the transcription is a live OCR model output."""
        return self.provenance_mode == ProvenanceMode.LIVE_OCR.value

    def is_comparable_with(self, other: "SourceIdentity") -> bool:
        """
        Validates that two source identities originate from the exact same source image.
        Rejects cross-image comparisons that would produce invalid palaeographical metrics.
        """
        if not isinstance(other, SourceIdentity):
            return False
        return self.image_sha256 == other.image_sha256

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_image_id": self.source_image_id,
            "image_sha256": self.image_sha256,
            "original_filename": self.original_filename,
            "session_id": self.session_id,
            "preset_id": self.preset_id,
            "ocr_execution_id": self.ocr_execution_id,
            "provenance_mode": self.provenance_mode,
            "dimensions": list(self.dimensions) if self.dimensions else None,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceIdentity":
        dims = tuple(data["dimensions"]) if data.get("dimensions") else None
        return cls(
            source_image_id=data["source_image_id"],
            image_sha256=data["image_sha256"],
            original_filename=data["original_filename"],
            session_id=data["session_id"],
            preset_id=data.get("preset_id"),
            ocr_execution_id=data.get("ocr_execution_id"),
            provenance_mode=data.get("provenance_mode", ProvenanceMode.LIVE_OCR.value),
            dimensions=dims,
            created_at=data.get("created_at", time.time()),
            metadata=data.get("metadata", {}),
        )


def create_source_identity(
    image: Union[bytes, Image.Image, np.ndarray, str],
    filename: str = "specimen.png",
    session_id: Optional[str] = None,
    preset_id: Optional[str] = None,
    provenance_mode: Optional[Union[ProvenanceMode, str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> SourceIdentity:
    """
    Factory creating a SourceIdentity with computed hash and unique IDs.
    """
    img_hash = compute_image_sha256(image)
    source_id = f"SRC_{img_hash[:12].upper()}"
    active_session = session_id or str(uuid.uuid4())
    execution_id = f"OCR_{uuid.uuid4().hex[:8].upper()}"

    dims = None
    if isinstance(image, Image.Image):
        dims = (image.width, image.height)
    elif isinstance(image, np.ndarray) and len(image.shape) >= 2:
        dims = (int(image.shape[1]), int(image.shape[0]))

    if provenance_mode is None:
        p_mode = ProvenanceMode.CURATED_PRESET.value if preset_id else ProvenanceMode.LIVE_OCR.value
    elif isinstance(provenance_mode, ProvenanceMode):
        p_mode = provenance_mode.value
    else:
        p_mode = str(provenance_mode)

    return SourceIdentity(
        source_image_id=source_id,
        image_sha256=img_hash,
        original_filename=filename,
        session_id=active_session,
        preset_id=preset_id,
        ocr_execution_id=execution_id,
        provenance_mode=p_mode,
        dimensions=dims,
        metadata=metadata or {},
    )

