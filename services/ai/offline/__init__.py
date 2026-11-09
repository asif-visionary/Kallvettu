"""Offline and Local AI Providers Package."""

from .offline_ocr_provider import LocalCatalogOCRProvider
from .paddleocr_provider import PaddleOCRProvider

__all__ = [
    "LocalCatalogOCRProvider",
    "PaddleOCRProvider",
]

