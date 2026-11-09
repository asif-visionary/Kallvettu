"""
Response Validator & Content Sanitation Utility for Kallvettu AI Providers.
Validates Tamil Unicode integrity, normalizes bounding coordinates, and cleans artifacts.
"""

import re
from typing import Optional, Union, Dict, Any, List, Tuple
from .base_provider import BoundingBox

# Tamil Unicode code point range: U+0B80 to U+0BFF
TAMIL_UNICODE_PATTERN = re.compile(r"[\u0B80-\u0BFF]")


def validate_tamil_unicode(text: Optional[str]) -> bool:
    """Returns True if the string contains at least one valid Tamil Unicode character."""
    if not text or not isinstance(text, str):
        return False
    return bool(TAMIL_UNICODE_PATTERN.search(text))


def calculate_tamil_character_ratio(text: Optional[str]) -> float:
    """
    Calculates the proportion of Tamil Unicode characters relative to all non-whitespace characters.
    Useful for filtering false positive ASCII recognition or OCR hallucination.
    """
    if not text or not isinstance(text, str):
        return 0.0

    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0

    tamil_count = sum(1 for c in chars if '\u0B80' <= c <= '\u0BFF')
    return tamil_count / len(chars)


def clamp_confidence(score: Optional[Union[float, int, str]]) -> Optional[float]:
    """Clamps an arbitrary raw confidence score to the standard range [0.0, 1.0]."""
    if score is None:
        return None
    try:
        val = float(score)
        # Handle percentage (e.g. 85.0 -> 0.85)
        if val >= 5.0 and val <= 100.0:
            val = val / 100.0
        return max(0.0, min(1.0, val))
    except (ValueError, TypeError):
        return None


def normalize_bounding_box(
    raw_coords: Union[List[float], Tuple[float, ...], Dict[str, Any]],
    image_width: int,
    image_height: int,
    confidence: Optional[float] = None,
    text: Optional[str] = None
) -> Optional[BoundingBox]:
    """
    Converts various provider bounding box formats (absolute pixels, polygons, 4-tuples)
    into a standardized normalized BoundingBox [ymin, xmin, ymax, xmax] in [0.0, 1.0].
    """
    if not raw_coords or image_width <= 0 or image_height <= 0:
        return None

    try:
        ymin, xmin, ymax, xmax = 0.0, 0.0, 0.0, 0.0

        if isinstance(raw_coords, dict):
            # Formats: {"ymin": ..., "xmin": ...} or {"y": ..., "x": ..., "width": ..., "height": ...}
            if "ymin" in raw_coords:
                ymin = float(raw_coords["ymin"])
                xmin = float(raw_coords["xmin"])
                ymax = float(raw_coords["ymax"])
                xmax = float(raw_coords["xmax"])
            elif "top" in raw_coords:
                ymin = float(raw_coords["top"])
                xmin = float(raw_coords["left"])
                ymax = ymin + float(raw_coords.get("height", 0))
                xmax = xmin + float(raw_coords.get("width", 0))
        elif isinstance(raw_coords, (list, tuple)):
            if raw_coords and all(isinstance(pt, (list, tuple)) and len(pt) >= 2 for pt in raw_coords):
                # Polygon or quadrilateral [[x1, y1], [x2, y2], [x3, y3], [x4, y4], ...]
                xs = [float(pt[0]) for pt in raw_coords]
                ys = [float(pt[1]) for pt in raw_coords]
                if xs and ys:
                    xmin = min(xs) / image_width if max(xs) > 1.0 else min(xs)
                    xmax = max(xs) / image_width if max(xs) > 1.0 else max(xs)
                    ymin = min(ys) / image_height if max(ys) > 1.0 else min(ys)
                    ymax = max(ys) / image_height if max(ys) > 1.0 else max(ys)
            elif len(raw_coords) == 4:
                # Flat list of 4 numbers [ymin, xmin, ymax, xmax] or [xmin, ymin, xmax, ymax]
                c0, c1, c2, c3 = [float(c) for c in raw_coords]
                # If values are > 1.0, treat as absolute pixels
                if c0 > 1.0 or c1 > 1.0 or c2 > 1.0 or c3 > 1.0:
                    ymin = min(c0, c2) / image_height
                    ymax = max(c0, c2) / image_height
                    xmin = min(c1, c3) / image_width
                    xmax = max(c1, c3) / image_width
                else:
                    ymin, xmin, ymax, xmax = c0, c1, c2, c3

        # Final bounds clamp
        ymin = max(0.0, min(1.0, ymin))
        xmin = max(0.0, min(1.0, xmin))
        ymax = max(0.0, min(1.0, ymax))
        xmax = max(0.0, min(1.0, xmax))

        return BoundingBox(
            ymin=ymin,
            xmin=xmin,
            ymax=ymax,
            xmax=xmax,
            confidence=clamp_confidence(confidence),
            text=text.strip() if text else None
        )
    except Exception:
        return None


def is_visual_description(text: Optional[str]) -> bool:
    """
    Detects whether text is an English/prose visual description of the inscription
    rather than an authentic inscriptional transcription.
    """
    if not text or not isinstance(text, str):
        return False

    t_clean = text.strip().lower()
    has_tamil = validate_tamil_unicode(text)

    desc_phrases = [
        "image shows", "image depicts", "photograph shows", "photo shows",
        "this image", "this picture", "picture shows", "visual description",
        "stone inscription photograph", "weathered stone slab", "carving on stone",
        "carved on the rock", "ancient stone", "grey stone", "gray stone",
        "close-up of", "a photo of", "a view of", "an image of",
        "shows an ancient", "shows a stone", "depicts an ancient",
        "generic visual description", "stone inscription with",
        "rock-cut inscription", "ancient inscription on",
    ]

    for phrase in desc_phrases:
        if phrase in t_clean:
            if not has_tamil or calculate_tamil_character_ratio(text) < 0.20:
                return True

    # If the text has no Tamil characters and contains archaeological image description terminology
    if not has_tamil and len(t_clean.split()) >= 4:
        arch_words = {
            "stone", "inscription", "carving", "rock", "slab",
            "weathered", "ancient", "carved", "surface", "texture", "granite"
        }
        words_in_text = set(re.findall(r"\b\w+\b", t_clean))
        if len(words_in_text & arch_words) >= 2:
            return True

    return False


def clean_text_artifacts(raw_text: Optional[str]) -> str:
    """Strips markdown code fences, null characters, and unwanted conversational wraps."""
    if not raw_text or not isinstance(raw_text, str):
        return ""

    cleaned = raw_text.strip()

    # Remove null bytes and non-printable control chars first
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned).strip()

    # Remove markdown code fences
    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # Normalize multiple newlines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()
