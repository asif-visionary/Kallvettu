"""
IIIF Exact Coordinate Transformation & Region Resolver (Phase D).

Guarantees:
- Original master image coordinates are never conflated with downsampled/resized display coordinates.
- Bidirectional scaling between original high-res matrix and preview/display viewports.
- Standard IIIF region URI syntax parsing (pixel coordinates, percentages, square, full).
- Standard IIIF region URL and deep-link generation.
"""

from __future__ import annotations

import re
from typing import Dict, Any, Tuple, Optional


def scale_box_to_original(
    display_box: Dict[str, Any],
    display_width: int,
    display_height: int,
    original_width: int,
    original_height: int,
) -> Dict[str, int]:
    """
    Transforms display/preview bounding box coordinates into exact original high-res coordinates.
    """
    if display_width <= 0 or display_height <= 0 or original_width <= 0 or original_height <= 0:
        return {"x": 0, "y": 0, "w": 0, "h": 0}

    sx = original_width / float(display_width)
    sy = original_height / float(display_height)

    ox = int(round(display_box.get("x", 0) * sx))
    oy = int(round(display_box.get("y", 0) * sy))
    ow = int(round(display_box.get("w", 0) * sx))
    oh = int(round(display_box.get("h", 0) * sy))

    # Clamp to original boundaries
    ox = max(0, min(ox, original_width))
    oy = max(0, min(oy, original_height))
    ow = max(1, min(ow, original_width - ox))
    oh = max(1, min(oh, original_height - oy))

    return {"x": ox, "y": oy, "w": ow, "h": oh}


def scale_box_to_display(
    original_box: Dict[str, Any],
    original_width: int,
    original_height: int,
    display_width: int,
    display_height: int,
) -> Dict[str, int]:
    """
    Transforms exact original high-res bounding box into preview/display coordinates.
    """
    if original_width <= 0 or original_height <= 0 or display_width <= 0 or display_height <= 0:
        return {"x": 0, "y": 0, "w": 0, "h": 0}

    sx = display_width / float(original_width)
    sy = display_height / float(original_height)

    dx = int(round(original_box.get("x", 0) * sx))
    dy = int(round(original_box.get("y", 0) * sy))
    dw = int(round(original_box.get("w", 0) * sx))
    dh = int(round(original_box.get("h", 0) * sy))

    dx = max(0, min(dx, display_width))
    dy = max(0, min(dy, display_height))
    dw = max(1, min(dw, display_width - dx))
    dh = max(1, min(dh, display_height - dy))

    return {"x": dx, "y": dy, "w": dw, "h": dh}


def parse_iiif_region(
    region_str: str,
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    """
    Parses IIIF Image API 3.0 {region} syntax into (x, y, w, h).
    Spec:
    - 'full': entire image
    - 'square': center square crop
    - 'x,y,w,h': pixel values
    - 'pct:x,y,w,h': percentage values (0..100)
    """
    r = (region_str or "full").strip().lower()

    if r == "full":
        return 0, 0, image_width, image_height

    if r == "square":
        side = min(image_width, image_height)
        x = (image_width - side) // 2
        y = (image_height - side) // 2
        return x, y, side, side

    if r.startswith("pct:"):
        parts = r[4:].split(",")
        if len(parts) == 4:
            try:
                px, py, pw, ph = [float(p) for p in parts]
                x = int(round((px / 100.0) * image_width))
                y = int(round((py / 100.0) * image_height))
                w = int(round((pw / 100.0) * image_width))
                h = int(round((ph / 100.0) * image_height))
                x = max(0, min(x, image_width - 1))
                y = max(0, min(y, image_height - 1))
                w = max(1, min(w, image_width - x))
                h = max(1, min(h, image_height - y))
                return x, y, w, h
            except ValueError:
                pass
        return 0, 0, image_width, image_height

    # Pixel syntax: x,y,w,h
    parts = r.split(",")
    if len(parts) == 4:
        try:
            x, y, w, h = [int(float(p)) for p in parts]
            x = max(0, min(x, image_width - 1))
            y = max(0, min(y, image_height - 1))
            w = max(1, min(w, image_width - x))
            h = max(1, min(h, image_height - y))
            return x, y, w, h
        except ValueError:
            pass

    return 0, 0, image_width, image_height


def format_iiif_region(
    x: int,
    y: int,
    w: int,
    h: int,
    pct: bool = False,
    image_width: int = 0,
    image_height: int = 0,
) -> str:
    """
    Formats bounding box into standard IIIF region syntax.
    - Pixel format: '{x},{y},{w},{h}'
    - Percentage format: 'pct:{px},{py},{pw},{ph}'
    """
    if pct and image_width > 0 and image_height > 0:
        px = max(0.0, min(100.0, (float(x) / float(image_width)) * 100.0))
        py = max(0.0, min(100.0, (float(y) / float(image_height)) * 100.0))
        pw = max(0.1, min(100.0 - px, (float(w) / float(image_width)) * 100.0))
        ph = max(0.1, min(100.0 - py, (float(h) / float(image_height)) * 100.0))
        return f"pct:{px:.1f},{py:.1f},{pw:.1f},{ph:.1f}"
    return f"{int(round(x))},{int(round(y))},{int(round(w))},{int(round(h))}"


def build_iiif_region_url(
    base_uri: str,
    asset_id: str,
    x: int,
    y: int,
    w: int,
    h: int,
    pct: bool = False,
    image_width: int = 0,
    image_height: int = 0,
    size: str = "max",
    rotation: int = 0,
    quality: str = "default",
    format_ext: str = "jpg",
) -> str:
    """
    Constructs an absolute, compliant IIIF Image API 3.0 Region extraction URL.
    Spec: {scheme}://{server}{/prefix}/{identifier}/{region}/{size}/{rotation}/{quality}.{format}
    """
    region_str = format_iiif_region(
        x=x, y=y, w=w, h=h, pct=pct, image_width=image_width, image_height=image_height
    )
    clean_base = base_uri.rstrip("/")
    return f"{clean_base}/{asset_id}/{region_str}/{size}/{rotation}/{quality}.{format_ext}"


def build_deep_link_url(
    base_url: str,
    inscription_id: str,
    x: int,
    y: int,
    w: int,
    h: int,
    pct: bool = False,
    image_width: int = 0,
    image_height: int = 0,
) -> str:
    """
    Constructs a persistent deep-link URL restoring the OpenSeadragon viewport
    directly to the specified inscription and region.
    """
    region_str = format_iiif_region(
        x=x, y=y, w=w, h=h, pct=pct, image_width=image_width, image_height=image_height
    )
    clean_base = base_url.rstrip("/")
    return f"{clean_base}/?studio=iiif&inscription_id={inscription_id}&region={region_str}"
