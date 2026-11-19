"""
OpenSeadragon Deep-Zoom Inscription Viewer Component for Streamlit (Phase D).

Features:
- Pure client-side OpenSeadragon 4.1.1 viewer rendering via Streamlit Components.
- Pan, deep-zoom, rotation (90° CW / CCW), and fit-to-screen home controls.
- Dynamic Image API 3.0 tileSource loading with automatic offline fallback.
- Live zoom percentage and master pixel dimension badges.
- Side-by-side comparative inspection between raw master plate and restored CLAHE / Squeeze rubbing.
- Epigraphical character annotation overlays color-coded by legibility status:
  CLEAR (gold), WEATHERED (amber), PARTIAL (cyan), ILLEGIBLE (crimson).
- Interactive tooltip cards displaying Unicode label, diplomatic reading, and coordinates.
- Spatial region restoration for shared deep links.
- Loading indicator and graceful error handling on unavailable manifests.
"""

from __future__ import annotations

import base64
import io
import json
import os
from typing import Dict, Any, List, Optional, Union, Tuple
from PIL import Image
import streamlit as st


def image_to_base64_data_uri(image_input: Union[Image.Image, bytes, str]) -> str:
    """Converts a PIL Image, bytes, or file path to a base64 data URI."""
    if isinstance(image_input, Image.Image):
        buf = io.BytesIO()
        image_input.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    elif isinstance(image_input, bytes):
        b64 = base64.b64encode(image_input).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    elif isinstance(image_input, str):
        if image_input.startswith("data:image"):
            return image_input
        if os.path.isfile(image_input):
            with open(image_input, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
                return f"data:image/jpeg;base64,{b64}"
        return image_input
    return ""


def get_image_dimensions(image_input: Any) -> Optional[Tuple[int, int]]:
    """Helper to safely determine pixel dimensions (width, height)."""
    if isinstance(image_input, Image.Image):
        return image_input.size
    elif isinstance(image_input, str) and os.path.isfile(image_input):
        try:
            with Image.open(image_input) as im:
                return im.size
        except Exception:
            return None
    return None


def render_openseadragon_viewer(
    primary_image: Optional[Union[Image.Image, bytes, str]] = None,
    comparison_image: Optional[Union[Image.Image, bytes, str]] = None,
    annotations: Optional[List[Dict[str, Any]]] = None,
    viewer_height: int = 600,
    primary_label: str = "Master High-Resolution Photograph",
    comparison_label: str = "Enhanced / Squeeze Rubbing",
    tile_source: Optional[Union[str, Dict[str, Any]]] = None,
    comparison_tile_source: Optional[Union[str, Dict[str, Any]]] = None,
    initial_region: Optional[Dict[str, int]] = None,
    image_dimensions: Optional[Tuple[int, int]] = None,
    inscription_metadata: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> None:
    """
    Embeds an OpenSeadragon deep-zoom viewer into Streamlit.
    Supports single plate view or side-by-side synchronized comparison.
    """
    # Accept parameter aliases for backward compatibility
    if primary_image is None and "primary_image_uri" in kwargs:
        primary_image = kwargs["primary_image_uri"]
    if comparison_image is None and "comparison_image_uri" in kwargs:
        comparison_image = kwargs["comparison_image_uri"]
    if "height_px" in kwargs:
        viewer_height = kwargs["height_px"]
    if annotations is None and "glyph_boxes" in kwargs:
        adapted_anns = []
        for gb in kwargs["glyph_boxes"]:
            box = gb.get("box", (0, 0, 0, 0))
            if isinstance(box, (list, tuple)) and len(box) >= 4:
                adapted_anns.append({
                    "bounding_box": {"x": box[0], "y": box[1], "w": box[2], "h": box[3]},
                    "character_label": gb.get("label", ""),
                    "script_category": gb.get("script", ""),
                    "legibility_status": "CLEAR",
                })
        annotations = adapted_anns

    primary_uri = image_to_base64_data_uri(primary_image) if primary_image else ""
    comp_uri = image_to_base64_data_uri(comparison_image) if comparison_image else ""
    is_side_by_side = bool(comp_uri or comparison_tile_source)

    # Determine display dimensions
    dims = image_dimensions or get_image_dimensions(primary_image)
    dim_text = f"{dims[0]} × {dims[1]} px" if dims else ""

    ann_json = json.dumps(annotations or [])
    init_region_json = json.dumps(initial_region) if initial_region else "null"
    tile_source_json = json.dumps(tile_source) if tile_source else "null"
    comp_tile_source_json = json.dumps(comparison_tile_source) if comparison_tile_source else "null"

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.1/openseadragon.min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background-color: #111820;
      color: #F1F2F3;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      overflow: hidden;
    }}
    .viewer-container {{
      display: flex;
      gap: 10px;
      width: 100%;
      height: {viewer_height}px;
      background: #171D24;
      border: 1px solid #2D3642;
      border-radius: 8px;
      padding: 8px;
    }}
    .viewer-pane {{
      flex: 1;
      height: 100%;
      display: flex;
      flex-direction: column;
      background: #111820;
      border: 1px solid #2D3642;
      border-radius: 6px;
      overflow: hidden;
      position: relative;
    }}
    .pane-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 6px 12px;
      background: #202832;
      border-bottom: 1px solid #2D3642;
      font-size: 0.82rem;
      font-weight: 600;
      color: #C5A15A;
      z-index: 10;
    }}
    .header-left {{
      display: flex;
      align-items: center;
      gap: 8px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .meta-badge {{
      background: #171D24;
      border: 1px solid #2D3642;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 0.72rem;
      color: #94A3B8;
      font-weight: normal;
    }}
    .zoom-badge {{
      background: #1A2836;
      border: 1px solid #38BDF8;
      color: #38BDF8;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 0.72rem;
      font-weight: 600;
    }}
    .toolbar {{
      display: flex;
      gap: 4px;
      align-items: center;
    }}
    .btn {{
      background: #202832;
      border: 1px solid #2D3642;
      color: #CBD5E1;
      padding: 3px 8px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 0.78rem;
      font-weight: 500;
      transition: all 0.15s;
    }}
    .btn:hover {{
      background: #2D3642;
      color: #C5A15A;
      border-color: #C5A15A;
    }}
    .btn-active {{
      background: #3B2E15;
      color: #C5A15A;
      border-color: #C5A15A;
    }}
    .osd-viewport {{
      flex: 1;
      width: 100%;
      height: 100%;
      background: #0D1217;
      position: relative;
    }}
    /* Glyphs Overlay Styling */
    .glyph-overlay {{
      cursor: pointer;
      border-radius: 2px;
      transition: background 0.2s, box-shadow 0.2s;
    }}
    .glyph-clear {{
      border: 2px solid #C5A15A;
      background: rgba(197, 161, 90, 0.22);
    }}
    .glyph-clear:hover {{
      background: rgba(197, 161, 90, 0.45);
      box-shadow: 0 0 8px #C5A15A;
    }}
    .glyph-weathered {{
      border: 2px dashed #F59E0B;
      background: rgba(245, 158, 11, 0.22);
    }}
    .glyph-weathered:hover {{
      background: rgba(245, 158, 11, 0.45);
      box-shadow: 0 0 8px #F59E0B;
    }}
    .glyph-partial {{
      border: 2px dashed #38BDF8;
      background: rgba(56, 189, 248, 0.22);
    }}
    .glyph-partial:hover {{
      background: rgba(56, 189, 248, 0.45);
      box-shadow: 0 0 8px #38BDF8;
    }}
    .glyph-illegible {{
      border: 2px dotted #EF4444;
      background: rgba(239, 68, 68, 0.25);
    }}
    .glyph-illegible:hover {{
      background: rgba(239, 68, 68, 0.50);
      box-shadow: 0 0 8px #EF4444;
    }}
    /* Tooltip Banner */
    .overlay-tooltip {{
      position: absolute;
      bottom: 8px;
      left: 8px;
      background: rgba(23, 29, 36, 0.95);
      border: 1px solid #C5A15A;
      border-radius: 4px;
      padding: 6px 10px;
      font-size: 0.75rem;
      color: #F1F2F3;
      z-index: 20;
      pointer-events: none;
      display: none;
      box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }}
    /* Error / Notification Banner */
    .error-banner {{
      position: absolute;
      top: 40px;
      left: 12px;
      right: 12px;
      background: rgba(239, 68, 68, 0.9);
      color: #FFFFFF;
      padding: 6px 12px;
      border-radius: 4px;
      font-size: 0.75rem;
      display: none;
      z-index: 30;
    }}
    /* Loading Spinner */
    .spinner {{
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      width: 32px;
      height: 32px;
      border: 3px solid #2D3642;
      border-top: 3px solid #C5A15A;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      z-index: 25;
    }}
    @keyframes spin {{
      0% {{ transform: translate(-50%, -50%) rotate(0deg); }}
      100% {{ transform: translate(-50%, -50%) rotate(360deg); }}
    }}
  </style>
</head>
<body>
  <div class="viewer-container">
    <!-- Primary Plate Viewer -->
    <div class="viewer-pane">
      <div class="pane-header">
        <div class="header-left">
          <span>📷 {primary_label}</span>
          {'<span class="meta-badge">📐 ' + dim_text + '</span>' if dim_text else ''}
          <span id="p-zoom-level" class="zoom-badge">🔍 100%</span>
        </div>
        <div class="toolbar">
          <button class="btn" id="p-zoom-in" title="Deep Zoom In">+</button>
          <button class="btn" id="p-zoom-out" title="Deep Zoom Out">-</button>
          <button class="btn" id="p-home" title="Fit to Screen / Reset View">⟲</button>
          <button class="btn" id="p-rotate-cw" title="Rotate 90° Clockwise">↻ 90°</button>
          <button class="btn" id="p-rotate-ccw" title="Rotate 90° Counter-Clockwise">↺ 90°</button>
          <button class="btn btn-active" id="p-toggle-ann" title="Toggle Glyph Overlays">🏷️ Glyphs</button>
        </div>
      </div>
      <div id="p-spinner" class="spinner"></div>
      <div id="p-error-banner" class="error-banner"></div>
      <div id="openseadragon-primary" class="osd-viewport"></div>
      <div id="p-tooltip" class="overlay-tooltip"></div>
    </div>

    <!-- Secondary Comparative Viewer -->
    {'<div class="viewer-pane">' if is_side_by_side else ''}
    {'<div class="pane-header"><div class="header-left"><span>🔬 ' + comparison_label + '</span></div><div class="toolbar"><button class="btn btn-active" id="s-sync" title="Synchronize Pan & Zoom">🔗 Sync Viewports</button></div></div>' if is_side_by_side else ''}
    {'<div id="openseadragon-secondary" class="osd-viewport"></div>' if is_side_by_side else ''}
    {'</div>' if is_side_by_side else ''}
  </div>

  <script>
    const annotations = {ann_json};
    const initialRegion = {init_region_json};
    const primaryFallbackUri = "{primary_uri}";
    const customTileSource = {tile_source_json};

    function resolveTileSource() {{
      if (customTileSource && typeof customTileSource === "string") {{
        return customTileSource;
      }} else if (customTileSource && typeof customTileSource === "object") {{
        return customTileSource;
      }}
      return {{
        type: "image",
        url: primaryFallbackUri
      }};
    }}

    const primaryViewer = OpenSeadragon({{
      id: "openseadragon-primary",
      prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.1/images/",
      tileSources: resolveTileSource(),
      showNavigationControl: false,
      maxZoomPixelRatio: 4.0,
      minZoomImageRatio: 0.8,
      visibilityRatio: 0.5,
      constrainDuringPan: true,
      animationTime: 0.35,
      blendTime: 0.1,
      springStiffness: 8.0
    }});

    // Hide spinner on open
    primaryViewer.addHandler("open", function() {{
      const spinner = document.getElementById("p-spinner");
      if (spinner) spinner.style.display = "none";

      // If initial spatial region is specified, pan and zoom to it
      if (initialRegion && initialRegion.w > 0 && initialRegion.h > 0) {{
        try {{
          const rect = primaryViewer.viewport.imageToViewportRectangle(
            initialRegion.x, initialRegion.y, initialRegion.w, initialRegion.h
          );
          primaryViewer.viewport.fitBounds(rect, true);
        }} catch(e) {{
          console.error("Failed to fit initial region:", e);
        }}
      }}
    }});

    // Graceful error fallback
    primaryViewer.addHandler("open-failed", function(e) {{
      console.warn("TileSource failed, falling back to direct image:", e);
      const errBanner = document.getElementById("p-error-banner");
      if (errBanner) {{
        errBanner.innerText = "⚠️ Remote IIIF tile service unavailable. Local image representation restored.";
        errBanner.style.display = "block";
        setTimeout(() => {{ errBanner.style.display = "none"; }}, 4000);
      }}
      if (primaryFallbackUri) {{
        primaryViewer.open({{ type: "image", url: primaryFallbackUri }});
      }}
    }});

    // Update live zoom badge
    primaryViewer.addHandler("zoom", function() {{
      try {{
        const z = primaryViewer.viewport.getZoom(true);
        const el = document.getElementById("p-zoom-level");
        if (el) {{
          el.innerText = "🔍 " + Math.round(z * 100) + "%";
        }}
      }} catch(e) {{}}
    }});

    // Toolbar controls
    document.getElementById("p-zoom-in")?.addEventListener("click", () => primaryViewer.viewport.zoomBy(1.25));
    document.getElementById("p-zoom-out")?.addEventListener("click", () => primaryViewer.viewport.zoomBy(0.8));
    document.getElementById("p-home")?.addEventListener("click", () => primaryViewer.viewport.goHome());
    document.getElementById("p-rotate-cw")?.addEventListener("click", () => {{
      const current = primaryViewer.viewport.getRotation();
      primaryViewer.viewport.setRotation((current + 90) % 360);
    }});
    document.getElementById("p-rotate-ccw")?.addEventListener("click", () => {{
      const current = primaryViewer.viewport.getRotation();
      primaryViewer.viewport.setRotation((current - 90 + 360) % 360);
    }});

    // Add and toggle glyph annotations overlays
    let showAnnotations = true;
    const overlayElements = [];
    const tooltip = document.getElementById("p-tooltip");

    primaryViewer.addHandler("open", function() {{
      annotations.forEach((ann, idx) => {{
        if (ann.bounding_box) {{
          const bb = ann.bounding_box;
          const status = (ann.legibility_status || "CLEAR").toLowerCase();
          const elt = document.createElement("div");
          elt.className = "glyph-overlay glyph-" + status;
          elt.id = "glyph-marker-" + idx;

          // Tooltip interaction
          elt.addEventListener("mouseenter", (ev) => {{
            if (tooltip) {{
              tooltip.innerHTML = "<strong>" + (ann.character_label || "") + "</strong> (" + (ann.diplomatic_reading || "") + ") &bull; <em>" + (ann.script_category || "") + "</em><br><small>Status: " + (ann.legibility_status || "CLEAR") + " | [" + bb.x + ", " + bb.y + ", " + bb.w + ", " + bb.h + "]</small>";
              tooltip.style.display = "block";
            }}
          }});
          elt.addEventListener("mouseleave", () => {{
            if (tooltip) tooltip.style.display = "none";
          }});

          try {{
            const rect = primaryViewer.viewport.imageToViewportRectangle(bb.x, bb.y, bb.w, bb.h);
            primaryViewer.addOverlay({{
              element: elt,
              location: rect
            }});
            overlayElements.push(elt);
          }} catch(e) {{
            console.error("Failed to add glyph overlay:", e);
          }}
        }}
      }});
    }});

    document.getElementById("p-toggle-ann")?.addEventListener("click", function() {{
      showAnnotations = !showAnnotations;
      this.classList.toggle("btn-active", showAnnotations);
      overlayElements.forEach(el => {{
        el.style.display = showAnnotations ? "block" : "none";
      }});
    }});

    // Synchronized Side-by-Side Viewport Engine
    {'''
    const compViewer = OpenSeadragon({
      id: "openseadragon-secondary",
      prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.1/images/",
      tileSources: ''' + (comp_tile_source_json if comparison_tile_source else f'{{"type": "image", "url": "{comp_uri}"}}') + ''',
      showNavigationControl: false,
      maxZoomPixelRatio: 4.0,
      minZoomImageRatio: 0.8,
      visibilityRatio: 0.5,
      constrainDuringPan: true,
      animationTime: 0.35
    });

    let isSyncing = true;
    const syncBtn = document.getElementById("s-sync");
    syncBtn?.addEventListener("click", () => {
      isSyncing = !isSyncing;
      syncBtn.classList.toggle("btn-active", isSyncing);
    });

    primaryViewer.addHandler("pan", function() {
      if (isSyncing && compViewer && compViewer.viewport) {
        compViewer.viewport.panTo(primaryViewer.viewport.getCenter(), true);
      }
    });

    primaryViewer.addHandler("zoom", function() {
      if (isSyncing && compViewer && compViewer.viewport) {
        compViewer.viewport.zoomTo(primaryViewer.viewport.getZoom(), null, true);
      }
    });

    compViewer.addHandler("pan", function() {
      if (isSyncing && primaryViewer && primaryViewer.viewport) {
        primaryViewer.viewport.panTo(compViewer.viewport.getCenter(), true);
      }
    });

    compViewer.addHandler("zoom", function() {
      if (isSyncing && primaryViewer && primaryViewer.viewport) {
        primaryViewer.viewport.zoomTo(compViewer.viewport.getZoom(), null, true);
      }
    });
    ''' if is_side_by_side else ''}
  </script>
</body>
</html>
"""
    st.components.v1.html(html_content, height=viewer_height + 25, scrolling=False)
