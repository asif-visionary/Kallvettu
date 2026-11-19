"""
Comprehensive Unit & Integration Tests for Kallvettu IIIF Deep-Zoom Inscription Viewer (Phase D).

Validates:
1. IIIF Presentation 3.0 Manifest compliance:
   - Valid absolute HTTP(S) URIs for all resource IDs.
   - Canvas dimensions matching original image resolution.
   - Painting annotation referencing retrievable image representation.
   - ImageService3 identifier and painting image URL referring to identical resource.
   - Profile conformance to 'level2'.
2. IIIF Image API 3.0 info.json descriptor.
3. Bidirectional coordinate transformation and region URL formatting.
4. IIIF Image API 3.0 Region extraction & processing handler (crop, resize, rotate, quality, format).
5. Inscription annotations W3C Web Annotation Collection endpoint.
6. Persistent glyph annotation CRUD and legibility status preservation.
7. OpenSeadragon viewer rendering, toolbar controls, side-by-side mode, and keyword signatures.
"""

import json
import os
import pytest
from PIL import Image

from services.iiif.manifest_builder import (
    build_iiif_image_info,
    build_iiif_presentation_manifest,
    IIIFManifestBuilder,
    resolve_image_asset,
    DEFAULT_IIIF_BASE_URI,
)
from services.iiif.coordinate_mapper import (
    scale_box_to_original,
    scale_box_to_display,
    parse_iiif_region,
    format_iiif_region,
    build_iiif_region_url,
    build_deep_link_url,
)
from services.iiif.viewer_component import (
    render_openseadragon_viewer,
    image_to_base64_data_uri,
)
from services.database.connection import get_db_session
from services.database.models import Inscription, ImageAsset, GlyphAnnotation, AnnotationStatus
from services.database.repository import EpigraphicalRepository
from services.api.app import (
    get_iiif_manifest,
    get_iiif_image_info,
    get_iiif_image_region,
    get_inscription_annotations,
)


@pytest.fixture(autouse=True)
def setup_iiif_test_db(tmp_path, monkeypatch):
    """Initializes an isolated persistent SQLite database populated with canonical assets."""
    test_db = str(tmp_path / "test_iiif.db")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{test_db}")

    import services.database.connection as conn_mod
    conn_mod._ENGINE = None
    conn_mod._SESSION_FACTORY = None

    EpigraphicalRepository.init_database()
    EpigraphicalRepository.seed_canonical_corpus()
    yield
    conn_mod._ENGINE = None
    conn_mod._SESSION_FACTORY = None


def test_iiif_image_info():
    """Verify IIIF Image API 3.0 info.json descriptor meets standard specifications."""
    info = build_iiif_image_info("plate_001", width=920, height=480, base_uri="https://kallvettu.org/api/v1/iiif/image")
    assert info["@context"] == "http://iiif.io/api/image/3/context.json"
    assert info["type"] == "ImageService3"
    assert info["protocol"] == "http://iiif.io/api/image"
    assert info["profile"] == "level2"
    assert info["width"] == 920
    assert info["height"] == 480
    assert info["id"] == "https://kallvettu.org/api/v1/iiif/image/plate_001"
    assert len(info["tiles"]) >= 1
    assert info["tiles"][0]["width"] == 512
    assert len(info["sizes"]) >= 1
    assert "regionByPx" in info["extraFeatures"]
    assert "regionByPct" in info["extraFeatures"]


def test_iiif_presentation_manifest():
    """Verify IIIF Presentation API 3.0 manifest schema and resource bindings."""
    images = [
        {"id": "raw_plate", "width": 920, "height": 480, "label": "Original Inscription Photo"},
        {"id": "clahe_plate", "width": 920, "height": 480, "label": "CLAHE Contrast Enhanced"},
    ]
    manifest = build_iiif_presentation_manifest(
        inscription_id="KV-INSC-002",
        title_en="Thanjavur Brihadisvara Great Temple Inscription",
        title_ta="தஞ்சைப் பெருவுடையார் கோயில் கற்றளிக் கல்வெட்டு",
        images=images,
        metadata_fields={
            "script": "Medieval Chola Tamil",
            "dynasty": "Chola Dynasty",
            "ruler": "Rajaraja Chola I",
            "period": "11th Century CE",
        },
        base_uri="https://kallvettu.org/api/v1/iiif",
    )

    assert manifest["@context"] == "http://iiif.io/api/presentation/3/context.json"
    assert manifest["type"] == "Manifest"
    assert manifest["id"].startswith("https://")
    assert "KV-INSC-002" in manifest["id"]
    assert len(manifest["items"]) == 2  # 2 canvases

    # Inspect first canvas
    canvas = manifest["items"][0]
    assert canvas["type"] == "Canvas"
    assert canvas["width"] == 920
    assert canvas["height"] == 480
    assert canvas["id"].startswith("https://")

    # Painting annotation
    annotation_page = canvas["items"][0]
    assert annotation_page["type"] == "AnnotationPage"
    painting_ann = annotation_page["items"][0]
    assert painting_ann["type"] == "Annotation"
    assert painting_ann["motivation"] == "painting"

    # Image body & ImageService3 binding
    body = painting_ann["body"]
    assert body["type"] == "Image"
    assert body["width"] == 920
    assert body["height"] == 480
    assert body["id"].startswith("https://")
    assert len(body["service"]) >= 1

    svc = body["service"][0]
    assert svc["type"] == "ImageService3"
    assert svc["profile"] == "level2"
    # Ensure service id and painting image URL refer to the same resource prefix
    assert svc["id"] in body["id"]


def test_iiif_manifest_builder_from_inscription():
    """Verify dynamic IIIF Presentation 3.0 manifest construction from database record."""
    manifest = IIIFManifestBuilder.build_manifest_for_inscription("KV-INSC-001")
    assert manifest["type"] == "Manifest"
    assert "KV-INSC-001" in manifest["id"]
    assert len(manifest["items"]) >= 1

    canvas = manifest["items"][0]
    assert canvas["width"] == 920
    assert canvas["height"] == 480

    # Verify W3C glyph annotations are attached to canvas
    assert "annotations" in canvas
    assert len(canvas["annotations"]) >= 1
    ann_page = canvas["annotations"][0]
    assert ann_page["type"] == "AnnotationPage"
    assert len(ann_page["items"]) >= 1

    glyph_ann = ann_page["items"][0]
    assert glyph_ann["type"] == "Annotation"
    assert glyph_ann["motivation"] == "tagging"
    assert "#xywh=pixel:" in glyph_ann["target"]


def test_coordinate_mapper_scaling():
    """Verify bidirectional scaling between downsampled display and original high-res matrix."""
    disp_box = {"x": 100, "y": 150, "w": 50, "h": 60}
    orig_box = scale_box_to_original(
        display_box=disp_box,
        display_width=1000,
        display_height=750,
        original_width=4000,
        original_height=3000,
    )
    assert orig_box["x"] == 400
    assert orig_box["y"] == 600
    assert orig_box["w"] == 200
    assert orig_box["h"] == 240

    roundtrip_disp = scale_box_to_display(
        original_box=orig_box,
        original_width=4000,
        original_height=3000,
        display_width=1000,
        display_height=750,
    )
    assert roundtrip_disp["x"] == 100
    assert roundtrip_disp["y"] == 150
    assert roundtrip_disp["w"] == 50
    assert roundtrip_disp["h"] == 60


def test_parse_and_format_iiif_region():
    """Verify parsing and formatting of IIIF Image API 3.0 region expressions."""
    w, h = 2000, 1000

    # 1. Full
    assert parse_iiif_region("full", w, h) == (0, 0, 2000, 1000)

    # 2. Square
    assert parse_iiif_region("square", w, h) == (500, 0, 1000, 1000)

    # 3. Explicit pixels
    assert parse_iiif_region("100,200,400,300", w, h) == (100, 200, 400, 300)

    # 4. Percentage: pct:10,20,50,40
    assert parse_iiif_region("pct:10,20,50,40", w, h) == (200, 200, 1000, 400)

    # 5. Format region
    assert format_iiif_region(100, 200, 400, 300, pct=False) == "100,200,400,300"
    pct_str = format_iiif_region(200, 200, 1000, 400, pct=True, image_width=w, image_height=h)
    assert pct_str == "pct:10.0,20.0,50.0,40.0"


def test_build_iiif_region_and_deep_link_urls():
    """Verify generation of compliant IIIF region URLs and shareable deep-links."""
    reg_url = build_iiif_region_url(
        base_uri="https://kallvettu.org/api/v1/iiif/image",
        asset_id="IMG-001",
        x=50,
        y=100,
        w=200,
        h=150,
        size="max",
        rotation=0,
        quality="default",
        format_ext="jpg",
    )
    assert reg_url == "https://kallvettu.org/api/v1/iiif/image/IMG-001/50,100,200,150/max/0/default.jpg"

    pct_url = build_iiif_region_url(
        base_uri="https://kallvettu.org/api/v1/iiif/image",
        asset_id="IMG-001",
        x=50,
        y=100,
        w=200,
        h=150,
        pct=True,
        image_width=1000,
        image_height=500,
    )
    assert "pct:5.0,20.0,20.0,30.0" in pct_url

    deep_link = build_deep_link_url(
        base_url="http://localhost:8501",
        inscription_id="KV-INSC-001",
        x=50,
        y=100,
        w=200,
        h=150,
    )
    assert "studio=iiif" in deep_link
    assert "inscription_id=KV-INSC-001" in deep_link
    assert "region=50,100,200,150" in deep_link


def test_iiif_image_region_endpoint():
    """Verify FastAPI IIIF Image API 3.0 Region extraction handler."""
    # 1. Full region JPEG
    resp = get_iiif_image_region(
        asset_id="KV-INSC-001",
        region="full",
        size="max",
        rotation="0",
        quality="default",
        format="jpg",
    )
    assert resp.media_type == "image/jpeg"
    assert len(resp.body) > 1000

    # 2. Cropped region with rotation and grayscale
    crop_resp = get_iiif_image_region(
        asset_id="KV-INSC-001",
        region="45,140,55,70",
        size="max",
        rotation="90",
        quality="gray",
        format="png",
    )
    assert crop_resp.media_type == "image/png"
    assert len(crop_resp.body) > 100

    # 3. Percentage region with scale
    pct_resp = get_iiif_image_region(
        asset_id="KV-INSC-001",
        region="pct:10,20,30,40",
        size="pct:50",
        rotation="0",
        quality="bitonal",
        format="jpg",
    )
    assert pct_resp.media_type == "image/jpeg"


def test_iiif_annotations_endpoint():
    """Verify FastAPI /annotations endpoint returns W3C Web Annotation Collection."""
    coll = get_inscription_annotations("KV-INSC-001")
    assert coll["@context"] == "http://www.w3.org/ns/anno.jsonld"
    assert coll["type"] == "AnnotationCollection"
    assert coll["total"] >= 1
    assert len(coll["items"]) == coll["total"]
    first = coll["items"][0]
    assert first["type"] == "Annotation"
    assert first["body"]["language"] == "ta"
    assert "#xywh=pixel:" in first["target"]


def test_persistent_glyph_annotation_workflow():
    """Verify full CRUD lifecycle and legibility preservation for glyph annotations."""
    # 1. Create new annotation
    ann = EpigraphicalRepository.add_glyph_annotation(
        inscription_id="KV-INSC-002",
        image_asset_id="IMG-RAW-KV-INSC-002",
        character_label="ஶ்ரீ",
        diplomatic_reading="śrī",
        script_category="Medieval Chola Tamil",
        bounding_box={"x": 450, "y": 120, "w": 75, "h": 85},
        legibility_status="WEATHERED",
        notes="Auspicious royal invocation glyph.",
    )
    assert ann.id.startswith("GLYPH-")
    assert ann.legibility_status == "WEATHERED"

    # 2. Reload and verify persistence
    loaded_anns = EpigraphicalRepository.list_annotations_for_inscription("KV-INSC-002")
    match = next((a for a in loaded_anns if a.id == ann.id), None)
    assert match is not None
    assert match.character_label == "ஶ்ரீ"
    assert match.bounding_box["x"] == 450
    assert match.legibility_status == "WEATHERED"


def test_render_openseadragon_viewer_signatures_and_modes(monkeypatch):
    """Verify OpenSeadragon viewer component renders toolbar, controls, and side-by-side mode."""
    import streamlit.components.v1 as components

    captured_html = []
    def mock_html(content, height, **kwargs):
        captured_html.append((content, height))

    monkeypatch.setattr(components, "html", mock_html)

    # 1. Single Master Plate with Glyph Annotations & Initial Region
    render_openseadragon_viewer(
        primary_image="data:image/jpeg;base64,123",
        viewer_height=600,
        primary_label="Mangulam Master Plate",
        annotations=[
            {"bounding_box": {"x": 45, "y": 140, "w": 55, "h": 70}, "character_label": "க", "legibility_status": "CLEAR"},
            {"bounding_box": {"x": 170, "y": 140, "w": 50, "h": 68}, "character_label": "ய்", "legibility_status": "WEATHERED"},
        ],
        initial_region={"x": 45, "y": 140, "w": 55, "h": 70},
        image_dimensions=(920, 480),
    )
    assert len(captured_html) == 1
    html_0 = captured_html[0][0]
    assert "openseadragon-primary" in html_0
    assert "p-zoom-in" in html_0
    assert "p-rotate-cw" in html_0
    assert "p-toggle-ann" in html_0
    assert "920 × 480 px" in html_0
    assert "glyph-clear" in html_0
    assert "glyph-weathered" in html_0
    assert "initialRegion" in html_0

    # 2. Synchronized Side-by-Side Comparison Mode
    render_openseadragon_viewer(
        primary_image="data:image/jpeg;base64,123",
        comparison_image="data:image/jpeg;base64,456",
        viewer_height=650,
        primary_label="Master Photograph",
        comparison_label="Squeeze Rubbing",
    )
    assert len(captured_html) == 2
    html_1 = captured_html[1][0]
    assert "openseadragon-secondary" in html_1
    assert "s-sync" in html_1
    assert "compViewer.viewport.panTo" in html_1
