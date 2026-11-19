"""
Comprehensive Unit & Integration Tests for K-IEFE Visual Epigraph Dispute Heatmap.

Tests:
1. Region detection accuracy and non-fabrication of phantom boxes.
2. Dual-layer crop correspondence (Original vs Restored).
3. Candidate distribution Shannon entropy and normalized entropy calculation.
4. Color semantic categorization (Red, Amber, Green, Grey).
5. Safeguard handling of missing candidates and single-engine outputs (NOT_ASSESSABLE).
6. Heatmap overlay rendering with opacity control and reticle highlighting.
7. Unicode Tamil label rendering and fallback without tofu mojibake.
8. Epigraphist adjudication actions and audit tracking.
"""

import math
from typing import Dict, List
import numpy as np
import pytest
from PIL import Image

from services.ai.common.base_provider import OCRResult, BoundingBox
from services.ocr_fusion import (
    fuse_epigraphical_evidence,
    MasterEvidenceFusionBundle,
    ResearcherReviewManager,
    AdjudicationDecision,
    AdjudicationItem,
)
from services.ocr_fusion.consensus import (
    ConsensusEntropyCalculator,
    DisagreementSeverity,
    PositionEntropyResult,
)
from services.ocr_fusion.cross_verifier import (
    CharacterAlignmentUnit,
    EpigraphicalCrossVerifier,
)
from services.ocr_fusion.region_resolver import (
    resolve_text_regions,
    crop_region_layers,
)
from services.ocr_fusion.ui_renderer import (
    render_evidence_heatmap_overlay,
    _get_overlay_font,
)


@pytest.fixture
def sample_inscription_image():
    """Synthetic master stone photograph with simulated inscription bands."""
    img = Image.new("RGB", (600, 300), color=(140, 130, 120))
    # Draw simple darker bands to simulate incised grooves
    arr = np.array(img)
    arr[100:150, 50:550] = (80, 70, 60)
    arr[180:230, 50:550] = (85, 75, 65)
    return Image.fromarray(arr)


def test_region_resolver_does_not_fabricate_phantom_full_width_boxes(sample_inscription_image):
    """Verifies that resolved regions are localized and do not span full width."""
    units = [
        CharacterAlignmentUnit(
            position_index=i,
            candidate_character="க",
            raw_by_provider={"p1": "க", "p2": "க"},
            normalized_by_provider={"p1": "க", "p2": "க"},
            confidence_by_provider={"p1": 0.9, "p2": 0.9},
            status="match",
            is_disputed=False,
        )
        for i in range(5)
    ]
    boxes = resolve_text_regions(sample_inscription_image, units)
    assert len(boxes) == 5
    img_w, _ = sample_inscription_image.size
    for x1, y1, x2, y2 in boxes:
        # None of the single character boxes should span > 60% of the image width
        width = x2 - x1
        assert width < (img_w * 0.60), f"Region box width {width} was unexpectedly large!"
        assert width > 0
        assert y2 > y1


def test_region_resolver_respects_manual_coordinate_overrides(sample_inscription_image):
    """Verifies that manual overrides update exact coordinate slots."""
    units = [
        CharacterAlignmentUnit(
            position_index=0,
            candidate_character="ம",
            raw_by_provider={"p1": "ம"},
            normalized_by_provider={"p1": "ம"},
            confidence_by_provider={"p1": 0.9},
            status="match",
            is_disputed=False,
        )
    ]
    custom_coords = {0: (45, 65, 120, 150)}
    boxes = resolve_text_regions(sample_inscription_image, units, manual_coords=custom_coords)
    assert boxes[0] == (45, 65, 120, 150)


def test_crop_region_layers_correspondence(sample_inscription_image):
    """Verifies extraction of both original crop and enhanced restored crop."""
    coords = (50, 100, 150, 150)
    layers = crop_region_layers(sample_inscription_image, coords)
    assert "original" in layers
    assert "restored" in layers
    orig_crop = layers["original"]
    resto_crop = layers["restored"]
    assert orig_crop.size == (100, 50)
    assert resto_crop.size == (100, 50)


def test_entropy_and_color_classification_semantics():
    """
    Verifies Shannon entropy, normalized entropy, and color category classification:
    - Agreement (H <= 0.5) -> Green
    - Moderate dispute (0.5 < H <= 1.0) -> Amber
    - High dispute (H > 1.0) -> Red
    """
    # 1. Unanimous (Agreement)
    unit_unanimous = CharacterAlignmentUnit(
        position_index=0,
        candidate_character="த",
        raw_by_provider={"p1": "த", "p2": "த", "p3": "த"},
        normalized_by_provider={"p1": "த", "p2": "த", "p3": "த"},
        confidence_by_provider={"p1": 0.9, "p2": 0.9, "p3": 0.9},
        status="match",
        is_disputed=False,
    )
    res_unanimous = ConsensusEntropyCalculator.calculate_position_entropy(unit_unanimous)
    assert res_unanimous.entropy == 0.0
    assert res_unanimous.severity == DisagreementSeverity.UNANIMOUS
    assert res_unanimous.is_assessable is True

    # 2. Moderate Dispute (2 candidates equal: H = 1.0 bits)
    unit_divided = CharacterAlignmentUnit(
        position_index=1,
        candidate_character="ர",
        raw_by_provider={"p1": "ர", "p2": "ர", "p3": "த", "p4": "த"},
        normalized_by_provider={"p1": "ர", "p2": "ர", "p3": "த", "p4": "த"},
        confidence_by_provider={"p1": 0.8, "p2": 0.8, "p3": 0.8, "p4": 0.8},
        status="substitution",
        is_disputed=True,
    )
    res_divided = ConsensusEntropyCalculator.calculate_position_entropy(unit_divided)
    assert math.isclose(res_divided.entropy, 1.0, rel_tol=1e-3)
    assert res_divided.severity == DisagreementSeverity.DIVIDED
    assert res_divided.normalized_entropy == 1.0  # H / log2(2) = 1.0 / 1.0

    # 3. High Dispute (3 candidates evenly split: H = log2(3) ≈ 1.585 bits > 1.0)
    unit_high = CharacterAlignmentUnit(
        position_index=2,
        candidate_character="க",
        raw_by_provider={"p1": "க", "p2": "ச", "p3": "த"},
        normalized_by_provider={"p1": "க", "p2": "ச", "p3": "த"},
        confidence_by_provider={"p1": 0.7, "p2": 0.7, "p3": 0.7},
        status="substitution",
        is_disputed=True,
    )
    res_high = ConsensusEntropyCalculator.calculate_position_entropy(unit_high)
    assert res_high.entropy > 1.0
    assert res_high.severity == DisagreementSeverity.HIGH_DISPUTE
    assert res_high.normalized_entropy == 1.0  # evenly distributed over 3 candidates


def test_missing_candidates_marked_not_assessable():
    """Verifies that missing candidate data is classified as NOT_ASSESSABLE, never unanimous."""
    unit_empty = CharacterAlignmentUnit(
        position_index=0,
        candidate_character="",
        raw_by_provider={"p1": "-", "p2": "", "p3": "-"},
        normalized_by_provider={"p1": "-", "p2": "", "p3": "-"},
        confidence_by_provider={"p1": 0.0, "p2": 0.0, "p3": 0.0},
        status="match",
        is_disputed=False,
    )
    res_empty = ConsensusEntropyCalculator.calculate_position_entropy(unit_empty)
    assert res_empty.severity == DisagreementSeverity.NOT_ASSESSABLE
    assert res_empty.is_assessable is False
    assert res_empty.distinct_candidates == []


def test_single_provider_marked_not_assessable():
    """Verifies that when only 1 engine participated, it is not marked as consensus."""
    unit_single = CharacterAlignmentUnit(
        position_index=0,
        candidate_character="அ",
        raw_by_provider={"p1": "அ"},
        normalized_by_provider={"p1": "அ"},
        confidence_by_provider={"p1": 0.88},
        status="match",
        is_disputed=False,
    )
    res_single = ConsensusEntropyCalculator.calculate_position_entropy(unit_single)
    assert res_single.severity == DisagreementSeverity.SINGLE_PROVIDER
    assert res_single.is_assessable is False
    assert res_single.participating_providers_count == 1


def test_heatmap_overlay_rendering_and_transparency(sample_inscription_image):
    """Verifies heatmap overlay creates an alpha blend and supports opacity and reticle."""
    # Build a synthetic MasterEvidenceFusionBundle
    ocr_results = {
        "paddle": OCRResult(provider_name="paddle", model_id="paddle-v5", text="கோ மாறன் சடையன்", confidence=0.88),
        "sarvam": OCRResult(provider_name="sarvam", model_id="sarvam-v1", text="கோ மாறன் சடையன்", confidence=0.85),
    }
    bundle = fuse_epigraphical_evidence(
        image=sample_inscription_image,
        results=ocr_results,
        script_hint="vatteluttu"
    )

    assert len(bundle.adjudication_items) > 0

    # Render overlay with default opacity
    overlay_img = render_evidence_heatmap_overlay(
        sample_inscription_image,
        bundle,
        opacity=0.40,
        selected_slot=0,
        show_labels=True
    )
    assert isinstance(overlay_img, Image.Image)
    assert overlay_img.size == sample_inscription_image.size
    assert overlay_img.mode == "RGB"

    # Verify that overlay didn't overwrite the entire image with a solid color
    orig_arr = np.array(sample_inscription_image)
    over_arr = np.array(overlay_img)
    # They should not be identical because overlay was drawn
    assert not np.array_equal(orig_arr, over_arr)
    # But stone surface markings should still vary (standard deviation > 0)
    assert np.std(over_arr) > 10.0


def test_font_fallback_and_unicode_label_safety():
    """Verifies font fallback returns a valid font and does not throw on Tamil Unicode."""
    font, has_tamil = _get_overlay_font(14)
    assert font is not None
    # Test drawing on a scratch canvas with Tamil character
    test_img = Image.new("RGBA", (100, 40), (0, 0, 0, 0))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(test_img)
    # Drawing shouldn't raise any exception
    draw.text((5, 5), "R1: க", font=font, fill=(255, 255, 255, 255))
    assert test_img.size == (100, 40)


def test_adjudication_item_review_workflow():
    """Verifies adjudication decisions update state and record audit trail."""
    item = AdjudicationItem(
        item_id="adj_1",
        region_index=0,
        bounding_box=BoundingBox(ymin=0.1, xmin=0.1, ymax=0.2, xmax=0.2, text="க"),
        pixel_crop_coords=(10, 10, 50, 50),
        provider_candidates={"p1": "க", "p2": "ச"},
        provider_confidences={"p1": 0.8, "p2": 0.7},
        fused_candidate="க",
        alternative_readings=["க", "ச"],
        disagreement_reason="Substitution dispute",
        consensus_entropy=0.98,
        normalized_entropy=0.98,
        candidate_distribution={"க": 0.5, "ச": 0.5},
        participating_providers_count=2,
        is_assessable=True,
        visual_hypothesis=None,
        historical_evidence=[],
        current_decision=AdjudicationDecision.PENDING.value,
        approved_text="க"
    )

    mgr = ResearcherReviewManager()
    mgr.add_item(item)

    # 1. Scholar accepts fused
    mgr.record_decision("adj_1", AdjudicationDecision.ACCEPTED_FUSION, "க", reviewer="Dr. Scholar", rationale="Clear loop visible")
    assert item.current_decision == AdjudicationDecision.ACCEPTED_FUSION.value
    assert item.approved_text == "க"
    assert len(item.revision_history) == 1

    # 2. Scholar modifies reading
    mgr.record_decision("adj_1", AdjudicationDecision.EDITED_SCHOLAR, "த", reviewer="Dr. Scholar", rationale="Re-read as dental ta")
    assert item.current_decision == AdjudicationDecision.EDITED_SCHOLAR.value
    assert item.approved_text == "த"
    assert len(item.revision_history) == 2
