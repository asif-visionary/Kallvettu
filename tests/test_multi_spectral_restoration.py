"""
Unit & Integration Tests for Multi-Spectral Image Processing Dashboard (Phases 2 & 3).

Verifies:
1. Section 2: Multi-Spectral Image Path & OCR Architecture (Paths A, B, and C).
   - Path A: Pristine untouched original.
   - Path B: Mild CLAHE (clip 1.8, blend 0.85) stroke preservation without halo.
   - Standard CLAHE: Full histogram equalization.
   - Denoised Grayscale: Bilateral stone smoothing.
   - Path C: Edge Relief (Sobel grooves) labeled visual inspection only.
2. Section 3: Multi-Spectral Processing Layers.
   - Layer 1: CLAHE Contrast Boost.
   - Layer 2: Chisel Groove Ridges.
   - Layer 3: Sauvola Adaptive Binarization.
   - Layer 4: Image Enhancement Difference Map (|Original - CLAHE|).
   - Layer 5: Experimental Surface Anomaly Map (Roughness Z-score).
3. Backend image persistence to disk with automatic directory creation and request isolation.
4. get_safe_display_image validation preventing 404 broken-image URL strings in Streamlit.
5. Strict isolation: Edge maps never designated as safe for OCR.
"""

import os
import shutil
import tempfile
import numpy as np
from PIL import Image
import pytest

from restoration_engine import (
    process_full_restoration,
    generate_image_paths,
    compute_enhancement_difference_map,
    compute_experimental_surface_anomalies,
    save_restoration_outputs,
    get_safe_display_image,
    DEFAULT_RESTORATION_OUTPUT_DIR,
)


@pytest.fixture
def synthetic_stone_image():
    """Generates a realistic test granite slab image with chiseled grooves."""
    img = Image.new("RGB", (320, 240), color=(130, 125, 120))
    arr = np.array(img)
    # Add simulated chiseled inscription line
    arr[60:65, 30:290] = [60, 55, 50]
    arr[120:125, 30:290] = [50, 45, 40]
    # Add granite speckle variance
    rng = np.random.default_rng(42)
    noise = rng.integers(-10, 10, size=arr.shape, dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def test_section_2_image_paths_architecture(synthetic_stone_image):
    """Verify Section 2: All 5 processing paths are present, properly typed, and strictly governed."""
    res = process_full_restoration(synthetic_stone_image, source_identifier="sec2_test")

    # 1. Path A: Original (untouched)
    assert "original" in res
    assert isinstance(res["original"], Image.Image)
    assert res["original"].size == (320, 240)

    # 2. Path B: Mild CLAHE
    assert "mild_clahe" in res
    assert isinstance(res["mild_clahe"], Image.Image)
    assert "path_b" in res
    assert isinstance(res["path_b"], Image.Image)
    assert "default_ocr_image" in res
    assert isinstance(res["default_ocr_image"], Image.Image)

    # 3. Standard CLAHE
    assert "clahe" in res
    assert isinstance(res["clahe"], Image.Image)

    # 4. Denoised Grayscale
    assert "denoised" in res
    assert isinstance(res["denoised"], Image.Image)

    # 5. Path C: Edge Relief (grooves)
    assert "grooves" in res
    assert isinstance(res["grooves"], Image.Image)

    # Architectural rule: Path C is strictly prohibited for OCR
    img_paths = res["image_paths"]
    assert img_paths["path_a"]["is_safe_for_ocr"] is True
    assert img_paths["path_b"]["is_safe_for_ocr"] is True
    assert img_paths["path_c"]["is_safe_for_ocr"] is False


def test_section_3_multi_spectral_processing_layers(synthetic_stone_image):
    """Verify Section 3: All 5 spectral layers are generated, valid RGB images, and non-empty."""
    res = process_full_restoration(synthetic_stone_image, source_identifier="sec3_test")

    # Layer 1: CLAHE Contrast Boost
    assert isinstance(res["clahe"], Image.Image)
    assert res["clahe"].mode == "RGB"

    # Layer 2: Chisel Groove Ridges
    assert isinstance(res["grooves"], Image.Image)
    assert res["grooves"].mode == "RGB"

    # Layer 3: Sauvola Adaptive Binarized Glyphs
    assert isinstance(res["binarized"], Image.Image)
    assert res["binarized"].mode == "RGB"

    # Layer 4: Enhancement Difference Map
    assert "enhancement_diff" in res
    assert isinstance(res["enhancement_diff"], Image.Image)
    assert res["enhancement_diff"].size == (320, 240)

    # Layer 5: Surface Anomaly Map
    assert "anomaly_map" in res
    assert isinstance(res["anomaly_map"], Image.Image)
    assert res["anomaly_map"].size == (320, 240)


def test_disk_persistence_and_directory_creation(synthetic_stone_image):
    """Verify outputs are saved to disk with consistent filenames and automatic directory creation."""
    temp_dir = tempfile.mkdtemp(prefix="kallvettu_test_out_")
    try:
        res = process_full_restoration(synthetic_stone_image, source_identifier="persist_test")
        saved_paths = save_restoration_outputs(res, output_dir=temp_dir, source_id="test_inscr")

        expected_files = [
            "original",
            "mild_clahe",
            "path_b",
            "clahe",
            "denoised",
            "squeeze",
            "grooves",
            "binarized",
            "enhancement_diff",
            "anomaly_map",
        ]

        for layer in expected_files:
            assert layer in saved_paths, f"Layer '{layer}' was not saved to disk"
            file_path = saved_paths[layer]
            assert os.path.isfile(file_path), f"File {file_path} does not exist"
            assert os.path.getsize(file_path) > 0, f"File {file_path} is empty"

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_concurrent_request_isolation(synthetic_stone_image):
    """Verify different images / requests are saved in distinct subdirectories without collision."""
    temp_dir = tempfile.mkdtemp(prefix="kallvettu_concurrent_")
    try:
        # Request 1
        res1 = process_full_restoration(synthetic_stone_image, source_identifier="req1")
        paths1 = save_restoration_outputs(res1, output_dir=temp_dir, source_id="req1")

        # Request 2 with modified image
        img2 = synthetic_stone_image.copy()
        arr2 = np.array(img2)
        arr2[10:20, 10:20] = [255, 255, 255]
        img2_mod = Image.fromarray(arr2)
        res2 = process_full_restoration(img2_mod, source_identifier="req2")
        paths2 = save_restoration_outputs(res2, output_dir=temp_dir, source_id="req2")

        dir1 = os.path.dirname(paths1["original"])
        dir2 = os.path.dirname(paths2["original"])
        assert dir1 != dir2, "Concurrent requests must not write to the same subdirectory"
        assert os.path.isfile(paths1["original"])
        assert os.path.isfile(paths2["original"])
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_safe_display_image_prevents_broken_icons(synthetic_stone_image):
    """Verify get_safe_display_image never returns missing file path strings to Streamlit."""
    # 1. Valid existing file returns path
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    temp_file.close()
    synthetic_stone_image.save(temp_file.name)
    try:
        assert get_safe_display_image(temp_file.name) == temp_file.name
    finally:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)

    # 2. Non-existent path returns None or fallback (never the broken string)
    assert get_safe_display_image("nonexistent_path/fake_image.png") is None
    fallback = synthetic_stone_image
    assert get_safe_display_image("nonexistent_path/fake_image.png", fallback_image=fallback) == fallback

    # 3. Unwraps nested dictionary
    dict_candidate = {"image": synthetic_stone_image, "label": "Path B"}
    assert get_safe_display_image(dict_candidate) == synthetic_stone_image

    # 4. Valid PIL Image returned directly
    assert get_safe_display_image(synthetic_stone_image) == synthetic_stone_image
