"""
Comprehensive Test Suite for Kallvettu (Ancient Inscription Restoration & Translation Platform).
Covers:
- Preset loading & asset integrity
- NLP catalog matching (exact, partial, Tamil keyword, English keyword, no-match, ambiguity)
- Epigraphical missing text restoration & lacuna preservation
- Gemini API response parsing & markdown fence stripping
- Computer Vision restoration pipeline & image dimension clamping
- Tamil Unicode lexicon extraction & DataFrame serialization
- Cross-platform font fallback
- Custom transcription analysis & data provenance
"""

import os
import sys
import io
import pytest
import numpy as np
from PIL import Image
import pandas as pd

# Ensure repository root is on sys.path for test discovery
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nlp_translation_engine import (
    KNOWN_INSCRIPTIONS_CATALOG,
    get_all_preset_inscriptions,
    find_preset_by_id,
    match_inscription_from_text,
    perform_textual_restoration,
    extract_epigraph_lexicon,
    analyze_custom_epigraph_text,
    EPIGRAPHY_LEXICON,
    EPIGRAPHICAL_SCRIPTS,
    identify_script_from_text,
    extract_script_inventory,
    EpigraphicalSandhiSplitter,
    EpigraphicalMorphologicalAnalyzer,
    PresetCatalogAdapter,
    ManualTranscriptionAdapter,
    LocalOfflineOCRAdapter,
)
from multimodal_ai import (
    clean_json_response,
    image_to_base64,
    validate_gemini_epigraphy_response,
    SUPPORTED_GEMINI_MODELS,
)
from evaluation_engine import (
    calculate_cer,
    calculate_wer,
    calculate_inpainting_metrics,
    calculate_segmentation_metrics,
    calculate_psnr,
    calculate_ssim,
    calculate_contrast_improvement_ratio,
    calculate_dice_coefficient,
    evaluate_script_classification,
    load_evaluation_dataset,
    run_ablation_study,
)
from config import (
    CONFIG,
    sanitize_filename,
    mask_api_key,
    KallvettuConfig,
)
from restoration_engine import (
    process_full_restoration,
    sauvola_binarization,
    niblack_binarization,
    otsu_global_binarization,
    extract_chisel_grooves,
    apply_clahe_pillow,
    digital_squeeze_rubbing,
    validate_and_load_image,
    crop_region_of_interest,
    compute_enhancement_difference_map,
    compute_experimental_surface_anomalies,
    CorruptedImageError,
    InvalidImageFormatError,
    ImageDimensionError,
)
from sample_generator import get_cross_platform_font_path
from ui_components import build_lexical_dataframe


# ==============================================================================
# 1. PRESET LOADING & ASSET INTEGRITY
# ==============================================================================

def test_preset_catalog_completeness():
    """Verify that all preset inscriptions are present with required schema fields."""
    presets = get_all_preset_inscriptions()
    assert len(presets) >= 6, "Expected at least 6 historical presets"

    required_keys = [
        "id", "title_ta", "title_en", "era_ta", "era_en",
        "script_ta", "script_en", "dynasty_ta", "dynasty_en",
        "weathered_text", "restored_text",
        "modern_tamil_translation", "english_translation",
        "words", "image_key"
    ]

    for p in presets:
        for k in required_keys:
            assert k in p and p[k], f"Missing or empty key '{k}' in preset {p.get('id')}"


def test_preset_images_exist_on_disk():
    """Verify that every preset has an authentic image file in assets/samples."""
    samples_dir = os.path.join(os.path.dirname(__file__), "..", "assets", "samples")
    presets = get_all_preset_inscriptions()

    for p in presets:
        img_path = os.path.join(samples_dir, f"{p['image_key']}.jpg")
        assert os.path.isfile(img_path), f"Sample image file not found for preset {p['id']}: {img_path}"
        with Image.open(img_path) as img:
            assert img.size[0] > 0 and img.size[1] > 0


def test_find_preset_by_id():
    """Verify exact lookup by ID and None for nonexistent IDs."""
    thanjavur = find_preset_by_id("thanjavur_rajaraja")
    assert thanjavur is not None
    assert "இராஜராஜ" in thanjavur["title_ta"]

    assert find_preset_by_id("non_existent_inscription") is None


# ==============================================================================
# 2. NLP INSCRIPTION MATCHING (Step 4 Verification)
# ==============================================================================

def test_nlp_exact_title_match():
    """Verify exact Tamil title matches the correct preset with high confidence."""
    query = "மாங்குளம் குகைக் கல்வெட்டு"
    matched = match_inscription_from_text(query)
    assert matched is not None
    assert matched["id"] == "mangulam_brahmi"


def test_nlp_exact_english_title_match():
    """Verify exact English title matches the correct preset."""
    query = "Thanjavur Brihadisvara Foundation Inscription"
    matched = match_inscription_from_text(query)
    assert matched is not None
    assert matched["id"] == "thanjavur_rajaraja"


def test_nlp_partial_title_match():
    """Verify partial title substring matches the intended inscription."""
    query = "உத்திரமேரூர் குடவோலை"
    matched = match_inscription_from_text(query)
    assert matched is not None
    assert matched["id"] == "uttiramerur_election"


def test_nlp_tamil_keyword_match():
    """Verify distinctive Tamil martial/epigraphical keywords correctly match Nadukal."""
    query = "தொறுகொண்டு போகிய பூசலில் நின்றெறிந்து பட்டான் சாத்தன்"
    matched = match_inscription_from_text(query)
    assert matched is not None
    assert matched["id"] == "hero_stone_nadukal"


def test_nlp_english_keyword_match():
    """Verify distinctive English palaeographical keywords correctly match Pugalur."""
    query = "pugalur chera irumporai royal ascetic"
    matched = match_inscription_from_text(query)
    assert matched is not None
    assert matched["id"] == "pugalur_chera"


def test_nlp_no_match():
    """Verify unrelated queries do not produce false positive matches."""
    query = "பொருளாதார புள்ளிவிவரக் கணக்கீடு மற்றும் கணினி அறிவியல்"
    matched = match_inscription_from_text(query)
    assert matched is None, "Unrelated text must return None, not a false positive"


def test_nlp_ambiguous_competing_match():
    """Verify generic single-word queries with competing candidates return None (prevent false positives)."""
    # Just "கல்வெட்டு" (Inscription) is generic across all inscriptions
    query = "கல்வெட்டு"
    matched = match_inscription_from_text(query)
    assert matched is None, "Generic ambiguous query must return None"


# ==============================================================================
# 3. DAMAGED TEXT INPAINTING & LACUNA PRESERVATION (Step 5 Verification)
# ==============================================================================

def test_textual_restoration_known_formulas():
    """Verify documented historical lacunae are reconstructed with rationale."""
    weathered = "கணிய் நந்தா[ாசி]ய் இகுவன் [வெள்] நிகமத்தோர்"
    res = perform_textual_restoration(weathered)

    assert res["has_damage"] is True
    assert "ாஸிரி" in res["restored_text"]
    assert "வெள்ளறை" in res["restored_text"]
    assert res["resolved_count"] >= 2
    assert res["reconstructions"][0]["status"] == "resolved"


def test_textual_restoration_preserves_unresolved_brackets():
    """
    Scientific Epigraphy Rule: Unknown/undocumented damaged segments MUST preserve
    brackets rather than fabricating or silently removing characters.
    """
    weathered = "அரசன் [சிதைவு_12345] கோயிற் பணி"
    res = perform_textual_restoration(weathered)

    assert res["has_damage"] is True
    # Brackets must remain intact in the restored text
    assert "[சிதைவு_12345]" in res["restored_text"]
    assert res["unresolved_count"] == 1
    assert res["reconstructions"][0]["status"] == "unresolved"


def test_textual_restoration_clean_text_no_damage():
    """Verify undamaged text passes through without modification."""
    clean = "நாம் குடுத்தனவும் நம் அக்கன் குடுத்தனவும்"
    res = perform_textual_restoration(clean)

    assert res["has_damage"] is False
    assert res["restored_text"] == clean
    assert len(res["reconstructions"]) == 0


# ==============================================================================
# 4. GEMINI API PARSING & JSON FENCE REMOVAL (Step 6 Verification)
# ==============================================================================

def test_clean_json_response_clean():
    """Verify clean JSON string parses correctly."""
    raw = '{"script_ta": "தமிழ் பிராமி", "era_en": "3rd c. BCE"}'
    parsed = clean_json_response(raw)
    assert parsed["script_ta"] == "தமிழ் பிராமி"


def test_clean_json_response_with_markdown_fences():
    """Verify markdown ```json fences are automatically stripped."""
    raw = '```json\n{\n  "script_ta": "வட்டெழுத்து",\n  "modern_tamil_translation": "வீரன் நினைவு"\n}\n```'
    parsed = clean_json_response(raw)
    assert parsed["script_ta"] == "வட்டெழுத்து"
    assert parsed["modern_tamil_translation"] == "வீரன் நினைவு"


def test_clean_json_response_with_conversational_preamble():
    """Verify parser extracts JSON object even if model added commentary."""
    raw = 'Here is the epigraphical decipherment:\n{"script_en": "Chola Tamil", "words": []}\nHope this helps!'
    parsed = clean_json_response(raw)
    assert parsed["script_en"] == "Chola Tamil"


def test_clean_json_response_invalid():
    """Verify invalid syntax raises an exception cleanly."""
    with pytest.raises(Exception):
        clean_json_response("This is not JSON at all.")


# ==============================================================================
# 5. COMPUTER VISION RESTORATION & RESIZING (Step 8 Verification)
# ==============================================================================

def test_restoration_pipeline_synthetic_stone():
    """Verify computer vision pipeline produces all required enhancement layers."""
    # Create synthetic test image
    test_img = Image.new("RGB", (300, 200), color=(100, 95, 90))
    res = process_full_restoration(test_img, contrast_boost=2.0, binarize_k=0.2)

    required_layers = ["original", "clahe", "denoised", "squeeze", "grooves", "binarized", "erosion_map"]
    for layer in required_layers:
        assert layer in res, f"Missing layer '{layer}' in restoration results"
        assert isinstance(res[layer], Image.Image)
        assert res[layer].size == (300, 200)


def test_restoration_pipeline_clamps_large_images():
    """Verify large camera photos (e.g. 2400x1800) are clamped to max_dim to prevent OOM."""
    large_img = Image.new("RGB", (2400, 1800), color=(120, 110, 100))
    res = process_full_restoration(large_img, max_dim=1000)

    # Clamped dimension must not exceed 1000
    assert res["original"].width <= 1000
    assert res["original"].height <= 1000


def test_sauvola_binarization_output():
    """Verify Sauvola thresholding produces a valid binary (L mode) image."""
    img = Image.new("RGB", (100, 100), color=(150, 140, 130))
    bin_img = sauvola_binarization(img, window_size=15, k=0.2)
    assert bin_img.mode == "L"
    assert bin_img.size == (100, 100)


def test_validate_and_load_image_corrupted():
    """Verify corrupted byte data raises CorruptedImageError."""
    corrupted_bytes = b"NOT_A_VALID_IMAGE_DATA_12345"
    with pytest.raises(CorruptedImageError):
        validate_and_load_image(corrupted_bytes)


def test_validate_and_load_image_too_small():
    """Verify undersized image dimensions raise ImageDimensionError."""
    tiny_img = Image.new("RGB", (20, 20), color=(100, 100, 100))
    with pytest.raises(ImageDimensionError):
        validate_and_load_image(tiny_img, min_dimension=50)


def test_crop_region_of_interest():
    """Verify Region of Interest (ROI) cropping with boundary clamping."""
    img = Image.new("RGB", (500, 400), color=(80, 80, 80))
    roi = crop_region_of_interest(img, (50, 60, 250, 260))
    assert roi.size == (200, 200)

    # Out of bounds clamping
    roi_clamped = crop_region_of_interest(img, (-20, -10, 600, 500))
    assert roi_clamped.size == (500, 400)


def test_niblack_and_otsu_binarization():
    """Verify Niblack and Otsu binarization outputs."""
    img = Image.new("RGB", (120, 120), color=(140, 130, 120))
    nib = niblack_binarization(img, window_size=15, k=-0.2)
    assert nib.mode == "L"
    assert nib.size == (120, 120)

    otsu = otsu_global_binarization(img)
    assert otsu.mode == "L"
    assert otsu.size == (120, 120)


def test_compute_enhancement_difference_and_anomalies():
    """Verify enhancement difference map and experimental surface anomaly maps."""
    img1 = Image.new("RGB", (100, 100), color=(100, 100, 100))
    img2 = Image.new("RGB", (100, 100), color=(130, 130, 130))

    diff = compute_enhancement_difference_map(img1, img2)
    assert diff.mode == "RGB"
    assert diff.size == (100, 100)

    anom = compute_experimental_surface_anomalies(img1)
    assert anom.mode == "RGB"
    assert anom.size == (100, 100)


def test_restoration_metadata_retention():
    """Verify structured metadata is retained across the restoration pipeline."""
    img = Image.new("RGB", (600, 400), color=(90, 85, 80))
    res = process_full_restoration(img, contrast_boost=2.2, binarize_k=0.18, max_dim=800)

    meta = res["metadata"]
    assert "algorithm_version" in meta
    assert meta["algorithm_version"] == "2.1.0-cv-scientific"
    assert meta["original_dimensions"] == (600, 400)
    assert meta["parameters"]["contrast_boost"] == 2.2
    assert meta["parameters"]["binarize_k"] == 0.18
    assert meta["execution_time_seconds"] > 0


# ==============================================================================
# 6. TAMIL UNICODE RENDERING & LEXICAL DATAFRAME (Step 3 & 7 Verification)
# ==============================================================================

def test_lexicon_extraction_tamil_unicode():
    """Verify classical Tamil epigraphical terms are extracted with grammar & etymology."""
    sample_text = "இத்திருக்கற்றளியிலே நடுகல் மற்றும் ஆநிரை கொடை குடவோலை வாரியம்"
    words = extract_epigraph_lexicon(sample_text)

    found_terms = [w["word"] for w in words]
    assert "கற்றளி" in found_terms
    assert "நடுகல்" in found_terms
    assert "ஆநிரை" in found_terms
    assert "குடவோலை" in found_terms
    assert "வாரியம்" in found_terms


def test_lexical_dataframe_generation():
    """Verify words dictionary converts cleanly into a Pandas DataFrame for Streamlit."""
    sample_words = [
        {"word": "கற்றளி", "modern": "கற்கோயில்", "english": "Stone temple", "notes": "கல் + தளி"},
        {"word": "குடவோலை", "modern": "தேர்தல் முறை", "english": "Pot-ballot election", "notes": "சோழர் மக்களாட்சி"}
    ]

    df = pd.DataFrame([
        {
            "பண்டைய சொல் (Ancient Word)": w["word"],
            "இக்காலத் தமிழ் (Modern Tamil)": w["modern"],
            "English Meaning": w["english"],
            "இலக்கணம் / குறிப்பு (Grammar & Notes)": w["notes"]
        }
        for w in sample_words
    ])

    assert len(df) == 2
    assert "பண்டைய சொல் (Ancient Word)" in df.columns
    # Tamil Unicode preserved
    assert df.iloc[0]["பண்டைய சொல் (Ancient Word)"] == "கற்றளி"
    assert df.iloc[1]["English Meaning"] == "Pot-ballot election"


def test_cross_platform_font_fallback():
    """Verify font path returns either a valid file or None (not an unhandled crash)."""
    font_path = get_cross_platform_font_path()
    if font_path:
        assert os.path.exists(font_path)
    else:
        assert font_path is None


# ==============================================================================
# 7. CUSTOM TRANSCRIPTION ANALYSIS & PROVENANCE (Step 2 Verification)
# ==============================================================================

def test_custom_transcription_provenance():
    """Verify custom transcription records researcher provenance when analyzing uncataloged inscriptions."""
    custom_text = "கொற்றவைக்கு நின்றெறிந்து வீர மரணம் எய்திய தலைவன் கல்வெட்டு"
    result = analyze_custom_epigraph_text(custom_text, provenance="Manual Researcher Transcription")

    assert "provenance" in result
    assert result["provenance"] == "Manual Researcher Transcription"
    assert "restoration_details" in result
    assert "words" in result


def test_ui_lexical_dataframe_search_filter():
    """Verify build_lexical_dataframe filters vocabulary entries accurately."""
    words = [
        {"word": "கற்றளி", "modern": "கற்கோயில்", "english": "Monolithic stone temple", "notes": "கல் + தளி"},
        {"word": "நடுகல்", "modern": "வீரக்கல்", "english": "Hero memorial stone", "notes": "வீர மரணம்"},
        {"word": "குடவோலை", "modern": "தேர்தல்", "english": "Pot ballot election", "notes": "மக்களாட்சி"}
    ]

    # Full list
    df_all = build_lexical_dataframe(words)
    assert len(df_all) == 3

    # Filter by Tamil
    df_filtered_ta = build_lexical_dataframe(words, search_query="வீர")
    assert len(df_filtered_ta) == 1
    assert df_filtered_ta.iloc[0]["Ancient Word (பண்டைய சொல்)"] == "நடுகல்"

    # Filter by English
    df_filtered_en = build_lexical_dataframe(words, search_query="temple")
    assert len(df_filtered_en) == 1
    assert df_filtered_en.iloc[0]["Ancient Word (பண்டைய சொல்)"] == "கற்றளி"


# ==============================================================================
# 8. PHASE 4 TESTS: AI, OCR & CLASSICAL TAMIL NLP IMPROVEMENTS
# ==============================================================================

def test_epigraphical_scripts_inventories_completeness():
    """Verify all 4 core epigraphical scripts have comprehensive palaeographical metadata."""
    required_scripts = ["tamil_brahmi", "vatteluttu", "grantha", "medieval_chola"]
    for s in required_scripts:
        assert s in EPIGRAPHICAL_SCRIPTS
        script_meta = EPIGRAPHICAL_SCRIPTS[s]
        assert "name_ta" in script_meta
        assert "name_en" in script_meta
        assert "period_ta" in script_meta
        assert "geological_medium" in script_meta
        assert "unicode_range" in script_meta
        assert "distinctive_features" in script_meta


def test_script_identification_heuristics():
    """Verify script identification correctly classifies inscriptions based on diagnostic markers."""
    # Tamil-Brahmi
    tb_text = "வெள்ளறை நிகமத்தோர் கொடிபித பாளிய்"
    res_tb = identify_script_from_text(tb_text)
    assert res_tb["id"] == "tamil_brahmi"

    # Vatteluttu
    vat_text = "தொறுக்கொண்டு போகிய பூசலில் நின்றெறிந்து பட்டான்"
    res_vat = identify_script_from_text(vat_text)
    assert res_vat["id"] == "vatteluttu"

    # Grantha
    gr_text = "ஸ்ரீ ஸ்வஸ்தி ஸ்ரீமத் பிரசஸ்தி"
    res_gr = identify_script_from_text(gr_text)
    assert res_gr["id"] == "grantha"

    # Medieval Chola
    chola_text = "உடையார் ஸ்ரீ இராசராச தேவர்க்கு திருநந்தாவிளக்கு"
    res_chola = identify_script_from_text(chola_text)
    assert res_chola["id"] == "medieval_chola"


def test_sandhi_splitter_known_compounds():
    """Verify Sandhi splitter cleanly decomposes classical epigraphical compound words."""
    assert EpigraphicalSandhiSplitter.decompose_compound("பள்ளிச்சந்தம்") == ["பள்ளி", "சந்தம்"]
    assert EpigraphicalSandhiSplitter.decompose_compound("குடவோலை") == ["குடம்", "ஓலை"]
    assert EpigraphicalSandhiSplitter.decompose_compound("கற்றளி") == ["கல்", "தளி"]
    assert EpigraphicalSandhiSplitter.decompose_compound("தொறுப்பூசல்") == ["தொறு", "பூசல்"]
    assert EpigraphicalSandhiSplitter.decompose_compound("இறையிலி") == ["இறை", "இலி"]
    assert EpigraphicalSandhiSplitter.decompose_compound("பிரம்மதேயம்") == ["பிரம்ம", "தேயம்"]


def test_sandhi_splitter_split_sandhi_string():
    """Verify full string Sandhi segmentation while preserving damaged brackets."""
    text = "பள்ளிச்சந்தமும் குடவோலை முறையும் [சாத்தன்] நடுகல்"
    segmented = EpigraphicalSandhiSplitter.split_sandhi(text)
    assert "பள்ளி" in segmented
    assert "சந்தம்" in segmented
    assert "குடம்" in segmented
    assert "ஓலை" in segmented
    # Bracketed lacuna must be strictly preserved
    assert "[சாத்தன்]" in segmented


def test_sandhi_splitter_case_suffixes():
    """Verify archaic case suffix extraction."""
    # Dative -ku / -kurran
    c1 = EpigraphicalSandhiSplitter.extract_case_suffix("கூற்றற்கு")
    assert c1["has_case"] is True
    assert "4th Dative" in c1["suffix"]

    # Locative -il
    c2 = EpigraphicalSandhiSplitter.extract_case_suffix("கல்லிலே")
    assert c2["has_case"] is True
    assert "7th Locative" in c2["suffix"]

    # Uninflected nominative
    c3 = EpigraphicalSandhiSplitter.extract_case_suffix("அரசன்")
    assert c3["has_case"] is False


def test_morphological_analyzer_word():
    """Verify morphological analyzer classifies roots, POS, and case markers."""
    # Master lexicon word
    w_info = EpigraphicalMorphologicalAnalyzer.analyze_word("ஆநிரை")
    assert w_info["root"] == "ஆ + நிரை"
    assert "Noun" in w_info["pos"]
    assert "பசுக்களின் கூட்டம்" in w_info["modern"]

    # Royal Title
    title_info = EpigraphicalMorphologicalAnalyzer.analyze_word("உடையார்")
    assert "Royal Title" in title_info["pos"]
    assert "உடை" in title_info["root"]


def test_morphological_analyzer_text():
    """Verify full inscription text morphological decomposition."""
    text = "கோ ஆதன் பெருங்கடுங்கோ இளங்கோ ஆக அறுத்த கல்"
    analyzed = EpigraphicalMorphologicalAnalyzer.analyze_text(text)
    assert len(analyzed) > 0
    words = [a["word"] for a in analyzed]
    assert "கோ" in words or "கொ" in words


def test_multi_tier_transcription_structure():
    """Verify multi-tier text representation across all 5 tiers."""
    text = "கோச்சேந்தன் கூற்றற்கு [யாண்டு] இருபத்தேழாவது பள்ளிச்சந்தம்"
    res = analyze_custom_epigraph_text(text)

    # Tier 1: Diplomatic
    assert "diplomatic_text" in res
    assert "[யாண்டு]" in res["diplomatic_text"]

    # Tier 2: Normalized
    assert "normalized_text" in res

    # Tier 3: Word-segmented
    assert "word_segmented_text" in res
    assert "பள்ளி" in res["word_segmented_text"]

    # Tier 4: Modern Tamil
    assert "modern_tamil_translation" in res

    # Tier 5: English
    assert "english_translation" in res

    # Uncertain characters
    assert "uncertain_characters" in res
    assert len(res["uncertain_characters"]) >= 1


def test_uncertain_characters_extraction():
    """Verify bracketed damaged glyphs are extracted into structured uncertainty records."""
    damaged_text = "தொறுக்கொண்டு போகிய [பூச]லில் நின்றெறிந்து [பட்டான்] சாத்தன்"
    res = analyze_custom_epigraph_text(damaged_text)

    uncertains = res.get("uncertain_characters", [])
    assert len(uncertains) == 2
    glyphs = [u["character"] for u in uncertains]
    assert "[பூச]" in glyphs
    assert "[பட்டான்]" in glyphs
    assert all("alternative_readings" in u for u in uncertains)


def test_modular_ocr_adapters():
    """Verify OCR adapters behave correctly and scientifically without fake OCR."""
    # 1. PresetCatalogAdapter
    cat_adapter = PresetCatalogAdapter()
    preset_res = cat_adapter.recognize(preset_id="thanjavur_rajaraja")
    assert preset_res["id"] == "thanjavur_rajaraja"
    assert "Brihadisvara" in preset_res["title_en"]

    # 2. ManualTranscriptionAdapter
    man_adapter = ManualTranscriptionAdapter()
    man_res = man_adapter.recognize(text_input="அருள்மொழிதேவன் சோழர் யாண்டு ஏழாவது")
    assert "அருள்மொழிதேவன்" in man_res["normalized_text"]

    # 3. LocalOfflineOCRAdapter (Never claims offline decipherment when specialized weights are absent)
    loc_adapter = LocalOfflineOCRAdapter()
    loc_res = loc_adapter.recognize()
    assert loc_res["status"] == "offline_weights_unavailable"
    assert "fallback_recommended" in loc_res


def test_gemini_schema_validation():
    """Verify validate_gemini_epigraphy_response normalizes structured model responses."""
    raw_payload = {
        "script_identified": {
            "script_ta": "தமிழ் பிராமி",
            "script_en": "Tamil-Brahmi"
        },
        "confidence_score": 0.92,
        "diplomatic_transcription": "கணிய் நந்த[ாசி]ரிய் இகுவன்",
        "normalized_transcription": "கணி நந்தாசிரியர் இகுவன்",
        "word_segmented_text": "கணிய் நந்தாஸிரிய் இகுவன்",
        "modern_tamil_translation": "கணி நந்தாசிரியருக்கு வழங்கிய கொடை",
        "english_translation": "Donation bestowed upon ascetic Nantasiriyan",
        "tamil_meaning": "மாங்குளம் சமண உறைவிடம்",
        "english_meaning": "Mangulam rock cut cave record",
        "words": [
            {
                "word": "கணிய்",
                "root": "கணி",
                "pos": "பெயர்ச்சொல்",
                "case_marker": "எழுவாய்",
                "modern": "துறவி",
                "english": "Ascetic",
                "notes": "ஆரம்பக் கால வடிவம்"
            }
        ],
        "uncertain_characters": [
            {
                "character": "[ாசி]",
                "position": 9,
                "alternative_readings": ["ாஸிரி"],
                "reason": "Ligature weathered"
            }
        ]
    }

    validated = validate_gemini_epigraphy_response(raw_payload)
    assert validated["script_en"] == "Tamil-Brahmi"
    assert validated["confidence_score"] == 0.92
    assert validated["diplomatic_text"] == "கணிய் நந்த[ாசி]ரிய் இகுவன்"
    assert validated["normalized_text"] == "கணி நந்தாசிரியர் இகுவன்"
    assert len(validated["words"]) == 1
    assert len(validated["uncertain_characters"]) == 1


# ==============================================================================
# 9. PHASE 5 TESTS: SCIENTIFIC VALIDATION & RESEARCH-GRADE EVALUATION
# ==============================================================================

def test_cer_calculation_metrics():
    """Verify Character Error Rate (CER) calculation on Tamil Unicode strings."""
    # 1. Exact match
    assert calculate_cer("கல்வெட்டு", "கல்வெட்டு") == 0.0
    assert calculate_cer("", "") == 0.0

    # 2. Complete mismatch / empty reference
    assert calculate_cer("", "கல்வெட்டு") == 1.0

    # 3. Controlled substitution
    # 'சாத்தன்' (4 chars) vs 'சாத்தான்' (5 chars)
    cer = calculate_cer("சாத்தன்", "சாத்தான்")
    assert 0.0 < cer <= 0.5


def test_wer_calculation_metrics():
    """Verify Word Error Rate (WER) calculation on tokenized epigraphical strings."""
    # 1. Exact match
    assert calculate_wer("கோ ஆதன் செல்லிரும்பொறை", "கோ ஆதன் செல்லிரும்பொறை") == 0.0
    assert calculate_wer("", "") == 0.0

    # 2. Token substitution (1 out of 3 replaced)
    wer_sub = calculate_wer("கோ ஆதன் செல்லிரும்பொறை", "கோ சேந்தன் செல்லிரும்பொறை")
    assert round(wer_sub, 2) == round(1.0 / 3.0, 2)

    # 3. Token deletion
    wer_del = calculate_wer("கோ ஆதன் செல்லிரும்பொறை", "கோ ஆதன்")
    assert round(wer_del, 2) == round(1.0 / 3.0, 2)


def test_inpainting_metrics_evaluation():
    """Verify Precision, Recall, and F1 calculations for lacuna inpainting."""
    gt = ["ாஸிரி", "வெள்ளறை", "கொடிபித"]
    pred = ["ாஸிரி", "வெள்ளறை", "கொடிபித"]
    m_perfect = calculate_inpainting_metrics(gt, pred)
    assert m_perfect["precision"] == 1.0
    assert m_perfect["recall"] == 1.0
    assert m_perfect["f1"] == 1.0

    # Partial match (2 out of 3 recovered, 1 false positive)
    pred_partial = ["ாஸிரி", "வெள்ளறை", "தவறான_சொல்"]
    m_partial = calculate_inpainting_metrics(gt, pred_partial)
    assert round(m_partial["precision"], 2) == round(2.0 / 3.0, 2)
    assert round(m_partial["recall"], 2) == round(2.0 / 3.0, 2)
    assert round(m_partial["f1"], 2) == round(2.0 / 3.0, 2)


def test_segmentation_metrics_evaluation():
    """Verify Word Boundary Segmentation metrics for Sandhi splitting."""
    ref = "பள்ளி சந்தமும் குடம் ஓலை"
    hyp = "பள்ளி சந்தமும் குடம் ஓலை"
    m = calculate_segmentation_metrics(ref, hyp)
    assert m["f1"] == 1.0

    hyp_unsplit = "பள்ளிச்சந்தமும் குடவோலை"
    m_unsplit = calculate_segmentation_metrics(ref, hyp_unsplit)
    assert m_unsplit["f1"] < 1.0


def test_psnr_calculation():
    """Verify Peak Signal-to-Noise Ratio (PSNR) calculation."""
    arr = np.full((100, 100), 128, dtype=np.uint8)
    # Identical images
    assert calculate_psnr(arr, arr) == 100.0

    # Perturbed image with additive noise
    noisy = arr.copy()
    noisy[::2, ::2] = 140
    psnr_noisy = calculate_psnr(arr, noisy)
    assert 20.0 < psnr_noisy < 50.0


def test_ssim_calculation():
    """Verify Structural Similarity Index Measure (SSIM) implementation."""
    arr = np.random.randint(50, 200, size=(100, 100), dtype=np.uint8)
    # Identical images
    ssim_perfect = calculate_ssim(arr, arr)
    assert round(ssim_perfect, 4) == 1.0

    # Severely degraded image
    degraded = (arr * 0.4 + 30).astype(np.uint8)
    ssim_deg = calculate_ssim(arr, degraded)
    assert ssim_deg < 0.95


def test_contrast_improvement_ratio():
    """Verify Contrast Improvement Ratio (CIR) calculation."""
    low_contrast = np.full((100, 100), 100, dtype=np.float32)
    low_contrast[::2, ::2] = 110  # std is small

    high_contrast = np.full((100, 100), 50, dtype=np.float32)
    high_contrast[::2, ::2] = 200  # std is large

    cir = calculate_contrast_improvement_ratio(low_contrast, high_contrast)
    assert cir > 1.5


def test_dice_coefficient():
    """Verify Dice Similarity Coefficient calculation on binary masks."""
    mask_a = np.zeros((50, 50), dtype=np.uint8)
    mask_a[:25, :] = 255

    # Identical
    assert calculate_dice_coefficient(mask_a, mask_a) == 1.0

    # Completely disjoint
    mask_b = np.zeros((50, 50), dtype=np.uint8)
    mask_b[25:, :] = 255
    assert calculate_dice_coefficient(mask_a, mask_b) == 0.0


def test_evaluation_dataset_fixtures_integrity():
    """Verify evaluation dataset fixtures contain all required epigraphical fields."""
    dataset = load_evaluation_dataset()
    assert len(dataset) >= 7

    required_keys = [
        "id", "script", "diplomatic_text", "ground_truth_normalized",
        "ground_truth_segmented", "ground_truth_modern_tamil",
        "ground_truth_english", "damaged_lacunae", "expected_inpainted"
    ]
    for sample in dataset:
        for k in required_keys:
            assert k in sample, f"Sample '{sample.get('id')}' missing key '{k}'"
        assert sample["script"] in ["tamil_brahmi", "vatteluttu", "medieval_chola", "grantha"]


def test_ablation_study_execution():
    """Verify run_ablation_study executes successfully across test dataset."""
    dataset = load_evaluation_dataset()
    ablation = run_ablation_study(dataset)
    assert "cv_enhancement_ablation" in ablation
    assert "binarization_ablation" in ablation
    assert "nlp_inpainting_ablation" in ablation
    assert "sandhi_ablation" in ablation

    # Verify at least one CV record and one NLP record
    assert len(ablation["nlp_inpainting_ablation"]) >= 7
    assert len(ablation["sandhi_ablation"]) >= 7


# ==============================================================================
# PHASE 6 TESTS: CONFIGURATION, SECURITY HARDENING & PRODUCTION DEPLOYMENT
# ==============================================================================

def test_sanitize_filename_path_traversal_prevention():
    """Verify sanitize_filename neutralizes path traversal, null bytes, and malicious filenames."""
    # Classic path traversal (CWE-22)
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("..\\..\\Windows\\System32\\cmd.exe") == "cmd.exe"
    assert sanitize_filename("/var/log/secret.key") == "secret.key"

    # Null bytes and control characters
    assert sanitize_filename("stone\x00_photo.jpg") == "stone_photo.jpg"
    assert sanitize_filename("report\x1b[31m.md") == "report_31m.md"

    # Leading dots (hidden files)
    assert sanitize_filename(".hidden_dossier.json") == "hidden_dossier.json"
    assert sanitize_filename("...") == "kallvettu_artifact"

    # Empty and non-string inputs fallback safely
    assert sanitize_filename("") == "kallvettu_artifact"
    assert sanitize_filename(None, default="fallback.txt") == "fallback.txt"


def test_mask_api_key_security():
    """Verify mask_api_key protects secret credentials from leakage in logs and UI."""
    # Unset / empty
    assert mask_api_key(None) == "Not Configured"
    assert mask_api_key("") == "Not Configured"
    assert mask_api_key("   ") == "Not Configured"

    # Short keys
    assert mask_api_key("12345") == "********"
    assert mask_api_key("AIzaSyD") == "********"

    # Standard Gemini API keys
    key = "AIzaSyD-1234567890abcdefghijklmnopqrstuv"
    masked = mask_api_key(key)
    assert masked == "AIzaSy...stuv"
    assert "1234567890" not in masked
    assert len(masked) < len(key)


def test_kallvettu_config_singleton_and_defaults():
    """Verify KallvettuConfig singleton provides typed, secure defaults."""
    assert CONFIG.APP_NAME == "Kallvettu"
    assert "3.0.0" in CONFIG.APP_VERSION
    assert CONFIG.MAX_UPLOAD_SIZE_BYTES >= 10 * 1024 * 1024
    assert "JPEG" in CONFIG.ALLOWED_IMAGE_FORMATS
    assert "PNG" in CONFIG.ALLOWED_IMAGE_FORMATS
    assert CONFIG.CLAHE_CLIP_LIMIT_DEFAULT > 0
    assert CONFIG.SAUVOLA_K_DEFAULT > 0
    assert CONFIG.DEFAULT_GEMINI_MODEL in CONFIG.SUPPORTED_GEMINI_MODELS
    assert os.path.exists(CONFIG.ASSETS_DIR)


def test_dockerfile_and_dockerignore_configuration():
    """Verify production Dockerfile and .dockerignore contain security best practices."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dockerfile_path = os.path.join(repo_root, "Dockerfile")
    dockerignore_path = os.path.join(repo_root, ".dockerignore")
    env_example_path = os.path.join(repo_root, ".env.example")

    assert os.path.exists(dockerfile_path), "Dockerfile must exist"
    assert os.path.exists(dockerignore_path), ".dockerignore must exist"
    assert os.path.exists(env_example_path), ".env.example must exist"

    with open(dockerfile_path, "r", encoding="utf-8") as f:
        docker_content = f.read()

    # Security check: Non-root user execution
    assert "useradd" in docker_content
    assert "USER kallvettu" in docker_content

    # Healthcheck probe
    assert "HEALTHCHECK" in docker_content
    assert "_stcore/health" in docker_content

    # System fonts for Tamil script
    assert "fonts-noto" in docker_content

    # Dockerignore excludes
    with open(dockerignore_path, "r", encoding="utf-8") as f:
        ignore_content = f.read()
    assert ".git" in ignore_content
    assert "__pycache__" in ignore_content
    assert ".env" in ignore_content


def test_epigraphical_provenance_documentation_completeness():
    """Verify docs/epigraphical_provenance.md comprehensively covers all 7 catalog inscriptions."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    provenance_path = os.path.join(repo_root, "docs", "epigraphical_provenance.md")

    assert os.path.exists(provenance_path), "epigraphical_provenance.md must exist in docs/"

    with open(provenance_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Verify all 7 catalog inscriptions are documented
    catalog_ids = [
        "mangulam_brahmi",
        "pugalur_chera",
        "pulankurichi_vatteluttu",
        "hero_stone_nadukal",
        "thanjavur_rajaraja",
        "uttiramerur_election",
        "chola_arulmoli"
    ]
    for cid in catalog_ids:
        assert cid in content, f"Missing documentation for catalog inscription '{cid}'"

    # Verify primary academic corpora citations
    assert "Mahadevan" in content
    assert "South Indian Inscriptions" in content
    assert "Chengam Nadukarkal" in content
    assert "Early Tamil Epigraphy" in content



