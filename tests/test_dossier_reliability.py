"""
Automated Regression Test Suite for Epigraphical Research Dossier Reliability
and Anti-Hallucination Pipeline Controls in Kallvettu.

Covers:
1. Provenance separation (Raw OCR, Multi-engine comparison, Fused, Researcher transcribed, Researcher approved, AI suggested).
2. Script and historical classification separation (Script family, variant, language, period, dynasty, monarch).
3. Diplomatic and normalized transcription separation with editorial operations audit.
4. Word segmentation tier separation and unverified boundary marking.
5. Translation safeguards against hallucinated endowments/grants for unverified OCR.
6. Lexical and morphological anti-hallucination (blocking fake nouns and nominative case).
7. Inter-engine consensus and confidence metrics (0% agreement is not consensus, single engine handling).
8. Dossier generation consistency across Markdown, JSON, and EpiDoc XML.
9. Null, whitespace, and boundary conditions.
10. Regression fixture for the exact problematic unverified OCR sequence ("கபழற தமடச லயன").
"""

import io
import json
import zipfile
import pytest

from services.ai.common.base_provider import OCRResult
from services.ai.common.comparison_engine import (
    MultiEngineOCRComparator,
    DisputedToken,
    MultiOCRComparisonResult,
)
from services.ocr_fusion.cross_verifier import (
    EpigraphicalCrossVerifier,
    CrossVerificationReport,
)
from services.epidoc.bundle import create_epidoc_research_bundle
from services.epidoc.validator import validate_epidoc_xml
from nlp_translation_engine import (
    EPIGRAPHICAL_SCRIPTS,
    identify_script_from_text,
    EpigraphicalSandhiSplitter,
    EpigraphicalMorphologicalAnalyzer,
    analyze_custom_epigraph_text,
    TRANSLATION_UNAVAILABLE_UNVERIFIED,
)
from app import generate_markdown_report, generate_epidoc_xml


# ==============================================================================
# 1. PROVENANCE SEPARATION TESTS
# ==============================================================================

def test_provenance_separation_ai_vs_researcher():
    """Verify that AI-generated transcriptions are never labeled as researcher-transcribed."""
    raw_ocr = "கோ ஆதன் செல்லிரும்பொறை"

    # Case A: AI-generated suggestion
    analysis_ai = analyze_custom_epigraph_text(
        raw_ocr,
        provenance="Google Gemini Vision AI",
        provenance_type="AI_GENERATED_SUGGESTION",
        is_researcher_approved=False,
        allow_preset_match=False
    )
    assert analysis_ai["provenance_type"] == "AI_GENERATED_SUGGESTION"
    assert analysis_ai["dossier_status"] == "UNVERIFIED"
    assert "Researcher Transcribed" not in analysis_ai["title_en"]
    assert "AI-Generated Inscription Suggestion" in analysis_ai["title_en"]

    # Case B: Researcher verified transcription
    analysis_researcher = analyze_custom_epigraph_text(
        raw_ocr,
        provenance="Researcher Epigraphist Review",
        provenance_type="RESEARCHER_APPROVED",
        is_researcher_approved=True,
        allow_preset_match=False
    )
    assert analysis_researcher["provenance_type"] == "RESEARCHER_APPROVED"
    assert analysis_researcher["dossier_status"] == "APPROVED"
    assert "Researcher" in analysis_researcher["title_en"]
    assert "Researcher" not in analysis_ai["title_en"]


# ==============================================================================
# 2. SCRIPT & HISTORICAL CLASSIFICATION SEPARATION
# ==============================================================================

def test_script_classification_undetermined_for_uncertain_text():
    """Verify that uncertain/unattested text does not infer Pallava/Chola or date ranges."""
    uncertain_text = "கபழற தமடச லயன"
    script_dict = identify_script_from_text(uncertain_text)
    assert script_dict["id"] == "undetermined"

    script_meta = EPIGRAPHICAL_SCRIPTS["undetermined"]
    assert script_meta["script_family"] == "UNDETERMINED"
    assert script_meta["script_variant"] == "UNDETERMINED"
    assert script_meta["dynasty"] == "UNDETERMINED"
    assert "UNDETERMINED" in script_meta["historical_period"]
    assert script_meta["monarch"] == "UNDETERMINED"
    assert script_meta["geographical_provenance"] == "UNDETERMINED"
    assert script_meta["material_type"] == "UNDETERMINED"

    analysis = analyze_custom_epigraph_text(
        uncertain_text,
        provenance="Raw Provider OCR",
        is_researcher_approved=False,
        allow_preset_match=False
    )
    assert analysis["script_key"] == "undetermined"
    assert analysis["dynasty_en"] == "UNDETERMINED"
    assert "UNDETERMINED" in analysis["period_en"]
    assert analysis["monarch"] == "UNDETERMINED"


def test_script_classification_preserves_separated_fields_for_known_scripts():
    """Verify that canonical scripts contain all separated classification metadata."""
    for s_key in ["tamil_brahmi", "vatteluttu", "grantha", "medieval_chola"]:
        meta = EPIGRAPHICAL_SCRIPTS[s_key]
        assert "script_family" in meta
        assert "script_variant" in meta
        assert "language" in meta
        assert "historical_period" in meta
        assert "dynasty" in meta
        assert "monarch" in meta
        assert "geographical_provenance" in meta
        assert "material_type" in meta


# ==============================================================================
# 3. DIPLOMATIC VS NORMALIZED TRANSCRIPTION SEPARATION
# ==============================================================================

def test_diplomatic_and_normalized_transcription_separation():
    """Verify Tier 1 preserves raw brackets and marks while Tier 2 records normalization operations."""
    text_with_lacunae = "கோ ஆதன் [செல்லிரும்]பொறை மகன்"

    analysis = analyze_custom_epigraph_text(
        text_with_lacunae,
        is_researcher_approved=False,
        allow_preset_match=False
    )

    # Tier 1 Diplomatic must preserve raw text verbatim
    assert analysis["diplomatic_text"] == text_with_lacunae
    assert "[" in analysis["diplomatic_text"] and "]" in analysis["diplomatic_text"]

    # Tier 2 Normalized
    assert "normalized_text" in analysis
    assert "normalization_operations" in analysis
    assert isinstance(analysis["normalization_operations"], list)


def test_identical_raw_and_normalized_flagged_as_unverified():
    """Verify that when raw text has no normalization operations, status is UNVERIFIED and not faked."""
    raw = "அறையன் பள்ளி"
    analysis = analyze_custom_epigraph_text(
        raw,
        is_researcher_approved=False,
        allow_preset_match=False
    )
    assert analysis["normalization_status"] == "UNVERIFIED"
    assert len(analysis["normalization_operations"]) == 0


# ==============================================================================
# 4. WORD SEGMENTATION TIER SEPARATION
# ==============================================================================

def test_word_segmentation_tier_separation():
    """Verify that word segmentation is a distinct tier and distinguishes unsegmented raw sequences."""
    splitter = EpigraphicalSandhiSplitter()

    # Unsegmented single continuous block
    unsegmented_raw = "கபழறதமடசலயன"
    seg_raw = splitter.segment_words(unsegmented_raw, researcher_approved=False)
    assert seg_raw["is_segmented"] is False
    assert seg_raw["segmentation_type"] == "unsegmented_raw"
    assert seg_raw["verification_status"] == "UNVERIFIED"

    # Space-delimited unverified tokens
    spaced_raw = "கோ ஆதன் செல்லிரும்பொறை மகன்"
    seg_spaced = splitter.segment_words(spaced_raw, researcher_approved=False)
    assert seg_spaced["is_segmented"] is True
    assert seg_spaced["segmentation_type"] == "raw_whitespace_tokens"
    assert seg_spaced["verification_status"] == "PROVISIONAL"

    # Researcher approved segmentation
    seg_approved = splitter.segment_words(spaced_raw, researcher_approved=True)
    assert seg_approved["verification_status"] == "APPROVED"


# ==============================================================================
# 5. TRANSLATION SAFEGUARDS
# ==============================================================================

def test_translation_safeguards_block_hallucinated_endowments():
    """Verify that unverified OCR never generates fake endowment or temple grant claims."""
    unverified_ocr = "கபழற தமடச லயன"

    analysis = analyze_custom_epigraph_text(
        unverified_ocr,
        is_researcher_approved=False,
        allow_preset_match=False
    )

    assert analysis["is_translation_available"] is False
    assert "மொழிபெயர்ப்பு கிடைக்கப்பெறவில்லை" in analysis["tamil_translation"]
    assert analysis["english_translation"] == TRANSLATION_UNAVAILABLE_UNVERIFIED

    # Check that forbidden hallucinated terms do not appear
    forbidden_terms = [
        "historical endowment",
        "royal donation",
        "administrative record",
        "heroic tradition",
        "temple grant",
        "நிலக்கொடை",
        "கோயில் கொடை",
        "நிவந்தம்",
    ]
    for term in forbidden_terms:
        assert term.lower() not in analysis["tamil_translation"].lower()
        assert term.lower() not in analysis["english_translation"].lower()
        assert term.lower() not in analysis["historical_context"].lower()

    assert "interpretation withheld" in analysis["historical_context"].lower()


def test_approved_transcription_enables_translation():
    """Verify that researcher-approved transcription allows scholarly translation."""
    approved_text = "ஸ்வஸ்திஸ்ரீ ராஜராஜ சோழதேவர்க்கு யாண்டு இருபத்தாறாவது"
    analysis = analyze_custom_epigraph_text(
        approved_text,
        is_researcher_approved=True,
        allow_preset_match=False
    )
    assert analysis["is_translation_available"] is True
    assert analysis["tamil_translation"] != TRANSLATION_UNAVAILABLE_UNVERIFIED
    assert analysis["english_translation"] != TRANSLATION_UNAVAILABLE_UNVERIFIED


# ==============================================================================
# 6. MORPHOLOGICAL ANALYSIS ANTI-HALLUCINATION
# ==============================================================================

def test_morphological_analysis_unrecognized_tokens():
    """Verify that unknown/corrupted tokens return unrecognized status without faked nouns."""
    analyzer = EpigraphicalMorphologicalAnalyzer()

    # Corrupted OCR token
    res = analyzer.analyze_word("கபழற")
    assert res["is_recognized"] is False
    assert res["pos_category"] == "UNRECOGNIZED"
    assert res["case_marker"] == "UNDETERMINED"
    assert "Unrecognized lexical sequence — requires epigraphist review." in res["meaning_ta"]
    assert "Unrecognized lexical sequence — requires epigraphist review." in res["meaning_en"]
    assert res["review_status"] == "REQUIRES_EPIGRAPHIST_REVIEW"


def test_morphological_analysis_attested_tokens():
    """Verify that attested epigraphical terms return authenticated dictionary evidence."""
    analyzer = EpigraphicalMorphologicalAnalyzer()

    res = analyzer.analyze_word("அறையன்")
    assert res["is_recognized"] is True
    assert "பெயர்ச்சொல்" in res["pos"]
    assert res["proposed_root"] == "அறை"
    assert "Tamil Lexicon" in res["dictionary_source"]
    assert res["review_status"] in ("APPROVED", "ATTESTED")


# ==============================================================================
# 7. INTER-ENGINE CONSENSUS & CONFIDENCE METRICS
# ==============================================================================

def test_zero_percent_agreement_not_described_as_consensus():
    """Verify that 0% inter-engine agreement sets ratio to 0.0 and zero_agreement status."""
    comparator = MultiEngineOCRComparator()

    # Two completely divergent readings with 0 agreed tokens
    res_a = OCRResult(text="சாத்தன் நடுகல்", provider_name="A", model_id="m1")
    res_b = OCRResult(text="கபழற தமடச", provider_name="B", model_id="m2")

    results_dict = {"prov_a": res_a, "prov_b": res_b}

    (
        disputed,
        ratio,
        total,
        agreed,
        disp_count,
        default_app,
        comp_status,
        insufficient_msg,
    ) = comparator._analyze_disagreements(results_dict)

    assert agreed == 0
    assert ratio == 0.0
    assert comp_status == "zero_agreement"
    assert default_app == "[DISPUTED - ZERO INTER-ENGINE AGREEMENT]"

    comp_result = MultiOCRComparisonResult(
        provider_results=results_dict,
        disputed_tokens=disputed,
        consensus_ratio=ratio,
        total_tokens_evaluated=total,
        agreed_tokens_count=agreed,
        disputed_tokens_count=disp_count,
        approved_transcription=default_app,
        comparison_status=comp_status,
    )
    # Consensus text must be empty when there is zero agreement
    assert comp_result.consensus_text == ""


def test_single_provider_agreement_ratio_is_zero():
    """Verify that a single provider output yields 0.0 inter-engine agreement, not 1.0."""
    verifier = EpigraphicalCrossVerifier()
    res = OCRResult(text="ஸ்வஸ்திஸ்ரீ ராஜராஜ", provider_name="Paddle", model_id="v5", confidence=0.91)

    rep = verifier.verify({"paddle": res})
    assert rep.agreement_ratio == 0.0
    assert rep.agreed_character_slots == 0
    assert rep.disputed_character_slots == 0


def test_disputed_token_agreement_ratio():
    """Verify DisputedToken agreement_ratio excludes empty values and handles ties."""
    # When readings are identical
    dt_same = DisputedToken(
        token_index=0,
        readings={"p1": "கோ", "p2": "கோ", "p3": "கோ"},
        is_disputed=False
    )
    assert dt_same.agreement_ratio == 1.0

    # When all 3 readings are completely different
    dt_diff = DisputedToken(
        token_index=0,
        readings={"p1": "கோ", "p2": "ஆதன்", "p3": "மகன்"},
        is_disputed=True
    )
    assert dt_diff.agreement_ratio == 0.0

    # When 1 is empty or single provider
    dt_single = DisputedToken(
        token_index=0,
        readings={"p1": "கோ", "p2": ""},
        is_disputed=True
    )
    assert dt_single.agreement_ratio == 0.0


# ==============================================================================
# 8. DOSSIER & EPIDOC EXPORT CONSISTENCY
# ==============================================================================

def test_markdown_report_contains_explicit_statuses():
    """Verify generate_markdown_report generates explicit statuses and separated metadata."""
    analysis = analyze_custom_epigraph_text(
        "கபழற தமடச லயன",
        provenance="Raw Provider OCR",
        is_researcher_approved=False,
        allow_preset_match=False
    )

    md = generate_markdown_report(analysis)
    assert "Scientific Dossier Status:" in md
    assert "`UNVERIFIED`" in md
    assert "Transcription Provenance:" in md
    assert "Tier 1: Diplomatic" in md
    assert "Tier 2: Normalized" in md
    assert "Tier 3: Word-Segmented" in md
    assert "Translation unavailable: the underlying transcription requires epigraphical verification." in md


def test_epidoc_xml_contains_explicit_status_and_metadata():
    """Verify generate_epidoc_xml includes separated metadata without hallucinated claims."""
    analysis = analyze_custom_epigraph_text(
        "கபழற தமடச லயன",
        provenance="Raw Provider OCR",
        is_researcher_approved=False,
        allow_preset_match=False
    )

    xml_str = generate_epidoc_xml(analysis)
    assert "<?xml version=" in xml_str
    assert "<TEI" in xml_str
    assert "status=\"unverified\"" in xml_str
    assert "Translation unavailable: the underlying transcription requires epigraphical verification." in xml_str


def test_epidoc_research_bundle_zip_contains_all_files():
    """Verify create_epidoc_research_bundle generates 5 files with separated metadata and safeguards."""
    bundle_data = {
        "id": "KV-INSC-REGRESSION-01",
        "title_en": "Unverified Lithic Fragment",
        "title_ta": "சரிபார்க்கப்படாத கல்வெட்டுத் துண்டு",
        "script_category": "Undetermined",
        "diplomatic_text": "கபழற தமடச லயன",
        "normalized_text": "கபழற தமடச லயன",
        "is_researcher_approved": False,
        "dossier_status": "UNVERIFIED",
        "provenance_type": "RAW_PROVIDER_OCR",
    }

    zip_bytes = create_epidoc_research_bundle(bundle_data)
    assert len(zip_bytes) > 200

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        namelist = zf.namelist()
        assert "KV-INSC-REGRESSION-01/inscription.xml" in namelist
        assert "KV-INSC-REGRESSION-01/transcription.txt" in namelist
        assert "KV-INSC-REGRESSION-01/translation.txt" in namelist
        assert "KV-INSC-REGRESSION-01/inscription.json" in namelist
        assert "KV-INSC-REGRESSION-01/research_report.md" in namelist

        # Inspect json
        j_data = json.loads(zf.read("KV-INSC-REGRESSION-01/inscription.json").decode("utf-8"))
        assert j_data["dossier_status"] == "UNVERIFIED"
        assert j_data["provenance_type"] == "RAW_PROVIDER_OCR"
        assert "Translation unavailable" in j_data["translations"]["english"]

        # Inspect report
        md_text = zf.read("KV-INSC-REGRESSION-01/research_report.md").decode("utf-8")
        assert "`UNVERIFIED`" in md_text
        assert "Translation unavailable" in md_text


# ==============================================================================
# 9. NULL, WHITESPACE, AND BOUNDARY CONDITIONS
# ==============================================================================

def test_null_and_boundary_conditions():
    """Verify graceful handling of empty strings, whitespace, and single characters."""
    # Empty string
    analysis_empty = analyze_custom_epigraph_text("", allow_preset_match=False)
    assert analysis_empty["diplomatic_text"] == ""
    assert analysis_empty["dossier_status"] == "UNVERIFIED"

    # Whitespace only
    analysis_ws = analyze_custom_epigraph_text("   \n\t  ", allow_preset_match=False)
    assert analysis_ws["dossier_status"] == "UNVERIFIED"

    # Single character
    analysis_single = analyze_custom_epigraph_text("க", provenance_type="RAW_PROVIDER_OCR", allow_preset_match=False)
    assert analysis_single["dossier_status"] == "UNVERIFIED"
    assert analysis_single["script_key"] == "undetermined"


# ==============================================================================
# 10. EXACT PROBLEMATIC EXAMPLE REGRESSION FIXTURE
# ==============================================================================

def test_exact_problematic_ocr_regression_fixture():
    """
    Regression Fixture for the exact problematic unverified OCR sequence:
    'கபழற தமடச லயன'

    Prior to the fix:
    - Defaulted to 'Medieval Chola'
    - Inferred 'Chola Dynasty'
    - Hallucinated temple grant / royal donation translation
    - Classified corrupted strings as Noun / Nominative
    - Claimed consensus without provider agreement

    Under the corrected pipeline:
    - Script is 'undetermined' (Undetermined Inscriptional Script)
    - Dynasty, monarch, historical period are 'UNDETERMINED'
    - Translation is withheld with scholarly verification notice
    - Morphology returns unrecognized lexical sequence
    - Dossier status is explicitly UNVERIFIED
    """
    raw_sequence = "கபழற தமடச லயன"

    analysis = analyze_custom_epigraph_text(
        raw_sequence,
        provenance="Raw Provider OCR (Gemini Vision AI)",
        provenance_type="RAW_PROVIDER_OCR",
        is_researcher_approved=False,
        allow_preset_match=False
    )

    # 1. Script & Period & Dynasty separation
    assert analysis["script_key"] == "undetermined"
    assert analysis["dynasty_en"] == "UNDETERMINED"
    assert "UNDETERMINED" in analysis["period_en"]
    assert analysis["monarch"] == "UNDETERMINED"

    # 2. Status & Provenance
    assert analysis["dossier_status"] == "UNVERIFIED"
    assert analysis["provenance_type"] == "RAW_PROVIDER_OCR"
    assert analysis["is_researcher_approved"] is False

    # 3. Translation blocked
    assert analysis["is_translation_available"] is False
    assert "மொழிபெயர்ப்பு கிடைக்கப்பெறவில்லை" in analysis["tamil_translation"]
    assert analysis["english_translation"] == TRANSLATION_UNAVAILABLE_UNVERIFIED

    # 4. Morphology anti-hallucination
    for m in analysis["morphological_analysis"]:
        assert m["is_recognized"] is False
        assert m["pos_category"] == "UNRECOGNIZED"
        assert m["case_marker"] == "UNDETERMINED"
        assert "Unrecognized lexical sequence" in m["meaning_ta"]
        assert "Unrecognized lexical sequence" in m["meaning_en"]

    # 5. Markdown Report Verification
    md_report = generate_markdown_report(analysis)
    assert "**Scientific Dossier Status:** `UNVERIFIED`" in md_report
    assert "**Transcription Provenance:** `RAW_PROVIDER_OCR`" in md_report
    assert "Translation unavailable: the underlying transcription requires epigraphical verification." in md_report
    assert "Chola" not in md_report or "UNDETERMINED" in md_report
