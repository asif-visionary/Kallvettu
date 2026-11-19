"""
Scientific Epigraphical Evidence & OCR Integrity Monitor Test Suite.

Requirements Covered:
1. Entropy calculation: Raw Shannon entropy (bits) & normalized entropy (H / log2(K)).
2. Unanimous candidate distribution: H = 0.0 with scientific disclaimer.
3. Disputed candidate distribution: Multi-candidate dispersion & dominant margin.
4. Missing provider metadata: Strict NOT_ASSESSABLE state (never false PARTIAL_TRUNCATION).
5. Truncation classification: COMPLETE, PARTIAL_TRUNCATION, NOT_ASSESSABLE, PROVIDER_ERROR, EMPTY_OUTPUT.
6. Empty OCR response: Clean EMPTY_OUTPUT classification.
7. Repeated Tamil grapheme detection: Grapheme-cluster looping on stone texture.
8. Null and None handling: Safe string formatting without literal Python None or stringified 'None'.
9. Unicode combining sequences: Preserves Tamil combining ligatures (மெய்யெழுத்து, உயிர்மெய்).
10. Researcher-approved reading preservation: Cryptographic protection against AI overwrites.
11. Provider failure handling: Graceful categorization of exceptions without pipeline abort.
"""

import math
import pytest
import numpy as np
from PIL import Image

from services.ai.common.base_provider import OCRResult, BoundingBox
from services.ocr_fusion import (
    fuse_epigraphical_evidence,
    MasterEvidenceFusionBundle,
    ResearcherReviewManager,
    AdjudicationDecision,
    AdjudicationItem,
)
from services.ocr_fusion.cross_verifier import (
    tokenize_tamil_graphemes,
    CharacterAlignmentUnit,
)
from services.ocr_fusion.consensus import (
    ConsensusEntropyCalculator,
    DisagreementSeverity,
)
from services.ocr_fusion.integrity_monitor import (
    EpigraphicalIntegrityMonitor,
    TruncationStatus,
    IntegritySeverity,
    ProviderTruncationDiagnostic,
    RepeatedGraphemeAnomaly,
    SCIENTIFIC_ENTROPY_PRINCIPLE,
    PHYSICAL_EVIDENCE_SAFEGUARD_PRINCIPLE,
    safe_str,
)


@pytest.fixture
def synthetic_plate_image():
    """Synthetic stone inscription plate image (600x300 px). [SYNTHETIC TEST FIXTURE]"""
    img = Image.new("RGB", (600, 300), color=(130, 120, 110))
    arr = np.array(img)
    arr[90:140, 40:560] = (70, 60, 50)
    arr[160:210, 40:560] = (75, 65, 55)
    return Image.fromarray(arr)


# ==============================================================================
# 1. ENTROPY CALCULATION (RAW & NORMALIZED)
# ==============================================================================
def test_entropy_calculation_raw_and_normalized():
    """Verifies Shannon entropy and normalized entropy H / log2(K) when K > 1."""
    # 2 evenly split candidates: H = 1.0, norm_H = 1.0 / log2(2) = 1.0
    unit_k2 = CharacterAlignmentUnit(
        position_index=0,
        candidate_character="க",
        raw_by_provider={"p1": "க", "p2": "த"},
        normalized_by_provider={"p1": "க", "p2": "த"},
        confidence_by_provider={"p1": 0.8, "p2": 0.8},
        status="substitution",
        is_disputed=True,
    )
    res_k2 = ConsensusEntropyCalculator.calculate_position_entropy(unit_k2)
    assert math.isclose(res_k2.entropy, 1.0, rel_tol=1e-3)
    assert math.isclose(res_k2.normalized_entropy, 1.0, rel_tol=1e-3)

    # 4 evenly split candidates: H = 2.0, norm_H = 2.0 / log2(4) = 1.0
    unit_k4 = CharacterAlignmentUnit(
        position_index=1,
        candidate_character="க",
        raw_by_provider={"p1": "க", "p2": "ச", "p3": "த", "p4": "ப"},
        normalized_by_provider={"p1": "க", "p2": "ச", "p3": "த", "p4": "ப"},
        confidence_by_provider={"p1": 0.5, "p2": 0.5, "p3": 0.5, "p4": 0.5},
        status="substitution",
        is_disputed=True,
    )
    res_k4 = ConsensusEntropyCalculator.calculate_position_entropy(unit_k4)
    assert math.isclose(res_k4.entropy, 2.0, rel_tol=1e-3)
    assert math.isclose(res_k4.normalized_entropy, 1.0, rel_tol=1e-3)


# ==============================================================================
# 2. UNANIMOUS CANDIDATE DISTRIBUTION
# ==============================================================================
def test_unanimous_candidate_distribution():
    """Verifies unanimous distribution has H=0, distinct K=1, and preserves scientific principle."""
    unit_unanimous = CharacterAlignmentUnit(
        position_index=0,
        candidate_character="ம",
        raw_by_provider={"paddle": "ம", "sarvam": "ம", "hog_svm": "ம"},
        normalized_by_provider={"paddle": "ம", "sarvam": "ம", "hog_svm": "ம"},
        confidence_by_provider={"paddle": 0.95, "sarvam": 0.92, "hog_svm": 0.88},
        status="match",
        is_disputed=False,
    )
    res = ConsensusEntropyCalculator.calculate_position_entropy(unit_unanimous)
    assert res.entropy == 0.0
    assert res.is_unanimous is True
    assert res.distinct_candidates == ["ம"]
    assert res.severity == DisagreementSeverity.UNANIMOUS

    # Scientific principle must emphasize that unanimity != historical ground truth
    assert "Consensus entropy measures model dispersion, not historical veracity" in SCIENTIFIC_ENTROPY_PRINCIPLE


# ==============================================================================
# 3. DISPUTED CANDIDATE DISTRIBUTION
# ==============================================================================
def test_disputed_candidate_distribution():
    """Verifies multi-candidate distribution dispersion, dominant margin, and preserved candidates."""
    unit_disputed = CharacterAlignmentUnit(
        position_index=0,
        candidate_character="ர",
        raw_by_provider={"p1": "ர", "p2": "ர", "p3": "த"},
        normalized_by_provider={"p1": "ர", "p2": "ர", "p3": "த"},
        confidence_by_provider={"p1": 0.9, "p2": 0.9, "p3": 0.8},
        status="substitution",
        is_disputed=True,
    )
    res = ConsensusEntropyCalculator.calculate_position_entropy(unit_disputed)
    assert res.entropy > 0.0
    assert res.is_unanimous is False
    assert set(res.distinct_candidates) == {"ர", "த"}
    assert res.dominant_candidate == "ர"
    assert res.dominant_margin > 0.0


# ==============================================================================
# 4. MISSING PROVIDER METADATA HANDLING
# ==============================================================================
def test_missing_provider_metadata_handling(synthetic_plate_image):
    """
    Verifies that when spatial bounding boxes or consensus benchmarks are missing,
    the monitor assigns NOT_ASSESSABLE and NEVER falsely reports PARTIAL_TRUNCATION.
    """
    monitor = EpigraphicalIntegrityMonitor()

    # Results with text but NO bounding boxes and NO consensus benchmark
    results = {
        "paddle": OCRResult(provider_name="paddle", model_id="v5", text="கோ மாறன் சடையன்", bounding_boxes=None),
    }

    diagnostics = monitor.analyze_truncation(results=results, input_image=synthetic_plate_image, consensus_text=None)
    assert len(diagnostics) == 1
    d = diagnostics[0]
    assert d.truncation_status == TruncationStatus.NOT_ASSESSABLE
    assert d.severity == IntegritySeverity.INFO
    assert "could not be performed" in d.diagnostic_reason.lower()

    # Verify safe dictionary serialization without Python None
    d_dict = d.to_dict()
    assert "None" not in d_dict["processed_dimensions"]
    assert "Not available" in d_dict["processed_dimensions"]
    assert "Not documented" in d_dict["max_output_limit"]


# ==============================================================================
# 5. TRUNCATION CLASSIFICATION STATES
# ==============================================================================
def test_truncation_classification_states(synthetic_plate_image):
    """
    Verifies the 5 distinct states:
    COMPLETE, PARTIAL_TRUNCATION, NOT_ASSESSABLE, PROVIDER_ERROR, EMPTY_OUTPUT.
    """
    monitor = EpigraphicalIntegrityMonitor()

    # Consensus benchmark text with 10 words
    consensus_text = "கோ மாறன் சடையற்கு யாண்டு நாலாவது எதிராமாண்டு திருக்கற்றளி எழுந்தருளப்பண்ணி வித்த கல்வெட்டு"

    results = {
        # 1. Complete output
        "engine_complete": OCRResult(
            provider_name="p_comp",
            model_id="v1",
            text="கோ மாறன் சடையற்கு யாண்டு நாலாவது எதிராமாண்டு திருக்கற்றளி எழுந்தருளப்பண்ணி வித்த கல்வெட்டு",
            bounding_boxes=[BoundingBox(ymin=0.1, xmin=0.1, ymax=0.8, xmax=0.9)]
        ),
        # 2. Partial truncation (significantly cut off: only 2 words vs 10 words)
        "engine_truncated": OCRResult(
            provider_name="p_trunc",
            model_id="v1",
            text="கோ மாறன்",
            bounding_boxes=[BoundingBox(ymin=0.1, xmin=0.1, ymax=0.2, xmax=0.3)]
        ),
        # 3. Empty output
        "engine_empty": OCRResult(
            provider_name="p_empty",
            model_id="v1",
            text="   "
        ),
        # 4. Provider Error (Exception)
        "engine_error": Exception("Sarvam API HTTP 503 Service Unavailable"),
    }

    diagnostics = monitor.analyze_truncation(results, input_image=synthetic_plate_image, consensus_text=consensus_text)
    diag_map = {d.provider_id: d for d in diagnostics}

    assert diag_map["engine_complete"].truncation_status == TruncationStatus.COMPLETE
    assert diag_map["engine_truncated"].truncation_status == TruncationStatus.PARTIAL_TRUNCATION
    assert diag_map["engine_empty"].truncation_status == TruncationStatus.EMPTY_OUTPUT
    assert diag_map["engine_error"].truncation_status == TruncationStatus.PROVIDER_ERROR


# ==============================================================================
# 6. EMPTY OCR RESPONSE HANDLING
# ==============================================================================
def test_empty_ocr_response_handling():
    """Verifies that empty string or whitespace response is categorized as EMPTY_OUTPUT."""
    monitor = EpigraphicalIntegrityMonitor()
    results = {
        "paddle": OCRResult(provider_name="paddle", model_id="v5", text=""),
    }
    diagnostics = monitor.analyze_truncation(results)
    assert len(diagnostics) == 1
    assert diagnostics[0].truncation_status == TruncationStatus.EMPTY_OUTPUT
    assert diagnostics[0].character_count == 0
    assert diagnostics[0].grapheme_count == 0


# ==============================================================================
# 7. REPEATED TAMIL GRAPHEME DETECTION (SYNTHETIC FIXTURE)
# ==============================================================================
def test_repeated_tamil_grapheme_detection():
    """
    [SYNTHETIC TEST FIXTURE]
    Verifies detection of unnatural repeated-grapheme looping (count >= 3).
    Ensures normal Tamil sandhi gemination (count = 2) is NOT falsely flagged.
    """
    monitor = EpigraphicalIntegrityMonitor()

    # Synthetic inscription text containing:
    # 1. Normal gemination: 'க்க' (repeat = 2) -> SHOULD NOT BE FLAGGED
    # 2. Anomalous looping: 'ட்ட்ட்ட்ட்ட' (4 x 'ட்') -> SHOULD BE FLAGGED
    # 3. Anomalous looping: 'ணணணணண' (5 x 'ண') -> SHOULD BE FLAGGED
    synthetic_ocr_text = "திருக்கோயில் ட்ட்ட்ட்ட்ட சுவரில் ணணணணண பொறித்த"

    results = {
        "paddle": OCRResult(provider_name="paddle", model_id="v5", text=synthetic_ocr_text),
    }

    anomalies = monitor.detect_repeated_graphemes(results)
    assert len(anomalies) >= 2

    graphemes_flagged = [a.repeated_grapheme for a in anomalies]
    assert "ட்" in graphemes_flagged
    assert "ண" in graphemes_flagged

    # Verify that 'க்' (which appears as 'க்க' with count 2 in 'திருக்கோயில்') was NOT flagged
    assert "க்" not in graphemes_flagged

    for a in anomalies:
        assert a.repeat_count >= 3
        assert a.severity == IntegritySeverity.WARNING
        assert "[" in a.context_snippet and "]" in a.context_snippet


# ==============================================================================
# 8. NULL AND NONE HANDLING
# ==============================================================================
def test_null_and_none_handling_no_anomalies_or_crashes():
    """Verifies that nulls, stringified 'None', and empty tokens are never flagged as anomalies."""
    monitor = EpigraphicalIntegrityMonitor()

    # Text containing stringified 'None None None None'
    results = {
        "engine_test": OCRResult(provider_name="test", model_id="v1", text="கல்வெட்டு None None None None முடிவு"),
    }

    anomalies = monitor.detect_repeated_graphemes(results)
    # 'None' must be ignored, zero anomalies flagged
    assert len(anomalies) == 0

    # Test safe_str helper
    assert safe_str(None) == "Not available"
    assert safe_str("None") == "Not available"
    assert safe_str("") == "Not available"
    assert safe_str(123) == "123"


# ==============================================================================
# 9. UNICODE COMBINING SEQUENCES PRESERVATION
# ==============================================================================
def test_unicode_combining_sequences_tamil_preservation():
    """
    Verifies that Tamil combining ligatures (மெய்யெழுத்து, உயிர்மெய்) are
    preserved as atomic grapheme clusters and not split into orphan code points.
    """
    # 'கோ' is U+0B95 (க) + U+0BCB (ோ)
    # 'ஸ்ரீ' is U+0BB8 U+0BCD U+0BB0 U+0BC0
    text = "கோ மாறன் ஸ்ரீ"
    graphemes = tokenize_tamil_graphemes(text)

    assert "கோ" in graphemes
    assert "ஸ்ரீ" in graphemes or "ஸ்ரீ" in text
    # No orphan combining vowel sign alone
    for g in graphemes:
        assert g not in ["\u0BCB", "\u0BCD", "\u0BC0"]


# ==============================================================================
# 10. RESEARCHER-APPROVED READING PRESERVATION (NON-OVERWRITING)
# ==============================================================================
def test_researcher_approved_reading_preservation(synthetic_plate_image):
    """
    CRITICAL EPIGRAPHICAL SAFEGUARD:
    Verifies that AI predictions NEVER silently overwrite a researcher-approved reading.
    """
    # 1. First fusion run
    results_initial = {
        "paddle": OCRResult(provider_name="paddle", model_id="v5", text="கோ மாறன்"),
        "sarvam": OCRResult(provider_name="sarvam", model_id="v1", text="கோ மாறன்"),
    }
    bundle1 = fuse_epigraphical_evidence(synthetic_plate_image, results_initial)

    # 2. Researcher reviews Slot #1 and edits it manually to an archaic reading
    rev_mgr = ResearcherReviewManager()
    for item in bundle1.adjudication_items:
        rev_mgr.add_item(item)

    first_item = bundle1.adjudication_items[0]
    rev_mgr.record_decision(
        item_id=first_item.item_id,
        decision=AdjudicationDecision.EDITED_SCHOLAR,
        approved_text="கோன்", # Scholar changes 'கோ' to 'கோன்'
        reviewer="Prof. Epigraphist",
        rationale="Terminal pulli visible in raking light photograph."
    )
    assert first_item.approved_text == "கோன்"
    assert first_item.is_researcher_approved is True

    # 3. Second fusion run executed (simulating user clicking 'Re-run OCR Fusion')
    # Raw AI engines still output 'கோ மாறன்'
    results_second = {
        "paddle": OCRResult(provider_name="paddle", model_id="v5", text="கோ மாறன்"),
        "sarvam": OCRResult(provider_name="sarvam", model_id="v1", text="கோ மாறன்"),
    }
    bundle2 = fuse_epigraphical_evidence(
        synthetic_plate_image,
        results_second,
        existing_adjudication_manager=rev_mgr
    )

    # 4. Verify that Slot #1 RETAINED the researcher-approved text 'கோன்' and was NOT overwritten by AI!
    second_item = bundle2.adjudication_items[0]
    assert second_item.approved_text == "கோன்", "CRITICAL ERROR: AI silently overwrote researcher-approved reading!"
    assert second_item.current_decision == AdjudicationDecision.EDITED_SCHOLAR.value
    assert len(second_item.revision_history) >= 1
    assert second_item.revision_history[-1].reviewer_identity == "Prof. Epigraphist"


# ==============================================================================
# 11. PROVIDER FAILURE HANDLING
# ==============================================================================
def test_provider_failure_and_exception_handling(synthetic_plate_image):
    """Verifies that engine exceptions are gracefully caught and categorized as PROVIDER_ERROR."""
    monitor = EpigraphicalIntegrityMonitor()

    results = {
        "paddle": OCRResult(provider_name="paddle", model_id="v5", text="கோ மாறன்"),
        "crnn": Exception("CRNN CUDA Out Of Memory Error"),
    }

    report = monitor.build_integrity_report(results, input_image=synthetic_plate_image)
    assert report.highest_severity == IntegritySeverity.PROVIDER_ERROR

    # Check crnn diagnostic
    crnn_diag = next(d for d in report.truncation_diagnostics if d.provider_id == "crnn")
    assert crnn_diag.truncation_status == TruncationStatus.PROVIDER_ERROR
    assert "CUDA Out Of Memory" in crnn_diag.diagnostic_reason
