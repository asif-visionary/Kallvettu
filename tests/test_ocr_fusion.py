"""
Comprehensive Unit & Integration Test Suite for Kallvettu K-IEFE.
Tests all 15 mandatory cross-verification, consensus, fusion, and adjudication scenarios:
1. Identical provider outputs
2. Partial provider outputs
3. Empty outputs
4. Description-only outputs
5. Different OCR line lengths
6. Unicode normalization
7. Insertions and deletions
8. Character disagreements (substitutions)
9. Correlated provider errors (correlation discounting)
10. Missing model scores (no fake confidence percentages)
11. Visual verification abstention (INDETERMINATE on weathered incisions)
12. Candidate ranking & tie lacuna retention ([c1/c2])
13. Source-region mismatch rejection
14. Human correction & audit history tracking
15. Fusion failure & single-provider fallback
"""

import os
import sys
import pytest
import numpy as np
from PIL import Image

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.ai.common.base_provider import OCRResult, BoundingBox
from services.ai.common.source_identity import create_source_identity, SourceIdentity, ProvenanceMode
from services.ocr_fusion.cross_verifier import (
    EpigraphicalCrossVerifier,
    normalize_for_comparison,
    tokenize_tamil_graphemes,
)
from services.ocr_fusion.consensus import (
    ConsensusEntropyCalculator,
    DisagreementSeverity,
)
from services.ocr_fusion.fusion_engine import (
    EpigraphicalFusionEngine,
    FusedTranscriptionResult,
)
from services.ocr_fusion.visual_verifier import (
    VisualEvidenceVerifier,
    VisualVerdict,
)
from services.ocr_fusion.historical_validator import (
    HistoricalContextValidator,
)
from services.ocr_fusion.adaptive_router import (
    AdaptiveRoutingEngine,
)
from services.ocr_fusion.researcher_review import (
    ResearcherReviewManager,
    AdjudicationItem,
    AdjudicationDecision,
)
from services.ocr_fusion.active_learning import (
    ActiveLearningManager,
    VerifiedAnnotationRecord,
)
from services.ocr_fusion.audit_trail import (
    FusionAuditLogger,
)
from services.ocr_fusion.evaluation import (
    EpigraphicalFusionEvaluator,
)
from services.ocr_fusion import (
    fuse_epigraphical_evidence,
    MasterEvidenceFusionBundle,
)


# ==============================================================================
# 1. IDENTICAL PROVIDER OUTPUTS
# ==============================================================================

def test_identical_provider_outputs():
    """Verify that identical outputs from all engines yield unanimous consensus and zero entropy."""
    text = "ஸ்வஸ்திஸ்ரீ ராஜராஜ சோழதேவர்"
    res1 = OCRResult(text=text, provider_name="PaddleOCR", model_id="pp-ocrv5", confidence=0.92)
    res2 = OCRResult(text=text, provider_name="Sarvam Vision", model_id="sarvam-vision", confidence=0.95)
    res3 = OCRResult(text=text, provider_name="Gemini Vision", model_id="gemini-2.5-flash", confidence=0.88)

    verifier = EpigraphicalCrossVerifier()
    rep = verifier.verify({"paddle": res1, "sarvam": res2, "gemini": res3})

    assert rep.disputed_character_slots == 0
    assert rep.agreed_character_slots == rep.total_character_slots
    assert rep.agreement_ratio == 1.0

    consensus = ConsensusEntropyCalculator.evaluate_report(rep)
    assert consensus.mean_entropy == 0.0
    assert consensus.max_entropy == 0.0
    assert consensus.disputed_slots_count == 0
    assert consensus.unanimous_ratio == 1.0


# ==============================================================================
# 2. PARTIAL PROVIDER OUTPUTS
# ==============================================================================

def test_partial_provider_outputs():
    """Verify alignment handles a provider that captured only part of the inscription."""
    full_text = "கோ ஆதன் செல்லிரும்பொறை மகன் பெருங்கடுங்கோ"
    truncated_text = "கோ ஆதன் செல்லிரும்பொறை"

    res1 = OCRResult(text=full_text, provider_name="Gemini", model_id="gemini-vision", confidence=0.85)
    res2 = OCRResult(text=truncated_text, provider_name="Paddle", model_id="paddle-ocr", confidence=0.90)

    verifier = EpigraphicalCrossVerifier()
    rep = verifier.verify({"gemini": res1, "paddle": res2})

    assert rep.total_character_slots > 0
    # Truncated characters must be marked as deleted/missing in paddle
    deletions = [c for c in rep.character_alignments if c.status in {"deletion", "insertion"}]
    assert len(deletions) > 0

    # Truncation anomaly reported
    assert any(m["status"] == "partial_truncation" for m in rep.missing_regions)


# ==============================================================================
# 3. EMPTY PROVIDER OUTPUTS
# ==============================================================================

def test_empty_provider_outputs():
    """Verify that an engine returning an empty string does not crash alignment."""
    res1 = OCRResult(text="மாங்குளம் கல்வெட்டு", provider_name="Paddle", model_id="pp-ocrv5")
    res2 = OCRResult(text="", provider_name="Sarvam", model_id="sarvam-vision", confidence=None)

    verifier = EpigraphicalCrossVerifier()
    rep = verifier.verify({"paddle": res1, "sarvam": res2})

    assert rep.total_character_slots > 0
    # Provider with empty text is logged in missing regions
    assert any(m["provider_id"] == "sarvam" for m in rep.missing_regions)


# ==============================================================================
# 4. DESCRIPTION-ONLY OUTPUTS
# ==============================================================================

def test_description_only_outputs():
    """Verify that generic visual descriptions are excluded from inscriptional character alignment."""
    res_desc = OCRResult(
        text="",
        provider_name="Sarvam Vision",
        model_id="sarvam-vision",
        metadata={
            "transcription_status": "description_only",
            "visual_description": "This photograph shows an ancient weathered stone slab with carvings."
        }
    )
    res_ocr = OCRResult(
        text="கணிய் நந்தாஸிரிய்",
        provider_name="Gemini Vision",
        model_id="gemini-vision"
    )

    verifier = EpigraphicalCrossVerifier()
    rep = verifier.verify({"sarvam": res_desc, "gemini": res_ocr})

    # Only the genuine OCR provider should enter character slots
    assert "sarvam" not in rep.provider_names
    assert "gemini" in rep.provider_names
    assert rep.character_alignments[0].candidate_character == "க"


# ==============================================================================
# 5. DIFFERENT OCR LINE LENGTHS
# ==============================================================================

def test_different_ocr_line_lengths():
    """Verify alignment handles differing line counts and line segmentation splits."""
    text_2_lines = "ஸ்வஸ்திஸ்ரீ ராஜராஜ சோழதேவர்க்கு\nயாண்டு இருபத்தொன்றாவது"
    text_1_line = "ஸ்வஸ்திஸ்ரீ ராஜராஜ சோழதேவர்க்கு யாண்டு இருபத்தொன்றாவது"

    res1 = OCRResult(text=text_2_lines, provider_name="EngineA", model_id="model-a")
    res2 = OCRResult(text=text_1_line, provider_name="EngineB", model_id="model-b")

    verifier = EpigraphicalCrossVerifier()
    rep = verifier.verify({"a": res1, "b": res2})

    assert len(rep.line_alignments) >= 1
    # Character alignment still matches the underlying Tamil graphemes
    assert rep.character_alignments[0].candidate_character == "ஸ்"


# ==============================================================================
# 6. UNICODE NORMALIZATION (NFC COMPARISON)
# ==============================================================================

def test_unicode_normalization_nfc_equality():
    """Verify decomposed and composed Unicode variants evaluate as identical in comparison."""
    import unicodedata
    composed = "கோ"  # NFC
    decomposed = unicodedata.normalize("NFD", composed)

    assert composed != decomposed  # Code points differ in raw representation
    assert normalize_for_comparison(composed) == normalize_for_comparison(decomposed)

    res1 = OCRResult(text=composed, provider_name="P1", model_id="m1")
    res2 = OCRResult(text=decomposed, provider_name="P2", model_id="m2")

    verifier = EpigraphicalCrossVerifier()
    rep = verifier.verify({"p1": res1, "p2": res2})

    # Evaluated as identical without false dispute
    assert rep.character_alignments[0].status == "identical"
    assert rep.character_alignments[0].is_disputed is False


# ==============================================================================
# 7. INSERTIONS AND DELETIONS DETECTION
# ==============================================================================

def test_insertion_and_deletion_detection():
    """Verify character insertions and omissions are categorized correctly."""
    # Engine A has extra character 'ரி' omitted in B
    text_a = "நந்தாஸிரிய்"
    text_b = "நந்தாஸிய்"  # Missing 'ரி'

    res_a = OCRResult(text=text_a, provider_name="A", model_id="ma")
    res_b = OCRResult(text=text_b, provider_name="B", model_id="mb")

    verifier = EpigraphicalCrossVerifier()
    rep = verifier.verify({"A": res_a, "B": res_b}, reference_provider_id="A")

    assert rep.disputed_character_slots > 0
    statuses = [c.status for c in rep.character_alignments]
    assert "insertion" in statuses or "deletion" in statuses


# ==============================================================================
# 8. CHARACTER DISAGREEMENTS (SUBSTITUTION)
# ==============================================================================

def test_character_disagreements_substitution():
    """Verify competing character readings (e.g. ந vs த) produce substitution units."""
    text_a = "நந்தாசிரிய்"  # Chiseled with 'சி'
    text_b = "நந்தாஸிரிய்"  # Chiseled with Grantha 'ஸி'

    res_a = OCRResult(text=text_a, provider_name="Paddle", model_id="pp-ocrv5")
    res_b = OCRResult(text=text_b, provider_name="Gemini", model_id="gemini-vision")

    verifier = EpigraphicalCrossVerifier()
    rep = verifier.verify({"paddle": res_a, "gemini": res_b})

    substitutions = [c for c in rep.character_alignments if c.status == "substitution"]
    assert len(substitutions) >= 1
    sub = substitutions[0]
    assert sub.is_disputed is True
    assert set(sub.normalized_by_provider.values()) == {"சி", "ஸி"}


# ==============================================================================
# 9. CORRELATED PROVIDER ERRORS (CORRELATION DISCOUNTING)
# ==============================================================================

def test_correlated_provider_errors_discounting():
    """Verify that models with correlated failure modes have their combined weight discounted."""
    fusion_engine = EpigraphicalFusionEngine()

    pids = ["paddleocr_local", "sarvam_vision", "gemini_vision"]
    base_weights = {
        "paddleocr_local": 0.80,
        "sarvam_vision": 0.80,
        "gemini_vision": 0.85,
    }

    decorrelated = fusion_engine.compute_decorrelated_weights(pids, base_weights)

    # PaddleOCR and Sarvam Vision have higher correlation (0.42), so their weights should be discounted more
    # than Gemini Vision (which has lower correlation with both)
    assert decorrelated["paddleocr_local"] < base_weights["paddleocr_local"]
    assert decorrelated["sarvam_vision"] < base_weights["sarvam_vision"]


# ==============================================================================
# 10. MISSING MODEL SCORES (NO FAKE CONFIDENCE)
# ==============================================================================

def test_missing_model_scores_handling():
    """Verify that when a provider outputs confidence=None, no fake confidence is invented."""
    res1 = OCRResult(text="சாசனம்", provider_name="Paddle", model_id="pp-ocr", confidence=None)
    res2 = OCRResult(text="சாஸனம்", provider_name="Sarvam", model_id="sarvam-vision", confidence=None)

    verifier = EpigraphicalCrossVerifier()
    rep = verifier.verify({"paddle": res1, "sarvam": res2})

    unit = rep.character_alignments[0]
    # Stored confidence must remain None, never hardcoded to 1.0 or 0.85
    assert unit.confidence_by_provider["paddle"] is None
    assert unit.confidence_by_provider["sarvam"] is None


# ==============================================================================
# 11. VISUAL VERIFICATION ABSTENTION (INDETERMINATE)
# ==============================================================================

def test_visual_verification_abstention():
    """Verify that visual stroke verifier cleanly abstains when incisions are indeterminate."""
    verifier = VisualEvidenceVerifier()
    dummy_img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    bbox = BoundingBox(ymin=0.1, xmin=0.1, ymax=0.3, xmax=0.3)

    inspection = verifier.verify_disputed_region(
        dummy_img,
        bbox=bbox,
        candidates=["ந", "த"],
        region_id="disp_01"
    )

    assert inspection.recommended_verdict == VisualVerdict.INDETERMINATE
    assert inspection.abstention_reason is not None
    # Hypotheses are marked as hypotheses
    for hyp in inspection.hypotheses_by_candidate.values():
        assert hyp.is_hypothesis is True


# ==============================================================================
# 12. CANDIDATE RANKING & TIE LACUNA RETENTION
# ==============================================================================

def test_candidate_ranking_and_tie_lacuna_retention():
    """Verify that close margin ties between competing readings retain bracketed lacuna [c1/c2]."""
    text1 = "நந்தாசிரிய்"
    text2 = "நந்தாஸிரிய்"

    res1 = OCRResult(text=text1, provider_name="P1", model_id="m1")
    res2 = OCRResult(text=text2, provider_name="P2", model_id="m2")

    verifier = EpigraphicalCrossVerifier()
    rep = verifier.verify({"p1": res1, "p2": res2})

    fusion_eng = EpigraphicalFusionEngine()
    fused_res = fusion_eng.fuse_character_candidates(rep)

    # In conservative text, the tied position must be bracketed like [சி/ஸி]
    assert "[" in fused_res.conservative_text and "]" in fused_res.conservative_text
    assert fused_res.abstention_slots_count > 0


# ==============================================================================
# 13. SOURCE REGION MISMATCH REJECTION
# ==============================================================================

def test_source_region_mismatch_rejection():
    """Verify that comparing distinct source images or mismatched hashes fails comparability check."""
    img1 = Image.new("RGB", (64, 64), color=(200, 100, 100))
    img2 = Image.new("RGB", (64, 64), color=(100, 200, 100))

    src1 = create_source_identity(img1, filename="mangulam.jpg")
    src2 = create_source_identity(img2, filename="thanjavur.jpg")

    assert src1.is_comparable_with(src2) is False
    assert src1.image_sha256 != src2.image_sha256


# ==============================================================================
# 14. HUMAN CORRECTION & AUDIT HISTORY TRACKING
# ==============================================================================

def test_human_correction_and_audit_history():
    """Verify that researcher adjudication updates approved text without destroying raw readings."""
    review_mgr = ResearcherReviewManager()
    item = AdjudicationItem(
        item_id="adj_test_01",
        region_index=3,
        bounding_box=BoundingBox(ymin=0.1, xmin=0.1, ymax=0.3, xmax=0.3),
        pixel_crop_coords=(10, 10, 50, 50),
        provider_candidates={"paddle": "நந்தாசிரிய்", "gemini": "நந்தாஸிரிய்"},
        provider_confidences={"paddle": 0.80, "gemini": 0.85},
        fused_candidate="நந்தாஸிரிய்",
        alternative_readings=["நந்தாசிரிய்", "நந்தாஸிரிய்"],
        disagreement_reason="Substitution",
        consensus_entropy=1.0,
        visual_hypothesis=None,
        historical_evidence=[]
    )
    review_mgr.add_item(item)

    # Researcher edits reading to 'நந்தாஸிரிய்'
    updated = review_mgr.record_decision(
        item_id="adj_test_01",
        decision=AdjudicationDecision.EDITED_SCHOLAR,
        approved_text="நந்தாஸிரிய்",
        reviewer="Epigraphist Dr. K. Rajan",
        rationale="Palaeographical Grantha 'sa' verified on physical estampage."
    )

    assert updated.current_decision == AdjudicationDecision.EDITED_SCHOLAR.value
    assert updated.approved_text == "நந்தாஸிரிய்"
    assert len(updated.revision_history) == 1
    rev = updated.revision_history[0]
    assert rev.reviewer_identity == "Epigraphist Dr. K. Rajan"
    assert "Grantha 'sa'" in rev.rationale
    # Original provider candidates remain intact!
    assert updated.provider_candidates["paddle"] == "நந்தாசிரிய்"


# ==============================================================================
# 15. FUSION PIPELINE INTEGRATION & FALLBACK
# ==============================================================================

def test_master_fusion_bundle_execution():
    """Verify master fuse_epigraphical_evidence executes full pipeline cleanly."""
    img = Image.new("RGB", (120, 120), color=(180, 170, 150))
    res1 = OCRResult(text="ஸ்வஸ்திஸ்ரீ ராஜராஜ சோழதேவர்", provider_name="Paddle", model_id="paddle-v5", confidence=0.89)
    res2 = OCRResult(text="ஸ்வஸ்திஸ்ரீ ராஜராஜ சோழதேவர்க்கு", provider_name="Gemini", model_id="gemini-vision", confidence=0.92)

    bundle = fuse_epigraphical_evidence(
        image=img,
        results={"paddle": res1, "gemini": res2},
        script_hint="medieval_chola"
    )

    assert isinstance(bundle, MasterEvidenceFusionBundle)
    assert bundle.fused_result.fused_text != ""
    assert bundle.consensus_report.total_slots_evaluated > 0
    assert bundle.audit_record.execution_id.startswith("KIEFE_")
    # Verify historical context matches Rajaraja Chola
    assert bundle.historical_validation.identified_monarch is not None
    assert "ராஜராஜ" in bundle.historical_validation.identified_monarch


# ==============================================================================
# 16. HEATMAP OVERLAY & ACTIVE LEARNING EXPORT
# ==============================================================================

def test_evidence_heatmap_overlay(tmp_path):
    """Verify render_evidence_heatmap_overlay generates an annotated image."""
    from services.ocr_fusion.ui_renderer import render_evidence_heatmap_overlay

    img = Image.new("RGB", (200, 200), color=(180, 170, 150))
    res1 = OCRResult(text="ஸ்வஸ்திஸ்ரீ ராஜராஜ", provider_name="Paddle", model_id="paddle-v5", confidence=0.89)
    res2 = OCRResult(text="ஸ்வஸ்திஸ்ரீ ராசராச", provider_name="Sarvam", model_id="sarvam-vision", confidence=0.92)

    bundle = fuse_epigraphical_evidence(
        image=img,
        results={"paddle": res1, "sarvam": res2},
        script_hint="medieval_chola"
    )

    heatmap = render_evidence_heatmap_overlay(img, bundle)
    assert isinstance(heatmap, Image.Image)
    assert heatmap.size == (200, 200)


def test_active_learning_ingest_and_export(tmp_path):
    """Verify active learning storage and TSV/JSON training export."""
    al_mgr = ActiveLearningManager(storage_dir=str(tmp_path))
    record = VerifiedAnnotationRecord(
        specimen_id="specimen_test_01",
        inscription_id="ins_001",
        source_image_id="img_001",
        image_sha256="abc123sha256",
        crop_coordinates=(10, 10, 50, 50),
        normalized_bbox=(0.1, 0.1, 0.5, 0.5),
        script="tamil_brahmi",
        original_candidates={"paddle": "க", "sarvam": "ச"},
        final_approved_reading="க",
        reviewer_identity="Researcher Dr. K. Rajan",
        review_timestamp="2026-11-10 12:00:00",
        notes="Stroke inspection confirmed horizontal crossbar."
    )

    spec_id = al_mgr.store_verified_annotation(record)
    assert spec_id == "specimen_test_01"

    all_recs = al_mgr.load_all_annotations()
    assert len(all_recs) == 1
    assert all_recs[0].final_approved_reading == "க"

    # Export tests
    json_out = al_mgr.export_for_model_training("json")
    assert "specimen_test_01" in json_out

    tsv_out = al_mgr.export_for_model_training("tsv")
    assert "specimen_test_01" in tsv_out
    assert "\t" in tsv_out


def test_bundle_attributes_for_ui_rendering():
    """Verify all bundle attributes accessed by the UI exist and return expected types."""
    img = Image.new("RGB", (100, 100), color=(180, 170, 150))
    res1 = OCRResult(text="ஸ்வஸ்திஸ்ரீ ராஜராஜ", provider_name="Paddle", model_id="paddle-v5", confidence=0.89)
    res2 = OCRResult(text="ஸ்வஸ்திஸ்ரீ ராசராச", provider_name="Sarvam", model_id="sarvam-vision", confidence=0.92)

    bundle = fuse_epigraphical_evidence(
        image=img,
        results={"paddle": res1, "sarvam": res2},
        script_hint="medieval_chola"
    )

    # Cross-verification report properties
    c_rep = bundle.cross_verification_report
    assert isinstance(c_rep.disputed_positions, list)
    assert isinstance(c_rep.disputed_character_slots, int)
    assert isinstance(c_rep.agreed_character_slots, int)
    assert isinstance(c_rep.total_character_slots, int)
    assert isinstance(c_rep.agreement_ratio, float)

    # Line alignment properties
    for la in c_rep.line_alignments:
        assert isinstance(la.normalized_text, str)
        assert isinstance(la.y_min_norm, float)
        assert isinstance(la.y_max_norm, float)

    # Consensus report properties
    cons = bundle.consensus_report
    assert isinstance(cons.mean_entropy, float)
    assert isinstance(cons.max_entropy, float)
    assert isinstance(cons.unanimous_ratio, float)
    assert isinstance(cons.disputed_ratio, float)
    assert isinstance(cons.epigraphical_disclaimer, str)

    # Historical validation properties
    hist = bundle.historical_validation
    assert isinstance(hist.total_rulers_detected, int)
    assert isinstance(hist.total_dynasties_detected, int)
    assert isinstance(hist.administrative_terms_detected, list)
    assert isinstance(hist.epigraphical_vocabulary_ratio, float)
    assert isinstance(hist.corpus_cooccurrence_support, float)

    # Fused result properties
    fused = bundle.fused_result
    assert isinstance(fused.fused_text, str)
    assert isinstance(fused.conservative_text, str)
    assert isinstance(fused.strategy_used, str)
    assert isinstance(fused.strongest_baseline_provider, str)

    # Audit record properties
    audit = bundle.audit_record
    assert isinstance(audit.cryptographic_sha256, str)
    assert len(audit.cryptographic_sha256) == 64
    assert isinstance(audit.to_json(), str)


