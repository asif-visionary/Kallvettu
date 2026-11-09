"""
UI Rendering Module for Kallvettu Intelligent Epigraphical Evidence Fusion Engine (K-IEFE).
Renders the complete 9-tab epigraphical evidence fusion workbench, region heatmap overlay,
and researcher adjudication interfaces in Streamlit.
"""

import io
import json
import time
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

from services.ocr_fusion import MasterEvidenceFusionBundle
from services.ocr_fusion.researcher_review import (
    ResearcherReviewManager,
    AdjudicationItem,
    AdjudicationDecision,
)
from services.ocr_fusion.active_learning import (
    ActiveLearningManager,
    VerifiedAnnotationRecord,
)
from services.ocr_fusion.consensus import DisagreementSeverity
from services.ocr_fusion.visual_verifier import VisualVerdict
from ui_components import render_metric_card
from nlp_translation_engine import analyze_custom_epigraph_text


def render_evidence_heatmap_overlay(
    image: Image.Image,
    bundle: MasterEvidenceFusionBundle
) -> Image.Image:
    """
    Renders an archaeological evidence heatmap overlay on top of the stone inscription.
    Draws color-coded bounding boxes around disputed or high-entropy inscriptional zones:
    - Red (H >= 0.8): High dispute
    - Amber (0.5 <= H < 0.8): Divided consensus
    - Green (H < 0.5): Mild / Agreement
    """
    if not isinstance(image, Image.Image):
        return image

    img = image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img.size

    try:
        font_size = max(11, int(min(w, h) * 0.022))
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    items = getattr(bundle, "adjudication_items", [])
    for it in items:
        coords = getattr(it, "pixel_crop_coords", None)
        bbox = getattr(it, "bounding_box", None)
        if coords and coords != (0, 0, 0, 0):
            x1, y1, x2, y2 = coords
        elif bbox:
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int(bbox.xmax * w)
            y2 = int(bbox.ymax * h)
        else:
            continue

        entropy = getattr(it, "consensus_entropy", 0.0)
        if entropy >= 0.8:
            border_col = (220, 53, 69, 255)  # Crimson Red
            fill_col = (220, 53, 69, 50)
            tag = "HIGH DISPUTE"
        elif entropy >= 0.5:
            border_col = (255, 152, 0, 255)  # Amber
            fill_col = (255, 152, 0, 45)
            tag = "DIVIDED"
        else:
            border_col = (40, 167, 69, 255)  # Green
            fill_col = (40, 167, 69, 35)
            tag = "CONSENSUS"

        # Draw bounding rectangle with semi-transparent fill
        draw.rectangle([x1, y1, x2, y2], outline=border_col, width=3, fill=fill_col)

        # Label tag above box
        lbl = f"Slot #{it.region_index + 1}: {it.fused_candidate} [H={entropy:.2f}]"
        try:
            tbox = draw.textbbox((x1, max(0, y1 - 22)), lbl, font=font)
            draw.rectangle(tbox, fill=border_col)
            draw.text((x1 + 3, max(0, y1 - 21)), lbl, fill=(255, 255, 255, 255), font=font)
        except Exception:
            draw.text((x1 + 2, max(0, y1 - 15)), lbl, fill=border_col)

    combined = Image.alpha_composite(img, overlay)
    return combined.convert("RGB")


def render_kiefe_dashboard(
    bundle: MasterEvidenceFusionBundle,
    provider_manager: Any,
    ocr_target: Any
) -> None:
    """
    Renders the complete 9-tab K-IEFE Epigraphical Evidence Fusion Workbench.
    """
    # Safeguard ocr_target as PIL Image
    if isinstance(ocr_target, np.ndarray):
        ocr_target = Image.fromarray(ocr_target.astype(np.uint8)).convert("RGB")
    elif not isinstance(ocr_target, Image.Image):
        ocr_target = Image.new("RGB", (400, 400), color=(180, 170, 150))
    st.markdown("---")
    st.markdown("### ⚡ K-IEFE: Intelligent Epigraphical Evidence Fusion & Verification")
    st.caption(
        "Unified cross-engine recognition, Shannon consensus entropy analysis, "
        "visual stroke verification, and researcher adjudication (Kallvettu v2.1.0)."
    )

    # Top KPI metrics bar
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_metric_card(
            "Inter-Engine Agreement",
            f"{bundle.cross_verification_report.agreement_ratio * 100:.1f}%",
            "Multi-sequence grapheme token overlap"
        )
    with k2:
        render_metric_card(
            "Consensus Entropy (H)",
            f"{bundle.consensus_report.mean_entropy:.3f} bits",
            "Agreement dispersion across models"
        )
    with k3:
        disputed_cnt = bundle.cross_verification_report.disputed_character_slots
        render_metric_card(
            "Disputed Zones",
            str(disputed_cnt),
            f"Positions requiring epigraphist review ({len(bundle.adjudication_items)} critical)"
        )
    with k4:
        render_metric_card(
            "Fusion Strategy",
            bundle.fused_result.strategy_used.replace("strategy_", "").replace("_", " ").title(),
            f"Baseline: {bundle.fused_result.strongest_baseline_provider}"
        )
    with k5:
        render_metric_card(
            "Historical Matches",
            str(len(bundle.historical_validation.supporting_matches)),
            f"Rulers: {bundle.historical_validation.total_rulers_detected} | Dyn: {bundle.historical_validation.total_dynasties_detected}"
        )

    # 9 Specialized Tabs
    tab_titles = [
        "🤖 1. Provider Outputs",
        "📏 2. Line & Region Alignment",
        "🔍 3. Character Comparison",
        "📊 4. Consensus Entropy",
        "👁️ 5. Visual Verification",
        "📜 6. Historical Context",
        "⚡ 7. Fused Candidate",
        "⚖️ 8. Researcher Adjudication",
        "🛡️ 9. Audit Trail & Active Learning",
    ]
    tabs = st.tabs(tab_titles)

    # ==========================================================================
    # TAB 1: 🤖 PROVIDER OUTPUTS
    # ==========================================================================
    with tabs[0]:
        st.markdown("#### 🤖 Multi-Engine Raw Decipherments & Manifest")
        st.caption(
            "Truthful display of individual provider adapter responses. "
            "Clearly distinguishes Registered vs Configured vs Enabled vs Executed states."
        )

        all_statuses = provider_manager.get_all_statuses() if hasattr(provider_manager, "get_all_statuses") else {}
        cross_rep = bundle.cross_verification_report

        manifest_rows = []
        for pid, s in all_statuses.items():
            if s.get("capability") in ["Local OCR", "Document OCR", "Multimodal Vision", "Historical Script OCR"]:
                is_executed = pid in cross_rep.raw_texts_by_provider
                ocr_obj = getattr(st.session_state.ocr_comparison_result, "provider_results", {}).get(pid) if hasattr(st.session_state, "ocr_comparison_result") and st.session_state.ocr_comparison_result else None
                conf_str = f"{ocr_obj.confidence * 100:.1f}%" if ocr_obj and ocr_obj.confidence is not None else "N/A"
                lat_str = f"{ocr_obj.latency_seconds:.3f} s" if ocr_obj else "—"

                manifest_rows.append({
                    "Engine": s.get("name", pid),
                    "Provider ID": pid,
                    "Capability": s.get("capability", "OCR"),
                    "Configured": s.get("configured", "NO"),
                    "Enabled": s.get("enabled", "Disabled"),
                    "Executed": "✓ YES" if is_executed else "— SKIPPED",
                    "Confidence": conf_str,
                    "Latency": lat_str,
                })

        if manifest_rows:
            st.dataframe(pd.DataFrame(manifest_rows), width="stretch", hide_index=True)

        st.markdown("##### Detailed Inscriptional Transcriptions by Engine:")
        p_names = list(cross_rep.raw_texts_by_provider.keys())
        if p_names:
            sub_tabs = st.tabs([f"📌 {pid}" for pid in p_names])
            for s_tab, pid in zip(sub_tabs, p_names):
                with s_tab:
                    raw_txt = cross_rep.raw_texts_by_provider.get(pid, "").strip()
                    display_txt = raw_txt if raw_txt else "*(No readable text returned by this engine)*"

                    c_txt, c_meta = st.columns([2.5, 1])
                    with c_txt:
                        st.markdown(
                            f"<div class='transcription-box' style='border-left-color: #C5A15A; font-family: monospace; font-size: 1.05rem;'>{display_txt}</div>",
                            unsafe_allow_html=True
                        )
                        if raw_txt and st.button(f"📋 Adopt {pid} Reading into Workbench", key=f"btn_adopt_kiefe_{pid}"):
                            st.session_state.analysis_output = analyze_custom_epigraph_text(
                                raw_txt,
                                provenance=f"Adopted Reading: {pid}",
                                allow_preset_match=False,
                                source_identity=st.session_state.source_identity
                            )
                            st.success(f"✓ Adopted reading from {pid} into Workbench!")
                            st.rerun()

                    with c_meta:
                        ocr_obj = getattr(st.session_state.ocr_comparison_result, "provider_results", {}).get(pid) if hasattr(st.session_state, "ocr_comparison_result") and st.session_state.ocr_comparison_result else None
                        if ocr_obj:
                            st.metric("Reported Confidence", f"{ocr_obj.confidence * 100:.1f}%" if ocr_obj.confidence is not None else "N/A")
                            st.metric("Execution Latency", f"{ocr_obj.latency_seconds:.3f} s")
                            st.metric("Bounding Boxes", len(ocr_obj.bounding_boxes))
                            st.caption(f"**Model ID:** `{ocr_obj.model_id}`")
                            if ocr_obj.provenance_notes:
                                st.caption(f"**Notes:** {ocr_obj.provenance_notes}")

                            with st.expander("🔍 Inspect Raw Adapter Response", expanded=False):
                                st.json({
                                    "text": ocr_obj.text,
                                    "confidence": ocr_obj.confidence,
                                    "latency_seconds": ocr_obj.latency_seconds,
                                    "model_id": ocr_obj.model_id,
                                    "raw_response_preview": str(ocr_obj.raw_response)[:400] if ocr_obj.raw_response else "None"
                                })

    # ==========================================================================
    # TAB 2: 📏 LINE & REGION ALIGNMENT
    # ==========================================================================
    with tabs[1]:
        st.markdown("#### 📏 Text-Line Segmentation & Spatial Coordinate Mapping")
        st.caption(
            "Aligns segmented text lines and spatial bounding boxes across engines. "
            "Maps cropped candidate coordinates back to original source photograph dimensions."
        )

        line_alignments = bundle.cross_verification_report.line_alignments
        if line_alignments:
            line_rows = []
            for la in line_alignments:
                norm_t = getattr(la, "normalized_text", "")
                y_min = getattr(la, "y_min_norm", 0.0)
                y_max = getattr(la, "y_max_norm", 1.0)
                conf = getattr(la, "confidence", None)
                row = {
                    "Line #": la.line_index + 1,
                    "Normalized Text": norm_t,
                    "Y-Range": f"{y_min:.3f} – {y_max:.3f}",
                    "Confidence": f"{conf * 100:.1f}%" if conf is not None else "N/A"
                }
                for pid, txt in la.text_by_provider.items():
                    row[f"{pid} Text"] = txt
                line_rows.append(row)
            st.dataframe(pd.DataFrame(line_rows), width="stretch", hide_index=True)
        else:
            st.info("ℹ️ Single continuous lithic line detected or line segmentation bypassed.")

        # Missing regions & truncation alerts
        missing = bundle.cross_verification_report.missing_regions
        if missing:
            st.markdown("##### ⚠️ Region Truncation & Missing Line Alerts:")
            for m in missing:
                st.warning(
                    f"**{m.get('provider_id')}**: Status `{m.get('status')}`. "
                    f"Length: {m.get('length')} chars (Max: {m.get('max_length')}, Ratio: {m.get('ratio', 0):.2f}). "
                    f"Palaeographical reason: {m.get('reason')}."
                )

        # Region coordinate mapping
        if bundle.visual_inspections:
            st.markdown("##### 📍 Crop-to-Source Physical Coordinate Mapping:")
            w_orig, h_orig = ocr_target.size
            crop_rows = []
            for insp in bundle.visual_inspections:
                x1, y1, x2, y2 = insp.pixel_coordinates
                crop_rows.append({
                    "Region ID": insp.region_id,
                    "Bounding Box (Norm)": f"({insp.bounding_box.ymin:.3f}, {insp.bounding_box.xmin:.3f}, {insp.bounding_box.ymax:.3f}, {insp.bounding_box.xmax:.3f})",
                    "Source Coordinates (Pixels)": f"X: [{x1} .. {x2}], Y: [{y1} .. {y2}]",
                    "Dimensions (W × H)": f"{x2 - x1} × {y2 - y1} px",
                    "Source Image Size": f"{w_orig} × {h_orig} px",
                    "Candidates": ", ".join(insp.competing_candidates),
                    "Recommended Verdict": insp.recommended_verdict.upper(),
                })
            st.dataframe(pd.DataFrame(crop_rows), width="stretch", hide_index=True)

    # ==========================================================================
    # TAB 3: 🔍 CHARACTER COMPARISON
    # ==========================================================================
    with tabs[2]:
        st.markdown("#### 🔍 Grapheme-Level Multi-Sequence Diff & Disagreements")
        st.caption(
            "Character-by-character alignment preserving Tamil combining ligatures (மெய்யெழுத்து, உயிர்மெய்). "
            "Highlights insertions, deletions, substitutions, and repeated-character OCR anomalies."
        )

        c_rep = bundle.cross_verification_report
        disputed_units = [u for u in c_rep.character_alignments if u.is_disputed]

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Grapheme Slots", len(c_rep.character_alignments))
        with m2:
            st.metric("Agreed Slots", len(c_rep.character_alignments) - len(disputed_units))
        with m3:
            st.metric("Disputed Slots", len(disputed_units))

        # Repeated-character stutter alerts
        anomalies = c_rep.repeated_character_anomalies
        if anomalies:
            st.markdown("##### ⚠️ Repeated-Character OCR Model Looping Anomalies:")
            for an in anomalies:
                st.error(
                    f"**Engine `{an.get('provider_id')}`**: Anomalous repeat of character "
                    f"`'{an.get('repeated_char')}'` ({an.get('repeat_count')} consecutive times). "
                    f"Token snippet: `{an.get('token_snippet')}`. {an.get('reason')}."
                )

        # Character alignment table
        if c_rep.character_alignments:
            filter_disputes_only = st.checkbox("Show Disputed Character Slots Only", value=True)
            display_units = disputed_units if filter_disputes_only else c_rep.character_alignments

            matrix_rows = []
            for u in display_units:
                status_icon = "⚠️ SUB" if u.status == "substitution" else (
                    "➕ INS" if u.status == "insertion" else (
                        "➖ DEL" if u.status == "deletion" else "✓ MATCH"
                    )
                )
                r = {
                    "Slot #": u.position_index + 1,
                    "Status": status_icon,
                    "Proposed Candidate": u.candidate_character,
                }
                for pid in c_rep.provider_names:
                    r[f"{pid} Reading"] = u.raw_by_provider.get(pid, "—")
                matrix_rows.append(r)

            if matrix_rows:
                st.dataframe(pd.DataFrame(matrix_rows), width="stretch", hide_index=True)
            else:
                st.success("✓ All character slots exhibit unanimous agreement across active engines!")

    # ==========================================================================
    # TAB 4: 📊 CONSENSUS ENTROPY
    # ==========================================================================
    with tabs[3]:
        st.markdown("#### 📊 Candidate-Distribution Shannon Consensus Entropy")

        # Explicit Scientific Epigraphical Disclaimer
        st.warning(
            "⚠️ **Scientific Epigraphical Principle on Consensus Entropy**:\n\n"
            f"> *\"{bundle.consensus_report.epigraphical_disclaimer}\"*\n\n"
            "**Rule:** Candidate-distribution Shannon entropy measures model agreement dispersion, "
            "**NOT** the probability of historical veracity. High-pass filter artifacts and granite weathering "
            "frequently cause independent neural models to agree on spurious strokes. Always inspect physical stone incisions."
        )

        crep = bundle.consensus_report
        e1, e2, e3, e4 = st.columns(4)
        with e1:
            render_metric_card("Mean Entropy", f"{crep.mean_entropy:.3f} bits", "Average agreement dispersion")
        with e2:
            render_metric_card("Max Position Entropy", f"{crep.max_entropy:.3f} bits", "Peak uncertainty slot")
        with e3:
            render_metric_card("Unanimous Slots", f"{crep.unanimous_ratio * 100:.1f}%", f"{crep.unanimous_slots_count} of {crep.total_slots_evaluated} slots (H = 0.0)")
        with e4:
            render_metric_card("Disputed Slots", f"{crep.disputed_ratio * 100:.1f}%", f"{crep.disputed_slots_count} slots requiring human review")

        st.markdown("##### Disagreement Severity Categorization:")
        s_dist = crep.severity_distribution
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("🟩 Unanimous (H = 0.0)", s_dist.get(DisagreementSeverity.UNANIMOUS, 0))
        with s2:
            st.metric("🟦 Strong (0 < H ≤ 0.5)", s_dist.get(DisagreementSeverity.STRONG_CONSENSUS, 0))
        with s3:
            st.metric("🟨 Divided (0.5 < H ≤ 1.0)", s_dist.get(DisagreementSeverity.DIVIDED, 0))
        with s4:
            st.metric("🟥 High Dispute (H > 1.0)", s_dist.get(DisagreementSeverity.HIGH_DISPUTE, 0))

        # Position entropy details
        if crep.position_entropy_details:
            with st.expander("📋 Detailed Position-by-Position Entropy Table", expanded=False):
                ent_rows = []
                for p in crep.position_entropy_details:
                    ent_rows.append({
                        "Slot #": p.position_index + 1,
                        "Entropy (bits)": f"{p.entropy:.3f}",
                        "Severity": p.severity.upper(),
                        "Dominant Candidate": p.dominant_candidate,
                        "Margin (P1 - P2)": f"{p.dominant_margin:.2f}",
                        "Distinct Candidates": ", ".join(p.distinct_candidates),
                        "Providers": p.participating_providers_count,
                    })
                st.dataframe(pd.DataFrame(ent_rows), width="stretch", hide_index=True)

    # ==========================================================================
    # TAB 5: 👁️ VISUAL EVIDENCE VERIFICATION
    # ==========================================================================
    with tabs[4]:
        st.markdown("#### 👁️ Disputed Zone Sub-Image Extraction & Physical Stroke Verification")
        st.caption(
            "Verifies candidate morphology strictly against physical carved incision strokes (crossbars, stems, loops, descenders). "
            "The model abstains ('indeterminate') when weathering obscures incision geometry. Hypotheses must be reviewed by researchers."
        )

        inspections = bundle.visual_inspections
        if not inspections:
            st.success("✓ No high-dispute zones requiring visual stroke geometry inspection.")
        else:
            for idx, insp in enumerate(inspections):
                with st.expander(f"🔍 Disputed Region #{idx + 1} [{insp.region_id}] — Recommended Verdict: {insp.recommended_verdict.upper()}", expanded=(idx == 0)):
                    st.caption(f"**Bounding Box:** `{insp.bounding_box.to_dict()}` | **Source Pixel Coordinates:** `{insp.pixel_coordinates}`")

                    # 4 Spectral Processing Variants
                    st.markdown("**Four Inscription Processing Variants:**")
                    iv1, iv2, iv3, iv4 = st.columns(4)
                    with iv1:
                        st.caption("Path A: Original Photograph")
                        st.image(insp.crop_original, width="stretch")
                    with iv2:
                        st.caption("Path B: Mild CLAHE")
                        st.image(insp.crop_mild_clahe, width="stretch")
                    with iv3:
                        st.caption("Path C: Denoised Stone")
                        st.image(insp.crop_denoised, width="stretch")
                    with iv4:
                        st.caption("Path D: Adaptive Sauvola")
                        st.image(insp.crop_adaptive, width="stretch")

                    # Stroke geometry hypotheses
                    st.markdown("##### Candidate Stroke Geometry Hypotheses:")
                    for cand, hyp in insp.hypotheses_by_candidate.items():
                        v_col = "#28a745" if hyp.verdict == VisualVerdict.SUPPORTED else (
                            "#dc3545" if hyp.verdict == VisualVerdict.CONTRADICTED else "#ffc107"
                        )
                        st.markdown(
                            f"""
                            <div style="border-left: 4px solid {v_col}; padding: 8px 12px; margin-bottom: 8px; background: rgba(255,255,255,0.03); border-radius: 4px;">
                                <b>Candidate Glyph:</b> <span style="font-size: 1.2rem; color: #C5A15A;">'{cand}'</span> — 
                                <b>Verdict:</b> <span style="color: {v_col}; font-weight: bold;">{hyp.verdict.upper()}</span> (Confidence: {hyp.confidence * 100:.0f}%)<br/>
                                <span style="font-size: 0.88rem; color: #D0D5DD;"><b>Observed Physical Strokes:</b> {hyp.observed_strokes}</span><br/>
                                <span style="font-size: 0.88rem; color: #A8B0BA;"><b>Contradicting / Missing Features:</b> {hyp.contradicting_features}</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    if insp.abstention_reason:
                        st.info(f"⚖️ **Abstention Explanation:** {insp.abstention_reason}")

    # ==========================================================================
    # TAB 6: 📜 HISTORICAL CONTEXT VALIDATION
    # ==========================================================================
    with tabs[5]:
        st.markdown("#### 📜 Historical Context & Epigraphical Lexicon Cross-Referencing")
        st.warning(
            "🏛️ **Hard Scientific Constraint**: Historical catalogue matches provide collateral Bayesian prior support, "
            "but **NEVER** override physical carved stroke incisions. If physical incision evidence contradicts a royal title, "
            "the visual evidence prevails."
        )

        hv = bundle.historical_validation
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            render_metric_card("Rulers Identified", str(hv.total_rulers_detected), "Royal monarchs catalogued")
        with h2:
            render_metric_card("Dynasties Identified", str(hv.total_dynasties_detected), "Chola, Pandya, Pallava, Chera")
        with h3:
            render_metric_card("Administrative Lexicon", str(len(hv.administrative_terms_detected)), "Epigraphical vocabulary terms")
        with h4:
            render_metric_card("Lexical Ratio", f"{hv.epigraphical_vocabulary_ratio * 100:.1f}%", "Tokens matching ancient lexicon")

        if hv.supporting_matches:
            st.markdown("##### Supporting Catalogue Matches & Bibliographic Citations:")
            h_rows = []
            for m in hv.supporting_matches:
                h_rows.append({
                    "Entity / Term": m.matched_entity,
                    "Category": m.category.title(),
                    "Standardized Form": m.standard_form,
                    "Confidence": f"{m.confidence * 100:.0f}%",
                    "Scholarly Citation": m.bibliographic_citation,
                    "Historical Notes": m.historical_notes,
                })
            st.dataframe(pd.DataFrame(h_rows), width="stretch", hide_index=True)
        else:
            st.info("ℹ️ No specific royal names or administrative formula matches detected in candidate text.")

    # ==========================================================================
    # TAB 7: ⚡ FUSED CANDIDATE
    # ==========================================================================
    with tabs[6]:
        st.markdown("#### ⚡ Fused Inscription Transcription & Multi-Strategy Benchmark")
        st.caption(
            "Synthesized epigraphical text combining evidence across all engines via sequence transition networks (ROVER) "
            "and calibrated reliability weighting with correlation discounting."
        )

        f_res = bundle.fused_result

        # Primary fused reading
        st.markdown("##### 🏆 Primary Recommended Fused Reading:")
        st.markdown(
            f"<div class='transcription-box' style='border-left-color: #4CAF50; font-family: monospace; font-size: 1.15rem;'>{f_res.fused_text}</div>",
            unsafe_allow_html=True
        )
        if st.button("📋 Adopt Primary Fused Reading into Workbench", key="adopt_primary_fused"):
            st.session_state.analysis_output = analyze_custom_epigraph_text(
                f_res.fused_text,
                provenance="K-IEFE Evidence Fusion Reading",
                allow_preset_match=False,
                source_identity=st.session_state.source_identity
            )
            st.success("✓ Adopted Primary Fused Reading across all studios!")
            st.rerun()

        # Conservative epigraphical reading with lacunae
        st.markdown("##### 📜 Conservative Scholarly Reading (Preserves Ambiguous Lacunae `[c1/c2]`):")
        st.markdown(
            f"<div class='transcription-box' style='border-left-color: #C5A15A; font-family: monospace; font-size: 1.05rem;'>{f_res.conservative_text}</div>",
            unsafe_allow_html=True
        )
        if st.button("📋 Adopt Conservative Reading (with Bracketed Lacunae)", key="adopt_conservative_fused"):
            st.session_state.analysis_output = analyze_custom_epigraph_text(
                f_res.conservative_text,
                provenance="K-IEFE Conservative Reading",
                allow_preset_match=False,
                source_identity=st.session_state.source_identity
            )
            st.success("✓ Adopted Conservative Reading into Workbench!")
            st.rerun()

        # Multi-strategy comparison table
        st.markdown("##### 🔬 Independent Fusion Strategy Comparison:")
        strat_rows = []
        for s_key, s_obj in bundle.all_fusion_strategies.items():
            strat_rows.append({
                "Strategy": s_key.replace("strategy_", "").replace("_", " ").title(),
                "Candidate Text": s_obj.fused_text[:80] + ("..." if len(s_obj.fused_text) > 80 else ""),
                "Length (Chars)": len(s_obj.fused_text),
                "Baseline Provider": s_obj.strongest_baseline_provider,
                "Relative CER Reduction": f"{s_obj.cer_reduction_pct:+.1f}%" if s_obj.cer_reduction_pct is not None else "Baseline",
            })
        st.dataframe(pd.DataFrame(strat_rows), width="stretch", hide_index=True)

        if f_res.competing_sequences:
            st.markdown("##### 🧬 Competing Sequence Hypotheses (Beam Search):")
            for rank, (seq, score) in enumerate(f_res.competing_sequences, start=1):
                st.caption(f"**Hypothesis #{rank}** (Score: `{score:.3f}`): `{seq}`")

    # ==========================================================================
    # TAB 8: ⚖️ RESEARCHER ADJUDICATION
    # ==========================================================================
    with tabs[7]:
        st.markdown("#### ⚖️ Epigraphist Adjudication & Scholarly Review Workstation")
        st.caption(
            "Scholarly review interface for disputed inscriptional zones. Original provider predictions are immutable. "
            "All researcher modifications are recorded with reviewer identity, timestamp, and rationale."
        )

        review_mgr: Optional[ResearcherReviewManager] = getattr(st.session_state, "kiefe_adjudication_manager", None)
        if review_mgr is None:
            review_mgr = ResearcherReviewManager()
            for it in bundle.adjudication_items:
                review_mgr.add_item(it)
            st.session_state.kiefe_adjudication_manager = review_mgr

        items = review_mgr.get_all_items()
        if not items:
            st.success("✓ No disputed regions requiring scholar adjudication. All models and visual rules agree!")
        else:
            col_rev_name, col_filter = st.columns([2, 1])
            with col_rev_name:
                reviewer_name = st.text_input(
                    "Epigraphist / Reviewer Identity:",
                    value=st.session_state.get("kiefe_reviewer_name", "Epigraphist / Digital Epigraphy Researcher"),
                    key="kiefe_reviewer_name_input"
                )
                st.session_state.kiefe_reviewer_name = reviewer_name

            with col_filter:
                filter_choice = st.selectbox(
                    "Adjudication Filter:",
                    ["All Zones", "Pending Review Only", "Resolved Only"],
                    key="kiefe_adj_filter"
                )

            filtered_items = [
                it for it in items
                if (filter_choice == "All Zones")
                or (filter_choice == "Pending Review Only" and it.current_decision == AdjudicationDecision.PENDING.value)
                or (filter_choice == "Resolved Only" and it.current_decision != AdjudicationDecision.PENDING.value)
            ]

            st.caption(f"Showing {len(filtered_items)} of {len(items)} disputed zones.")

            for it in filtered_items:
                with st.container():
                    st.markdown(
                        f"""
                        <div style="border: 1px solid #3E4651; border-radius: 8px; padding: 12px; margin-bottom: 12px; background: rgba(30,35,42,0.4);">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <b>Disputed Zone #{it.region_index + 1} ({it.item_id})</b>
                                <span class="badge {'badge-success' if it.current_decision != 'pending' else 'badge-gold'}">
                                    Status: {it.current_decision.upper()}
                                </span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    c_crop, c_details = st.columns([1, 2])
                    with c_crop:
                        # Extract region crop from ocr_target
                        x1, y1, x2, y2 = it.pixel_crop_coords
                        w_t, h_t = ocr_target.size
                        x1_c, y1_c = max(0, x1), max(0, y1)
                        x2_c, y2_c = min(w_t, max(x1_c + 10, x2)), min(h_t, max(y1_c + 10, y2))
                        crop_img = ocr_target.crop((x1_c, y1_c, x2_c, y2_c))
                        st.image(crop_img, caption=f"Zone Crop ({x2_c - x1_c} × {y2_c - y1_c} px)", width="stretch")

                    with c_details:
                        st.markdown(f"**Dispute Reason:** {it.disagreement_reason}")
                        st.markdown(f"**Consensus Entropy:** `{it.consensus_entropy:.3f} bits`")
                        st.markdown(f"**Engine Proposals:** {', '.join([f'{k}: {v}' for k, v in it.provider_candidates.items()])}")
                        st.markdown(f"**Current Reading:** <span style='font-size: 1.2rem; color: #C5A15A;'><b>'{it.approved_text or it.fused_candidate}'</b></span>", unsafe_allow_html=True)

                        # Action buttons
                        b1, b2, b3, b4 = st.columns(4)
                        with b1:
                            if st.button(f"✓ Accept Fused ('{it.fused_candidate}')", key=f"btn_accept_fus_{it.item_id}"):
                                review_mgr.record_decision(
                                    it.item_id,
                                    AdjudicationDecision.ACCEPTED_FUSION,
                                    it.fused_candidate,
                                    reviewer=reviewer_name,
                                    rationale="Accepted fused candidate reading."
                                )
                                st.success("✓ Accepted fusion candidate!")
                                st.rerun()

                        with b2:
                            if st.button("🚫 Mark Illegible [---]", key=f"btn_mark_ill_{it.item_id}"):
                                review_mgr.record_decision(
                                    it.item_id,
                                    AdjudicationDecision.MARK_ILLEGIBLE,
                                    "[---]",
                                    reviewer=reviewer_name,
                                    rationale="Stone surface physically damaged or abraded; illegible."
                                )
                                st.warning("Recorded as physically illegible [---]")
                                st.rerun()

                        with b3:
                            if st.button("⏳ Defer for Field Study", key=f"btn_defer_{it.item_id}"):
                                review_mgr.record_decision(
                                    it.item_id,
                                    AdjudicationDecision.DEFERRED,
                                    it.approved_text or it.fused_candidate,
                                    reviewer=reviewer_name,
                                    rationale="Deferred for in-situ field re-examination."
                                )
                                st.info("Deferred for field re-examination.")
                                st.rerun()

                        with b4:
                            # Dropdown to select a specific engine's reading
                            cand_options = [c for c in it.alternative_readings if c and c != "-"]
                            if cand_options:
                                sel_cand = st.selectbox(
                                    "Adopt Engine Reading:",
                                    cand_options,
                                    key=f"sel_cand_{it.item_id}"
                                )
                                if st.button("📋 Adopt", key=f"btn_adopt_cand_{it.item_id}"):
                                    review_mgr.record_decision(
                                        it.item_id,
                                        AdjudicationDecision.ACCEPTED_PROVIDER,
                                        sel_cand,
                                        reviewer=reviewer_name,
                                        rationale=f"Selected engine candidate '{sel_cand}'."
                                    )
                                    st.success(f"Adopted candidate '{sel_cand}'!")
                                    st.rerun()

                        # Manual Scholar Edit Input
                        with st.expander("✏️ Scholar Manual Edit & Rationale", expanded=False):
                            custom_txt = st.text_input(
                                "Scholar Transcribed Glyph(s):",
                                value=it.approved_text or it.fused_candidate,
                                key=f"scholar_txt_{it.item_id}"
                            )
                            rationale_txt = st.text_input(
                                "Palaeographical Rationale / Epigraphical Justification:",
                                value="Manual correction based on stone groove incisions and historical formula.",
                                key=f"scholar_rat_{it.item_id}"
                            )
                            if st.button("💾 Save Scholar Edit", key=f"btn_save_scholar_{it.item_id}"):
                                review_mgr.record_decision(
                                    it.item_id,
                                    AdjudicationDecision.EDITED_SCHOLAR,
                                    custom_txt.strip(),
                                    reviewer=reviewer_name,
                                    rationale=rationale_txt.strip()
                                )
                                st.success(f"✓ Saved scholar edit '{custom_txt.strip()}'!")
                                st.rerun()

                        # Revision audit trail for this item
                        if it.revision_history:
                            with st.expander(f"📜 Item Revision History ({len(it.revision_history)} revisions)", expanded=False):
                                for rev in reversed(it.revision_history):
                                    st.caption(
                                        f"**Rev #{rev.revision_id}** — `{rev.timestamp}` by *{rev.reviewer_identity}* | "
                                        f"**Decision:** `{rev.decision}` | **Reading:** `{rev.approved_reading}` | **Rationale:** {rev.rationale}"
                                    )
                    st.markdown("---")

            # Apply all approved adjudications to Workbench
            st.markdown("##### 🚀 Apply Adjudications to Master Workbench:")
            if st.button("💾 Apply All Approved Adjudications to Master Transcription Workbench", width="stretch", key="btn_apply_all_adj"):
                base_text = list(bundle.fused_result.fused_text)
                for it in items:
                    if it.current_decision != AdjudicationDecision.PENDING.value and it.approved_text:
                        pos = it.region_index
                        if 0 <= pos < len(base_text):
                            base_text[pos] = it.approved_text

                reconciled_text = "".join(base_text)
                st.session_state.analysis_output = analyze_custom_epigraph_text(
                    reconciled_text,
                    provenance=f"Epigraphist Adjudicated Reading ({reviewer_name})",
                    allow_preset_match=False,
                    source_identity=st.session_state.source_identity
                )
                st.success("✓ Updated master transcription workbench with all scholar adjudications!")
                st.rerun()

    # ==========================================================================
    # TAB 9: 🛡️ AUDIT TRAIL & ACTIVE LEARNING
    # ==========================================================================
    with tabs[8]:
        st.markdown("#### 🛡️ Cryptographic Evidence Audit Trail & Active Learning")
        st.caption(
            "Verifiable cryptographic evidence dossier ensuring academic reproducibility. "
            "Ingests verified researcher adjudications into active learning datasets for OCR fine-tuning."
        )

        rec = bundle.audit_record

        # Cryptographic Signature Banner
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.7); border: 2px solid #38BDF8; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
                <div style="font-size: 0.82rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">
                    Cryptographic SHA-256 Audit Signature
                </div>
                <div style="font-family: monospace; font-size: 1.15rem; color: #38BDF8; word-break: break-all; margin: 6px 0;">
                    {rec.cryptographic_sha256}
                </div>
                <div style="font-size: 0.85rem; color: #CBD5E1;">
                    🛡️ Deterministic hash of source image ID, preprocessing variant, raw engine predictions, and reliability matrices.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        a1, a2, a3, a4 = st.columns(4)
        with a1:
            st.metric("Execution ID", rec.execution_id)
        with a2:
            st.metric("Timestamp", rec.timestamp.split()[1] if " " in rec.timestamp else rec.timestamp)
        with a3:
            st.metric("Source Image ID", rec.source_identity.get("source_image_id", "—")[:12] + "...")
        with a4:
            st.metric("Model Versions Tracked", len(rec.model_versions))

        # Full Audit JSON Download
        audit_json = rec.to_json()
        st.download_button(
            "📥 Download Cryptographic Audit Dossier (.JSON)",
            data=audit_json,
            file_name=f"kallvettu_audit_{rec.execution_id}.json",
            mime="application/json",
            width="stretch",
            key="btn_download_audit_dossier"
        )

        st.markdown("---")
        st.markdown("#### 🎓 Active Learning Ground-Truth Dataset Pipeline")
        st.warning(
            "🛡️ **Strict Epigraphical Quality Guardrail**:\n\n"
            "Only researcher-approved or scholar-edited readings are eligible for ingestion into the training dataset. "
            "Raw unverified model predictions are strictly prohibited to prevent data poisoning."
        )

        al_mgr = ActiveLearningManager()
        existing_annots = al_mgr.load_all_annotations()

        al_c1, al_c2 = st.columns(2)
        with al_c1:
            st.metric("Current Verified Specimens in Training Dataset", len(existing_annots))
            st.caption(f"Storage path: `{al_mgr.jsonl_path}`")

        with al_c2:
            adj_items = getattr(st.session_state.kiefe_adjudication_manager, "get_all_items", lambda: [])()
            eligible_items = [
                it for it in adj_items
                if it.current_decision in [
                    AdjudicationDecision.ACCEPTED_FUSION.value,
                    AdjudicationDecision.ACCEPTED_PROVIDER.value,
                    AdjudicationDecision.EDITED_SCHOLAR.value,
                ] and it.approved_text
            ]
            st.metric("Adjudications Eligible for Dataset Ingestion", len(eligible_items))

        # Ingestion button
        if eligible_items:
            if st.button("💾 Ingest Approved Adjudications into Active Learning Dataset", width="stretch", key="btn_ingest_al"):
                ingested_cnt = 0
                for it in eligible_items:
                    record = VerifiedAnnotationRecord(
                        specimen_id=f"specimen_{it.item_id}_{int(time.time())}",
                        inscription_id=bundle.source_identity.source_image_id,
                        source_image_id=bundle.source_identity.source_image_id,
                        image_sha256=bundle.source_identity.image_sha256,
                        crop_coordinates=it.pixel_crop_coords,
                        normalized_bbox=(it.bounding_box.ymin, it.bounding_box.xmin, it.bounding_box.ymax, it.bounding_box.xmax),
                        script=bundle.source_identity.preset_id or "tamil_brahmi",
                        original_candidates=it.provider_candidates,
                        final_approved_reading=it.approved_text,
                        reviewer_identity=st.session_state.get("kiefe_reviewer_name", "Epigraphist"),
                        review_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                        notes=f"Decision: {it.current_decision}. Entropy: {it.consensus_entropy:.3f}."
                    )
                    al_mgr.store_verified_annotation(record)
                    ingested_cnt += 1

                st.success(f"✓ Successfully ingested {ingested_cnt} verified annotation specimens into {al_mgr.jsonl_path}!")
                st.rerun()

        # Export dataset buttons
        st.markdown("##### Export Verified Dataset for Model Fine-Tuning (CRNN, TrOCR, YOLO):")
        ex1, ex2 = st.columns(2)
        with ex1:
            json_export = al_mgr.export_for_model_training(output_format="json")
            st.download_button(
                "📥 Export Active Learning Dataset (.JSON)",
                data=json_export,
                file_name="kallvettu_active_learning_dataset.json",
                mime="application/json",
                width="stretch",
                key="btn_export_al_json"
            )
        with ex2:
            tsv_export = al_mgr.export_for_model_training(output_format="tsv")
            st.download_button(
                "📥 Export Active Learning Dataset (.TSV)",
                data=tsv_export,
                file_name="kallvettu_active_learning_dataset.tsv",
                mime="text/tab-separated-values",
                width="stretch",
                key="btn_export_al_tsv"
            )

