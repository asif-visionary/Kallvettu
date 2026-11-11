"""
Reusable UI Components for Kallvettu Archaeological Research Workbench.
Ensures visual consistency, accessible contrast, and standardized layout
across all 8 studios in accordance with Phase 2 specifications.
"""

from typing import Dict, Any, List, Optional
import streamlit as st
import pandas as pd
from PIL import Image


def render_workbench_header(
    active_studio: str,
    api_configured: bool = False,
    model_name: str = "gemini-1.5-flash",
    provider_status: Optional[str] = None
):
    """Renders the top application brand header with studio indicator & API connection status."""
    if provider_status:
        status_label = provider_status
        status_class = "ready" if ("Active" in provider_status or "Live" in provider_status) else "offline"
    else:
        status_class = "ready" if api_configured else "offline"
        status_label = f"✨ Gemini Live ({model_name})" if api_configured else "🏛️ Offline Engine Active"

    st.markdown(
        f"""
        <div class="workbench-header">
            <div>
                <h1 class="brand-title">
                    🏛️ KALLVETTU <span class="tamil">| கல்வெட்டு</span>
                </h1>
                <div class="brand-subtitle">
                    Epigraphical Computer Vision & Classical Tamil NLP Research Workbench • <b>{active_studio}</b>
                </div>
            </div>
            <div>
                <span class="header-status-pill {status_class}">
                    {status_label}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_section_header(title: str, subtitle: str = "", icon: str = ""):
    """Renders standardized studio section titles with optional icon and description."""
    icon_prefix = f"{icon} " if icon else ""
    st.markdown(f"### {icon_prefix}{title}")
    if subtitle:
        st.caption(subtitle)


def render_metric_card(label: str, value: Any, hint: str = ""):
    """Renders an individual dashboard metric card in dark graphite and muted bronze."""
    hint_html = f'<div class="metric-card-hint">{hint}</div>' if hint else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card-label">{label}</div>
            <div class="metric-card-value">{value}</div>
            {hint_html}
        </div>
        """,
        unsafe_allow_html=True
    )


def render_empty_state(icon: str, title: str, description: str, hint: str = ""):
    """Standardized empty state message when no active inscription or data is loaded."""
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-state-icon">{icon}</div>
            <div class="empty-state-title">{title}</div>
            <div class="empty-state-desc">{description}</div>
            {f'<div style="font-size: 0.8rem; color: #C5A15A;">{hint}</div>' if hint else ''}
        </div>
        """,
        unsafe_allow_html=True
    )


def render_provenance_badge(provenance: str, is_verified: bool = False):
    """Renders an epigraphical data provenance badge distinguishing AI vs human verification."""
    badge_style = "badge-success" if is_verified else "badge-gold"
    icon = "✓" if is_verified else "🔍"
    st.markdown(
        f"""
        <div style="margin-bottom: 10px;">
            <span class="badge {badge_style}">
                {icon} Provenance: {provenance}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_epigraph_metadata_grid(data: Dict[str, Any]):
    """Renders standard epigraphical metadata (script, era, dynasty, category, location)."""
    script_ta = data.get("script_ta", "தமிழ் பிராமி")
    script_en = data.get("script_en", "Tamil-Brahmi")
    era_ta = data.get("era_ta", "வரலாற்றுக் காலம்")
    era_en = data.get("era_en", "")
    dynasty_ta = data.get("dynasty_ta", "பண்டைய தமிழகம்")
    dynasty_en = data.get("dynasty_en", "")
    category = data.get("category", "வரலாற்றுக் கல்வெட்டு")
    location = data.get("location_ta", data.get("location_en", "கண்டறியப்பட்ட களம்"))

    st.markdown(
        f"""
        <div class="meta-grid">
            <div class="meta-item">
                <div class="meta-label">Epigraphical Script</div>
                <div class="meta-value">{script_ta}</div>
                <div style="font-size: 0.78rem; color: #A8B0BA;">{script_en}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Historical Era / Period</div>
                <div class="meta-value">{era_ta}</div>
                <div style="font-size: 0.78rem; color: #A8B0BA;">{era_en}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Dynasty / Monarch</div>
                <div class="meta-value">{dynasty_ta}</div>
                <div style="font-size: 0.78rem; color: #A8B0BA;">{dynasty_en}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Classification</div>
                <div class="meta-value">{category}</div>
                <div style="font-size: 0.78rem; color: #A8B0BA;">{location}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def build_lexical_dataframe(words_list: List[Dict[str, Any]], search_query: str = "") -> pd.DataFrame:
    """
    Transforms raw word entries into an interactive, sortable Pandas DataFrame
    with comprehensive grammatical and etymological columns.
    """
    rows = []
    for w in words_list:
        word = w.get("word", "")
        modern = w.get("modern", "")
        english = w.get("english", "")
        grammar = w.get("pos", w.get("grammar", "பெயர்ச்சொல்"))
        case_marker = w.get("case_marker", "வேற்றுமை உருபற்ற எழுவாய் (Nominative)")
        root = w.get("root", "—")
        notes = w.get("notes", "")

        # Optional search filtering
        if search_query:
            q = search_query.strip().lower()
            if not (q in word.lower() or q in modern.lower() or q in english.lower() or q in notes.lower() or q in root.lower()):
                continue

        rows.append({
            "Ancient Word (பண்டைய சொல்)": word,
            "Root / Stem (பகுதி)": root,
            "Grammar / POS (இலக்கணம்)": grammar,
            "Case Suffix (வேற்றுமை)": case_marker,
            "Modern Tamil (நவீன பொருள்)": modern,
            "English Meaning": english,
            "Etymological Notes (குறிப்பு)": notes
        })

    return pd.DataFrame(rows)

