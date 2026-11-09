"""
Historical Context Validation Module for Kallvettu (K-IEFE).
Cross-references candidate OCR readings against epigraphical lexicons, monarch catalogues,
dynastic titles, administrative terms, and corpus references.

HARD SCIENTIFIC RULE:
A historical knowledge-base match must NEVER automatically replace or override the visual reading.
Historical concordance provides Bayesian collateral evidence, but stone incisions are the
primary ground truth in epigraphical science. Every assertion retains its scholarly bibliographic citation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
import re

from nlp_translation_engine import EPIGRAPHY_LEXICON
from evaluation_engine import EVALUATION_DATASET


# Comprehensive Historical Ruler & Dynasty Epigraphical Catalogue
DYNASTY_RULER_CATALOGUE = {
    # Chola Dynastic Epigraphs
    "ராஜராஜ": {
        "full_title": "முதலாம் இராஜராஜ சோழன் (Rajaraja Chola I)",
        "regnal_years": "c. 985 – 1014 CE",
        "dynasty": "Imperial Chola",
        "common_titles": ["கோப்பரகேசரி", "ராஜகேசரி", "மும்முடிச் சோழன்", "சிவபாதசேகரன்"],
        "standard_meikeerthi": "திருமகள் போல பெருநிலச் செல்வியும்",
        "citation": "South Indian Inscriptions (SII) Vol. II, No. 1",
    },
    "ராஜேந்திர": {
        "full_title": "முதலாம் இராஜேந்திர சோழன் (Rajendra Chola I)",
        "regnal_years": "c. 1012 – 1044 CE",
        "dynasty": "Imperial Chola",
        "common_titles": ["கோப்பரகேசரி", "கங்கை கொண்ட சோழன்", "உத்தம சோழன்"],
        "standard_meikeerthi": "திருமன்னி வளர இருநில மடந்தையும்",
        "citation": "South Indian Inscriptions (SII) Vol. II, No. 20",
    },
    "பராந்தக": {
        "full_title": "முதலாம் பராந்தக சோழன் (Parantaka Chola I)",
        "regnal_years": "c. 907 – 955 CE",
        "dynasty": "Early Imperial Chola",
        "common_titles": ["மதுரையும் ஈழமும் கொண்ட கோப்பரகேசரி"],
        "citation": "Annual Report on Epigraphy (ARE) 1904, No. 283 (Uttiramerur)",
    },
    # Pandya Dynastic Epigraphs
    "நெடுஞ்செழியன்": {
        "full_title": "நெடுஞ்செழியன் (Pandyan King Neduncheliyan)",
        "regnal_years": "c. 3rd Century BCE – 2nd Century BCE",
        "dynasty": "Early Sangam Pandya",
        "common_titles": ["வழுதி", "மாறன்", "செழியன்"],
        "citation": "Mahadevan, I. (2003) Early Tamil Epigraphy, Mangulam Inscription No. 1",
    },
    # Chera Dynastic Epigraphs
    "பெருங்கடுங்கோ": {
        "full_title": "சேர மன்னன் பெருங்கடுங்கோ (Chera King Perunkadungo)",
        "regnal_years": "c. 2nd Century CE",
        "dynasty": "Irumporai Chera",
        "common_titles": ["கொ ஆதன் செல்லிரும்பொறை மகன்"],
        "citation": "Mahadevan, I. (2003) Pugalur Chera Inscriptions No. 1–4",
    },
    # Pallava Epigraphs
    "மகேந்திரவர்மன்": {
        "full_title": "முதலாம் மகேந்திரவர்ம பல்லவன் (Mahendravarman I)",
        "regnal_years": "c. 600 – 630 CE",
        "dynasty": "Pallava",
        "common_titles": ["விசித்திரசித்தன்", "குணபரன்", "மத்தவிலாசன்"],
        "citation": "Epigraphia Indica (EI) Vol. IV, Mandagapattu Cave Epigraph",
    },
    "நந்திவர்மன்": {
        "full_title": "இரண்டாம் நந்திவர்ம பல்லவமல்லன் (Nandivarman II)",
        "regnal_years": "c. 731 – 796 CE",
        "dynasty": "Pallava",
        "common_titles": ["பல்லவ மல்லன்", "வைணவ திலகன்"],
        "citation": "South Indian Inscriptions (SII) Vol. III, Kasakudi Plates",
    }
}

# Administrative, Land-Tenure & Constitutional Inscriptional Terminology
ADMINISTRATIVE_TERMINOLOGY = {
    "குடவோலை": {
        "term": "குடவோலை (Kudavolai)",
        "category": "Democratic Electoral System",
        "meaning": "Secret pot-ballot system for electing village assembly (சபை) executive committees (வாரியம்).",
        "citation": "Uttiramerur Chola Temple Inscription (Parantaka I, 921 CE)",
    },
    "பள்ளிச்சந்தம்": {
        "term": "பள்ளிச்சந்தம் (Pallichandam)",
        "category": "Religious Land Grant",
        "meaning": "Tax-exempt agricultural land endowment deeded to Jain or Buddhist monastic settlements.",
        "citation": "Pulankurichi Stone Inscriptions & Epigraphia Indica Vol. XXXIX",
    },
    "பிரம்மதேயம்": {
        "term": "பிரம்மதேயம் (Brahmadeyam)",
        "category": "Royal Endowment",
        "meaning": "Royal tax-free land granted to Vedic scholars and assembly scholars.",
        "citation": "Anbil and Karandai Copper Plates",
    },
    "ஆநிரை": {
        "term": "ஆநிரை / தொறு (Cattle Herd)",
        "category": "Hero Stone / Warfare",
        "meaning": "Village cattle herd defended during border raids (தொறுப்பூசல் / கரந்தை).",
        "citation": "Chengam Hero Stones (செங்கம் நடுகற்கள்), Tamil Nadu State Dept. of Archaeology",
    },
    "கழஞ்சு": {
        "term": "கழஞ்சு (Kalanju)",
        "category": "Epigraphical Weight / Currency",
        "meaning": "Standard ancient gold coin measurement equivalent to ~3.2 grams (one Molucca bean weight).",
        "citation": "Epigraphia Indica Vol. VII (Thanjavur Brihadisvara Inscriptions)",
    },
    "வரிப்பொத்தகக்கணக்கு": {
        "term": "வரிப்பொத்தகக் கணக்கு (Land Revenue Ledger)",
        "category": "Imperial Revenue Department",
        "meaning": "Central Chola imperial treasury land register and revenue documentation bureau.",
        "citation": "South Indian Inscriptions (SII) Vol. II",
    },
}


@dataclass
class HistoricalEvidenceMatch:
    """Scholarly evidence item matching candidate text against the epigraphical corpus."""
    candidate_term: str
    match_status: str                # 'supporting', 'contradicting', 'unavailable'
    entity_type: str                 # 'ruler', 'dynasty', 'administrative_term', 'lexicon'
    canonical_entry: str
    historical_commentary: str
    bibliographic_citation: str
    cannot_override_visual: bool = True  # Strict architectural guardrail

    @property
    def matched_entity(self) -> str:
        return self.candidate_term

    @property
    def category(self) -> str:
        return self.entity_type

    @property
    def standard_form(self) -> str:
        return self.canonical_entry

    @property
    def confidence(self) -> float:
        return 0.95 if self.match_status == "supporting" else 0.5

    @property
    def historical_notes(self) -> str:
        return self.historical_commentary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_term": self.candidate_term,
            "match_status": self.match_status,
            "entity_type": self.entity_type,
            "canonical_entry": self.canonical_entry,
            "historical_commentary": self.historical_commentary,
            "bibliographic_citation": self.bibliographic_citation,
            "cannot_override_visual": self.cannot_override_visual,
        }


@dataclass
class HistoricalValidationSummary:
    """Aggregated historical context report for an inscription candidate reading."""
    total_matches_found: int
    supporting_matches: List[HistoricalEvidenceMatch]
    contradicting_matches: List[HistoricalEvidenceMatch]
    primary_dynasty_detected: Optional[str]
    identified_monarch: Optional[str]
    administrative_terms_detected: List[str]
    epigraphical_notes: str

    @property
    def total_rulers_detected(self) -> int:
        return 1 if self.identified_monarch else 0

    @property
    def total_dynasties_detected(self) -> int:
        return 1 if self.primary_dynasty_detected else 0

    @property
    def epigraphical_vocabulary_ratio(self) -> float:
        return min(1.0, len(self.supporting_matches) * 0.15)

    @property
    def corpus_cooccurrence_support(self) -> float:
        return 1.0 if self.identified_monarch and self.primary_dynasty_detected else (0.5 if self.supporting_matches else 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_matches_found": self.total_matches_found,
            "primary_dynasty_detected": self.primary_dynasty_detected,
            "identified_monarch": self.identified_monarch,
            "administrative_terms_detected": self.administrative_terms_detected,
            "supporting_matches": [m.to_dict() for m in self.supporting_matches],
            "contradicting_matches": [m.to_dict() for m in self.contradicting_matches],
            "epigraphical_notes": self.epigraphical_notes,
        }


class HistoricalContextValidator:
    """
    Validates inscriptional candidates against authentic South Indian epigraphical databases.
    """

    def __init__(self):
        pass

    def validate_candidate_text(
        self,
        candidate_text: str,
        script_hint: Optional[str] = None
    ) -> HistoricalValidationSummary:
        """
        Cross-checks candidate reading against historical dictionaries and ruler records.
        """
        supporting: List[HistoricalEvidenceMatch] = []
        contradicting: List[HistoricalEvidenceMatch] = []
        dynasty_detected = None
        monarch_detected = None
        admin_terms = []

        if not candidate_text or not candidate_text.strip():
            return HistoricalValidationSummary(
                total_matches_found=0,
                supporting_matches=[],
                contradicting_matches=[],
                primary_dynasty_detected=None,
                identified_monarch=None,
                administrative_terms_detected=[],
                epigraphical_notes="Empty text; historical validation unavailable."
            )

        clean_text = candidate_text.strip()
        tokens = clean_text.split()

        # 1. Check Ruler & Dynastic Records
        for ruler_key, ruler_info in DYNASTY_RULER_CATALOGUE.items():
            if ruler_key in clean_text:
                supporting.append(HistoricalEvidenceMatch(
                    candidate_term=ruler_key,
                    match_status="supporting",
                    entity_type="ruler",
                    canonical_entry=ruler_info["full_title"],
                    historical_commentary=f"Regnal Period: {ruler_info['regnal_years']}. Dynasty: {ruler_info['dynasty']}.",
                    bibliographic_citation=ruler_info["citation"]
                ))
                monarch_detected = ruler_info["full_title"]
                dynasty_detected = ruler_info["dynasty"]

            # Check dynastic titles
            for title in ruler_info.get("common_titles", []):
                if title in clean_text:
                    supporting.append(HistoricalEvidenceMatch(
                        candidate_term=title,
                        match_status="supporting",
                        entity_type="ruler_title",
                        canonical_entry=f"Royal Title of {ruler_info['full_title']}",
                        historical_commentary=f"Formulaic imperial invocation associated with {ruler_info['dynasty']}.",
                        bibliographic_citation=ruler_info["citation"]
                    ))

        # 2. Check Administrative Terminology
        for term_key, term_info in ADMINISTRATIVE_TERMINOLOGY.items():
            if term_key in clean_text:
                admin_terms.append(term_key)
                supporting.append(HistoricalEvidenceMatch(
                    candidate_term=term_key,
                    match_status="supporting",
                    entity_type="administrative_term",
                    canonical_entry=term_info["term"],
                    historical_commentary=f"Category: {term_info['category']}. {term_info['meaning']}",
                    bibliographic_citation=term_info["citation"]
                ))

        # 3. Check Epigraphical Lexicon Vocabulary
        for token in tokens:
            # Strip punctuation
            tok_clean = re.sub(r"[^\u0B80-\u0BFF]", "", token)
            if not tok_clean:
                continue

            if tok_clean in EPIGRAPHY_LEXICON:
                item = EPIGRAPHY_LEXICON[tok_clean]
                supporting.append(HistoricalEvidenceMatch(
                    candidate_term=tok_clean,
                    match_status="supporting",
                    entity_type="lexicon_vocabulary",
                    canonical_entry=f"{tok_clean} ({item.get('modern_tamil', '')})",
                    historical_commentary=f"Grammar: {item.get('pos', 'Noun')}. Context: {item.get('etymology', '')}",
                    bibliographic_citation="Tamil Lexicon (University of Madras) & Epigraphia Indica"
                ))

        # 4. Check for Chronological / Script Anachronism Contradictions
        if script_hint and "brahmi" in script_hint.lower():
            # If text claims to be Sangam Tamil-Brahmi but contains High Chola titles
            if any(t in clean_text for t in ["ராஜராஜ", "கங்கை கொண்ட", "உத்தம சோழன்"]):
                contradicting.append(HistoricalEvidenceMatch(
                    candidate_term="Medieval Chola Title in Brahmi Script",
                    match_status="contradicting",
                    entity_type="chronological_anachronism",
                    canonical_entry="Chronological Discontinuity",
                    historical_commentary="High Chola imperial titles (c. 10th-11th Century CE) cannot occur in Tamil-Brahmi script (c. 3rd BCE - 2nd CE). Suggests OCR hallucination.",
                    bibliographic_citation="Epigraphia Indica & K.A. Nilakanta Sastri (The Colas)"
                ))

        notes = (
            f"Found {len(supporting)} authentic historical corpus concordances. "
            "Evidence acts as collateral Bayesian support and does not override lithic stone incision strokes."
        )

        return HistoricalValidationSummary(
            total_matches_found=len(supporting) + len(contradicting),
            supporting_matches=supporting,
            contradicting_matches=contradicting,
            primary_dynasty_detected=dynasty_detected,
            identified_monarch=monarch_detected,
            administrative_terms_detected=admin_terms,
            epigraphical_notes=notes
        )
