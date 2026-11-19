"""
Researcher Adjudication & Epigraphical Review Module for Kallvettu (K-IEFE).
Provides interactive data models and state management for scholarly adjudication
over disputed inscriptional zones.

PRINCIPLES:
- Original provider outputs are immutable and never overwritten.
- All scholar modifications record complete revision provenance (editor, timestamp, rationale).
- Supports 5 adjudication actions: ACCEPT, EDIT, REJECT, MARK_ILLEGIBLE, DEFER.
"""

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from PIL import Image

from services.ai.common.base_provider import BoundingBox
from services.ocr_fusion.visual_verifier import VisualEvidenceHypothesis
from services.ocr_fusion.historical_validator import HistoricalEvidenceMatch


class AdjudicationDecision(str, Enum):
    """Researcher adjudication decisions for a disputed reading."""
    PENDING = "pending"              # Awaiting scholar inspection
    DISPUTED = "disputed"            # Formally marked as contested reading
    REQUIRES_REVIEW = "requires_review" # Flagged for expert review
    ACCEPTED_FUSION = "accepted_fusion" # Fused candidate accepted without changes
    ACCEPTED_PROVIDER = "accepted_provider" # Specific engine's reading selected
    EDITED_SCHOLAR = "edited_scholar"# Manually revised by epigraphist
    REJECTED_ALL = "rejected_all"    # All model candidates rejected as spurious
    MARK_ILLEGIBLE = "mark_illegible"# Stone physically damaged/abraded; recorded as `[---]`
    DEFERRED = "deferred"            # Deferred for in-situ field re-examination


@dataclass
class ReviewRevisionEntry:
    """Audit log entry for an adjudication decision or modification."""
    revision_id: int
    timestamp: str
    reviewer_identity: str
    decision: str                    # AdjudicationDecision
    previous_reading: str
    approved_reading: str
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "revision_id": self.revision_id,
            "timestamp": self.timestamp,
            "reviewer_identity": self.reviewer_identity,
            "decision": self.decision,
            "previous_reading": self.previous_reading,
            "approved_reading": self.approved_reading,
            "rationale": self.rationale,
        }


@dataclass
class AdjudicationItem:
    """Comprehensive workbench packet presented to the scholar for region review."""
    item_id: str
    region_index: int
    bounding_box: BoundingBox
    pixel_crop_coords: Tuple[int, int, int, int]
    provider_candidates: Dict[str, str]       # Raw outputs from each engine
    provider_confidences: Dict[str, Optional[float]]
    fused_candidate: str
    alternative_readings: List[str]
    disagreement_reason: str
    consensus_entropy: float
    visual_hypothesis: Optional[Dict[str, Any]]
    historical_evidence: List[Dict[str, Any]]
    current_decision: str = AdjudicationDecision.PENDING.value
    approved_text: str = ""
    revision_history: List[ReviewRevisionEntry] = field(default_factory=list)
    normalized_entropy: Optional[float] = None
    candidate_distribution: Optional[Dict[str, float]] = None
    participating_providers_count: int = 0
    is_assessable: bool = True
    scholarly_notes: List[str] = field(default_factory=list)
    supporting_physical_evidence: List[str] = field(default_factory=list)
    is_researcher_approved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "region_index": self.region_index,
            "bounding_box": self.bounding_box.to_dict(),
            "pixel_crop_coords": self.pixel_crop_coords,
            "provider_candidates": self.provider_candidates,
            "provider_confidences": {
                k: round(v, 4) if v is not None else None
                for k, v in self.provider_confidences.items()
            },
            "fused_candidate": self.fused_candidate,
            "alternative_readings": self.alternative_readings,
            "disagreement_reason": self.disagreement_reason,
            "consensus_entropy": round(self.consensus_entropy, 4),
            "normalized_entropy": round(self.normalized_entropy, 4) if self.normalized_entropy is not None else None,
            "candidate_distribution": self.candidate_distribution,
            "participating_providers_count": self.participating_providers_count,
            "is_assessable": self.is_assessable,
            "visual_hypothesis": self.visual_hypothesis,
            "historical_evidence": self.historical_evidence,
            "current_decision": self.current_decision,
            "approved_text": self.approved_text,
            "scholarly_notes": self.scholarly_notes,
            "supporting_physical_evidence": self.supporting_physical_evidence,
            "is_researcher_approved": self.is_researcher_approved,
            "revision_history": [r.to_dict() for r in self.revision_history],
        }


class ResearcherReviewManager:
    """
    State manager tracking epigraphical review sessions and adjudication workflows.
    """

    def __init__(self):
        self._items: Dict[str, AdjudicationItem] = {}

    def add_item(self, item: AdjudicationItem) -> None:
        self._items[item.item_id] = item

    def get_item(self, item_id: str) -> Optional[AdjudicationItem]:
        return self._items.get(item_id)

    def get_all_items(self) -> List[AdjudicationItem]:
        return list(self._items.values())

    def get_pending_items(self) -> List[AdjudicationItem]:
        return [it for it in self._items.values() if it.current_decision == AdjudicationDecision.PENDING.value]

    def record_decision(
        self,
        item_id: str,
        decision: AdjudicationDecision,
        approved_text: str,
        reviewer: str = "Epigraphist",
        rationale: str = ""
    ) -> AdjudicationItem:
        """
        Records an immutable decision entry into the adjudication item's audit trail.
        """
        if item_id not in self._items:
            raise KeyError(f"Adjudication item '{item_id}' not found.")

        item = self._items[item_id]
        prev_reading = item.approved_text or item.fused_candidate
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")

        rev_entry = ReviewRevisionEntry(
            revision_id=len(item.revision_history) + 1,
            timestamp=timestamp_str,
            reviewer_identity=reviewer,
            decision=decision.value,
            previous_reading=prev_reading,
            approved_reading=approved_text,
            rationale=rationale or f"Decision: {decision.value}"
        )

        item.current_decision = decision.value
        item.approved_text = approved_text
        if decision in [
            AdjudicationDecision.ACCEPTED_FUSION,
            AdjudicationDecision.ACCEPTED_PROVIDER,
            AdjudicationDecision.EDITED_SCHOLAR,
            AdjudicationDecision.MARK_ILLEGIBLE,
        ]:
            item.is_researcher_approved = True
        item.revision_history.append(rev_entry)
        return item

    def mark_as_disputed(
        self,
        item_id: str,
        reason: str = "Disputed by epigraphist review",
        reviewer: str = "Epigraphist"
    ) -> AdjudicationItem:
        """Flags an adjudication item as formally disputed."""
        item = self.get_item(item_id)
        if not item:
            raise KeyError(f"Item {item_id} not found")
        item.disagreement_reason = f"{item.disagreement_reason}; {reason}".strip("; ")
        if reason not in item.scholarly_notes:
            item.scholarly_notes.append(reason)
        return self.record_decision(
            item_id,
            AdjudicationDecision.DISPUTED,
            item.approved_text or item.fused_candidate,
            reviewer=reviewer,
            rationale=reason
        )

    def mark_as_requiring_review(
        self,
        item_id: str,
        reviewer: str = "Epigraphist",
        notes: str = "Flagged for in-depth specialist review"
    ) -> AdjudicationItem:
        """Marks a reading as requiring specialist re-examination."""
        item = self.get_item(item_id)
        if not item:
            raise KeyError(f"Item {item_id} not found")
        if notes and notes not in item.scholarly_notes:
            item.scholarly_notes.append(notes)
        return self.record_decision(
            item_id,
            AdjudicationDecision.REQUIRES_REVIEW,
            item.approved_text or item.fused_candidate,
            reviewer=reviewer,
            rationale=notes
        )

    def add_alternative_reading(
        self,
        item_id: str,
        alternative_reading: str,
        reviewer: str = "Epigraphist",
        notes: str = ""
    ) -> AdjudicationItem:
        """Adds a candidate interpretation or variant reading without overwriting current approved text."""
        item = self.get_item(item_id)
        if not item:
            raise KeyError(f"Item {item_id} not found")
        clean_alt = alternative_reading.strip()
        if clean_alt and clean_alt not in item.alternative_readings:
            item.alternative_readings.append(clean_alt)
        msg = f"Added alternative candidate '{clean_alt}'"
        if notes:
            msg += f" ({notes})"
            item.scholarly_notes.append(f"Alt '{clean_alt}': {notes}")
        rev_entry = ReviewRevisionEntry(
            revision_id=len(item.revision_history) + 1,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            reviewer_identity=reviewer,
            decision="add_alternative",
            previous_reading=item.approved_text or item.fused_candidate,
            approved_reading=item.approved_text or item.fused_candidate,
            rationale=msg
        )
        item.revision_history.append(rev_entry)
        return item

    def add_scholarly_note(
        self,
        item_id: str,
        note: str,
        reviewer: str = "Epigraphist"
    ) -> AdjudicationItem:
        """Attaches a scholarly or palaeographical annotation to the item."""
        item = self.get_item(item_id)
        if not item:
            raise KeyError(f"Item {item_id} not found")
        clean_note = note.strip()
        if clean_note:
            item.scholarly_notes.append(clean_note)
            rev_entry = ReviewRevisionEntry(
                revision_id=len(item.revision_history) + 1,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                reviewer_identity=reviewer,
                decision="scholarly_note",
                previous_reading=item.approved_text or item.fused_candidate,
                approved_reading=item.approved_text or item.fused_candidate,
                rationale=clean_note
            )
            item.revision_history.append(rev_entry)
        return item

    def record_physical_evidence(
        self,
        item_id: str,
        evidence: str,
        reviewer: str = "Epigraphist"
    ) -> AdjudicationItem:
        """Records physical carved incision evidence observed on stone."""
        item = self.get_item(item_id)
        if not item:
            raise KeyError(f"Item {item_id} not found")
        clean_ev = evidence.strip()
        if clean_ev:
            item.supporting_physical_evidence.append(clean_ev)
            rev_entry = ReviewRevisionEntry(
                revision_id=len(item.revision_history) + 1,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                reviewer_identity=reviewer,
                decision="record_physical_evidence",
                previous_reading=item.approved_text or item.fused_candidate,
                approved_reading=item.approved_text or item.fused_candidate,
                rationale=f"Physical stone evidence: {clean_ev}"
            )
            item.revision_history.append(rev_entry)
        return item

    def is_approved(self, item_id: str) -> bool:
        """Returns True if the reading has been verified and approved by a researcher."""
        item = self.get_item(item_id)
        if not item:
            return False
        return (
            item.is_researcher_approved
            or (
                item.current_decision in [
                    AdjudicationDecision.ACCEPTED_FUSION.value,
                    AdjudicationDecision.ACCEPTED_PROVIDER.value,
                    AdjudicationDecision.EDITED_SCHOLAR.value,
                    AdjudicationDecision.MARK_ILLEGIBLE.value,
                ]
                and len(item.revision_history) > 0
            )
        )

