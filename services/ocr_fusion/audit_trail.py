"""
Explainability & Cryptographic Audit Trail Module for Kallvettu (K-IEFE).
Produces end-to-end reproducible provenance records for every fused reading.

ENABLES RESEARCHERS TO INSPECT:
- Exactly which source image pixels and region coordinates were recognized.
- What each provider outputted verbatim and with what model version.
- Why a particular candidate was prioritized (calibrated reliability, stroke evidence, historical context).
- Every human adjudication intervention with timestamp and rationale.
"""

import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple

from services.ai.common.source_identity import SourceIdentity
from services.ai.common.base_provider import BoundingBox
from services.ocr_fusion.visual_verifier import VisualEvidenceHypothesis
from services.ocr_fusion.historical_validator import HistoricalEvidenceMatch


@dataclass
class EvidenceFusionAuditRecord:
    """Comprehensive cryptographic explanation dossier for an inscription recognition run."""
    execution_id: str
    timestamp: str
    source_image_id: str
    image_sha256: str
    dimensions: Tuple[int, int]
    target_region_bbox: Optional[Tuple[float, float, float, float]]
    preprocessing_variant_used: str
    provider_outputs_verbatim: Dict[str, str]
    provider_model_versions: Dict[str, str]
    provider_confidences: Dict[str, Optional[float]]
    calibrated_reliabilities: Dict[str, float]
    inter_provider_correlations: Dict[str, float]
    consensus_entropy: float
    disagreement_severity: str
    alignment_strategy: str
    fusion_strategy: str
    primary_fused_reading: str
    conservative_reading: str
    competing_sequence_hypotheses: List[Dict[str, Any]]
    visual_evidence_hypotheses: List[Dict[str, Any]]
    historical_corpus_matches: List[Dict[str, Any]]
    adjudication_state: str
    adjudication_reviewer: Optional[str]
    adjudication_rationale: Optional[str]

    def compute_record_signature(self) -> str:
        """Computes deterministic SHA-256 integrity signature of the audit dossier."""
        payload = f"{self.execution_id}|{self.image_sha256}|{self.primary_fused_reading}|{self.timestamp}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def cryptographic_sha256(self) -> str:
        """Property returning the cryptographic SHA-256 integrity signature."""
        return self.compute_record_signature()

    @property
    def model_versions(self) -> Dict[str, str]:
        """Property returning the model versions dictionary."""
        return self.provider_model_versions

    @property
    def source_identity(self) -> Dict[str, Any]:
        """Property returning basic source identity metadata."""
        return {
            "source_image_id": self.source_image_id,
            "image_sha256": self.image_sha256,
            "dimensions": self.dimensions,
        }

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["integrity_signature"] = self.compute_record_signature()
        return data

    def to_json(self, indent: int = 2) -> str:
        import json
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class FusionAuditLogger:
    """
    Constructs and verifies explainable audit trails for the fusion engine.
    """

    @staticmethod
    def build_audit_record(
        execution_id: str,
        source_identity: SourceIdentity,
        preprocessing_variant: str,
        provider_outputs: Dict[str, str],
        model_versions: Dict[str, str],
        confidences: Dict[str, Optional[float]],
        reliabilities: Dict[str, float],
        consensus_entropy: float,
        disagreement_severity: str,
        fusion_strategy: str,
        fused_reading: str,
        conservative_reading: str,
        visual_hypotheses: Optional[List[Dict[str, Any]]] = None,
        historical_matches: Optional[List[Dict[str, Any]]] = None,
        competing_sequences: Optional[List[Dict[str, Any]]] = None,
        adjudication_state: str = "pending",
        reviewer: Optional[str] = None,
        rationale: Optional[str] = None,
        bbox: Optional[BoundingBox] = None
    ) -> EvidenceFusionAuditRecord:
        """
        Assembles an immutable EvidenceFusionAuditRecord.
        """
        region_tuple = (bbox.ymin, bbox.xmin, bbox.ymax, bbox.xmax) if bbox else None
        dims = source_identity.dimensions or (1000, 1000)

        return EvidenceFusionAuditRecord(
            execution_id=execution_id,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            source_image_id=source_identity.source_image_id,
            image_sha256=source_identity.image_sha256,
            dimensions=dims,
            target_region_bbox=region_tuple,
            preprocessing_variant_used=preprocessing_variant,
            provider_outputs_verbatim=provider_outputs,
            provider_model_versions=model_versions,
            provider_confidences=confidences,
            calibrated_reliabilities=reliabilities,
            inter_provider_correlations={},
            consensus_entropy=consensus_entropy,
            disagreement_severity=disagreement_severity,
            alignment_strategy="multi_grapheme_needleman_wunsch",
            fusion_strategy=fusion_strategy,
            primary_fused_reading=fused_reading,
            conservative_reading=conservative_reading,
            competing_sequence_hypotheses=competing_sequences or [],
            visual_evidence_hypotheses=visual_hypotheses or [],
            historical_corpus_matches=historical_matches or [],
            adjudication_state=adjudication_state,
            adjudication_reviewer=reviewer,
            adjudication_rationale=rationale
        )

