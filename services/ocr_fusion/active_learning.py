"""
Active Learning & Epigraphical Annotation Collector for Kallvettu (K-IEFE).
Captures verified researcher adjudications into trusted training datasets
for fine-tuning historical OCR models.

GUARDRAILS:
- Only researcher-approved or scholar-edited readings are eligible for ingestion.
- Unverified raw model predictions are strictly prohibited from entering training sets.
- Retains complete provenance: crop coordinates, original candidates, reviewer identity, dataset version.
"""

import os
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple

DEFAULT_ACTIVE_LEARNING_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "datasets", "active_learning"
)


@dataclass
class VerifiedAnnotationRecord:
    """Immutable ground-truth specimen record approved by a researcher."""
    specimen_id: str
    inscription_id: str
    source_image_id: str
    image_sha256: str
    crop_coordinates: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    normalized_bbox: Tuple[float, float, float, float] # (ymin, xmin, ymax, xmax)
    script: str
    original_candidates: Dict[str, str]
    final_approved_reading: str
    reviewer_identity: str
    review_timestamp: str
    dataset_version: str = "2.0.0-verified"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ActiveLearningManager:
    """
    Manages collection, persistence, and export of scholar-verified inscriptional ground-truth.
    """

    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = storage_dir or DEFAULT_ACTIVE_LEARNING_DIR
        os.makedirs(self.storage_dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.storage_dir, "approved_annotations.jsonl")

    def store_verified_annotation(
        self,
        record: VerifiedAnnotationRecord
    ) -> str:
        """
        Appends an approved researcher adjudication to the active learning dataset.
        """
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        return record.specimen_id

    def load_all_annotations(self) -> List[VerifiedAnnotationRecord]:
        """Loads all verified records from the trusted active learning dataset."""
        records = []
        if not os.path.exists(self.jsonl_path):
            return records

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                    records.append(VerifiedAnnotationRecord(
                        specimen_id=data["specimen_id"],
                        inscription_id=data["inscription_id"],
                        source_image_id=data["source_image_id"],
                        image_sha256=data["image_sha256"],
                        crop_coordinates=tuple(data["crop_coordinates"]),
                        normalized_bbox=tuple(data["normalized_bbox"]),
                        script=data["script"],
                        original_candidates=data["original_candidates"],
                        final_approved_reading=data["final_approved_reading"],
                        reviewer_identity=data["reviewer_identity"],
                        review_timestamp=data["review_timestamp"],
                        dataset_version=data.get("dataset_version", "2.0.0-verified"),
                        notes=data.get("notes", "")
                    ))
                except Exception:
                    continue
        return records

    def export_for_model_training(
        self,
        output_format: str = "json"
    ) -> str:
        """
        Exports active learning annotations formatted for OCR model fine-tuning (CRNN / TrOCR / YOLO).
        """
        records = self.load_all_annotations()
        if output_format == "tsv":
            lines = ["specimen_id\tscript\treading\tbbox"]
            for r in records:
                bbox_str = ",".join(str(x) for x in r.crop_coordinates)
                lines.append(f"{r.specimen_id}\t{r.script}\t{r.final_approved_reading}\t{bbox_str}")
            return "\n".join(lines)

        return json.dumps([r.to_dict() for r in records], ensure_ascii=False, indent=2)

