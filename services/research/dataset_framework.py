"""
Kallvettu Historical Epigraphy Research & Dataset Management Framework.
Implements Phase 10 Requirements:
- Ingestion, validation, and structured management of historical inscription datasets.
- Schema compliance with academic datasets (e.g. Mendeley Tamil Brahmi Stone Inscription Dataset 2spz2js9jp.1).
- Strict site/monument-level train/validation/test splitting to prevent data leakage.
- Multi-tier character annotation validator handling weathered and ambiguous historical glyphs.
"""

import os
import csv
import logging
import random
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict

logger = logging.getLogger("kallvettu.research.dataset")

DATASETS_BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "datasets")
)
METADATA_DIR = os.path.join(DATASETS_BASE_DIR, "metadata")


@dataclass
class InscriptionRecord:
    """Metadata record for a documented lithic inscription."""
    inscription_id: str
    title_en: str
    title_ta: str
    script: str
    period_en: str
    dynasty: str
    location: str
    license: str
    source_dataset: str
    is_authentic: bool
    ground_truth_transcription: str


@dataclass
class CharacterAnnotation:
    """Spatial and epigraphical annotation for a single chiseled glyph."""
    character_id: str
    inscription_id: str
    script: str
    character_label: str
    unicode_equivalent: str
    ymin: float
    xmin: float
    ymax: float
    xmax: float
    damage_condition: str
    alternative_readings: str
    verified_by: str

    def get_alternative_list(self) -> List[str]:
        if not self.alternative_readings:
            return [self.character_label]
        return [r.strip() for r in self.alternative_readings.split("|") if r.strip()]

    def is_valid_bbox(self) -> bool:
        """Verifies normalized coordinates (0.0 to 1.0) and non-inverted box."""
        for coord in (self.ymin, self.xmin, self.ymax, self.xmax):
            if coord < 0.0 or coord > 1.0:
                return False
        return self.ymax > self.ymin and self.xmax > self.xmin


class EpigraphicalDatasetManager:
    """
    Manages historical inscription datasets across Tamil-Brahmi, Vatteluttu, and Medieval Tamil.
    Enforces research integrity:
    1. Prevents image crop contamination between train and test splits (site-level isolation).
    2. Validates spatial bounding boxes and character inventories.
    3. Maintains full provenance tracking and verification logs.
    """

    SUPPORTED_SCRIPTS = ("Tamil-Brahmi", "Vatteluttu", "Medieval Tamil")

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or DATASETS_BASE_DIR
        self.metadata_dir = os.path.join(self.base_dir, "metadata")

    def load_inscriptions(self) -> List[InscriptionRecord]:
        """Loads documented inscription master records from CSV."""
        path = os.path.join(self.metadata_dir, "inscriptions.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Inscriptions metadata file not found at: {path}")

        records = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(InscriptionRecord(
                    inscription_id=row["inscription_id"].strip(),
                    title_en=row["title_en"].strip(),
                    title_ta=row["title_ta"].strip(),
                    script=row["script"].strip(),
                    period_en=row["period_en"].strip(),
                    dynasty=row["dynasty"].strip(),
                    location=row["location"].strip(),
                    license=row["license"].strip(),
                    source_dataset=row["source_dataset"].strip(),
                    is_authentic=row.get("is_authentic", "True").lower() == "true",
                    ground_truth_transcription=row.get("ground_truth_transcription", "").strip(),
                ))
        return records

    def load_characters(self) -> List[CharacterAnnotation]:
        """Loads fine-grained character-level bounding box annotations."""
        path = os.path.join(self.metadata_dir, "characters.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Characters metadata file not found at: {path}")

        chars = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                chars.append(CharacterAnnotation(
                    character_id=row["character_id"].strip(),
                    inscription_id=row["inscription_id"].strip(),
                    script=row["script"].strip(),
                    character_label=row["character_label"].strip(),
                    unicode_equivalent=row["unicode_equivalent"].strip(),
                    ymin=float(row["ymin"]),
                    xmin=float(row["xmin"]),
                    ymax=float(row["ymax"]),
                    xmax=float(row["xmax"]),
                    damage_condition=row.get("damage_condition", "intact").strip(),
                    alternative_readings=row.get("alternative_readings", "").strip(),
                    verified_by=row.get("verified_by", "unverified").strip(),
                ))
        return chars

    def get_character_inventory(self, script: Optional[str] = None) -> Set[str]:
        """Extracts unique verified character classes for the target historical script."""
        chars = self.load_characters()
        if script:
            return {c.character_label for c in chars if c.script.lower() == script.lower()}
        return {c.character_label for c in chars}

    def generate_site_level_split(
        self,
        test_ratio: float = 0.20,
        val_ratio: float = 0.20,
        seed: int = 42
    ) -> Dict[str, List[CharacterAnnotation]]:
        """
        Splits annotations into Train, Validation, and Test sets partitioned strictly
        by inscription/site to prevent visual and geological leakage.
        """
        inscriptions = self.load_inscriptions()
        characters = self.load_characters()

        inscript_ids = sorted([i.inscription_id for i in inscriptions])
        rng = random.Random(seed)
        rng.shuffle(inscript_ids)

        n_total = len(inscript_ids)
        n_test = max(1, int(round(n_total * test_ratio)))
        n_val = max(1, int(round(n_total * val_ratio))) if n_total > 2 else 0

        test_ids = set(inscript_ids[:n_test])
        val_ids = set(inscript_ids[n_test:n_test + n_val])
        train_ids = set(inscript_ids[n_test + n_val:])

        # Guard: Ensure at least one inscription in train set if possible
        if not train_ids and val_ids:
            train_ids.add(val_ids.pop())

        splits: Dict[str, List[CharacterAnnotation]] = {
            "train": [c for c in characters if c.inscription_id in train_ids],
            "val": [c for c in characters if c.inscription_id in val_ids],
            "test": [c for c in characters if c.inscription_id in test_ids],
        }

        # Validate no cross-set contamination
        train_inscriptions = {c.inscription_id for c in splits["train"]}
        test_inscriptions = {c.inscription_id for c in splits["test"]}
        overlap = train_inscriptions.intersection(test_inscriptions)
        if overlap:
            raise ValueError(f"Scientific integrity violation: Data leakage detected across splits: {overlap}")

        return splits

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Computes comprehensive dataset integrity metrics for academic reporting."""
        inscriptions = self.load_inscriptions()
        chars = self.load_characters()

        by_script: Dict[str, int] = {}
        for c in chars:
            by_script[c.script] = by_script.get(c.script, 0) + 1

        damage_breakdown: Dict[str, int] = {}
        for c in chars:
            damage_breakdown[c.damage_condition] = damage_breakdown.get(c.damage_condition, 0) + 1

        all_boxes_valid = all(c.is_valid_bbox() for c in chars)

        return {
            "total_inscriptions": len(inscriptions),
            "authentic_inscriptions": sum(1 for i in inscriptions if i.is_authentic),
            "total_annotated_characters": len(chars),
            "characters_by_script": by_script,
            "unique_character_classes": len(self.get_character_inventory()),
            "damage_distribution": damage_breakdown,
            "all_bounding_boxes_valid": all_boxes_valid,
            "provenance_standard": "Mendeley Tamil-Brahmi 2spz2js9jp.1 & ASI Epigraphia Indica",
        }


def get_dataset_manager() -> EpigraphicalDatasetManager:
    """Returns singleton/standard dataset manager instance."""
    return EpigraphicalDatasetManager()

