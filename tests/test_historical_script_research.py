"""
Phase 10 Unit Tests: Historical Script Recognition Research & Dataset Framework.
Covers:
- Dataset directory structure integrity
- Academic metadata schema validation (Mendeley 2spz2js9jp.1 & ASI references)
- Character bounding-box normalization & coordinate checks
- Multi-tier character alternative readings
- Site-level split isolation (leakage prevention between train/val/test)
- Epigraphical dataset statistical reporting
"""

import os
import pytest
from services.research.dataset_framework import (
    EpigraphicalDatasetManager,
    InscriptionRecord,
    CharacterAnnotation,
    get_dataset_manager,
    DATASETS_BASE_DIR,
)


def test_dataset_directory_hierarchy_integrity():
    """Verify standard research directory structure exists on disk."""
    mgr = get_dataset_manager()
    base = mgr.base_dir
    assert os.path.isdir(base)

    expected_subdirs = [
        os.path.join(base, "tamil_brahmi", "original_images"),
        os.path.join(base, "tamil_brahmi", "line_crops"),
        os.path.join(base, "tamil_brahmi", "character_crops"),
        os.path.join(base, "tamil_brahmi", "annotations"),
        os.path.join(base, "tamil_brahmi", "ground_truth"),
        os.path.join(base, "vatteluttu", "original_images"),
        os.path.join(base, "vatteluttu", "character_crops"),
        os.path.join(base, "medieval_tamil", "original_images"),
        os.path.join(base, "metadata"),
    ]
    for d in expected_subdirs:
        assert os.path.isdir(d), f"Missing dataset directory: {d}"


def test_inscriptions_metadata_schema_and_authenticity():
    """Verify master inscription records conform to academic provenance standards."""
    mgr = get_dataset_manager()
    inscriptions = mgr.load_inscriptions()

    assert len(inscriptions) >= 5
    for ins in inscriptions:
        assert isinstance(ins, InscriptionRecord)
        assert len(ins.inscription_id) >= 4
        assert len(ins.title_en) > 0
        assert len(ins.title_ta) > 0
        assert ins.script in ("Tamil-Brahmi", "Vatteluttu", "Medieval Tamil")
        assert len(ins.location) > 0
        assert len(ins.license) > 0
        assert ins.is_authentic is True
        assert len(ins.ground_truth_transcription) > 0


def test_character_annotations_bounding_boxes():
    """Verify glyph coordinates are strictly normalized and non-inverted."""
    mgr = get_dataset_manager()
    chars = mgr.load_characters()

    assert len(chars) >= 15
    for c in chars:
        assert isinstance(c, CharacterAnnotation)
        assert c.is_valid_bbox() is True
        assert 0.0 <= c.ymin < c.ymax <= 1.0
        assert 0.0 <= c.xmin < c.xmax <= 1.0
        assert len(c.character_label) > 0
        assert len(c.unicode_equivalent) > 0
        assert c.verified_by.startswith("Epigraphist")


def test_character_alternative_readings_resolution():
    """Verify weathered or ambiguous historical glyphs support multiple academic readings."""
    mgr = get_dataset_manager()
    chars = mgr.load_characters()

    # TB_CHR_0006 is ambiguous 'saa' / 'si' / 'ci'
    ambig_char = next((c for c in chars if c.character_id == "TB_CHR_0006"), None)
    assert ambig_char is not None
    assert ambig_char.damage_condition == "severely_eroded"
    alts = ambig_char.get_alternative_list()
    assert len(alts) >= 2
    assert "saa" in alts
    assert "si" in alts


def test_site_level_split_no_leakage():
    """Verify train, val, and test splits partition strictly by monument/site."""
    mgr = get_dataset_manager()
    splits = mgr.generate_site_level_split(test_ratio=0.25, val_ratio=0.25, seed=42)

    assert "train" in splits
    assert "val" in splits
    assert "test" in splits

    train_ids = {c.inscription_id for c in splits["train"]}
    val_ids = {c.inscription_id for c in splits["val"]}
    test_ids = {c.inscription_id for c in splits["test"]}

    # Mutual exclusivity check
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(train_ids.intersection(val_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0


def test_dataset_statistics_reporting():
    """Verify comprehensive dataset metrics calculation."""
    mgr = get_dataset_manager()
    stats = mgr.get_dataset_statistics()

    assert stats["total_inscriptions"] >= 5
    assert stats["authentic_inscriptions"] == stats["total_inscriptions"]
    assert stats["total_annotated_characters"] >= 15
    assert stats["all_bounding_boxes_valid"] is True
    assert "Tamil-Brahmi" in stats["characters_by_script"]
    assert "Vatteluttu" in stats["characters_by_script"]
    assert "Medieval Tamil" in stats["characters_by_script"]

