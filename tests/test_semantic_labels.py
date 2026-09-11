"""Unit tests for FOVEAX Phase 7 -- semantic_labels module.

Run with:
    python -m pytest tests/test_semantic_labels.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perception.semantic_labels import (
    FOVEAX_CLASSES,
    RELLIS3D_EXCLUDED_CLASSES,
    RELLIS3D_TO_FOVEAX,
)

# The real, authoritative RELLIS-3D ontology class list, read directly from
# C:\dev\data\rellis3d\Rellis_3D_ontology\ontology.yaml (2026-09-11) -- not
# assumed from documentation. If the real file's ID list ever changes, this
# test constant must be re-verified against it, not silently edited to match
# whatever the mapping table currently contains.
REAL_ONTOLOGY_CLASS_IDS: set[int] = {
    0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 18, 19, 23, 27, 31, 33, 34,
}


class TestRellis3DOntologyCoverage:
    """Every real ontology class ID must be handled -- mapped or excluded."""

    def test_ontology_has_20_classes(self) -> None:
        assert len(REAL_ONTOLOGY_CLASS_IDS) == 20

    def test_every_ontology_id_is_mapped_or_excluded(self) -> None:
        handled = set(RELLIS3D_TO_FOVEAX) | set(RELLIS3D_EXCLUDED_CLASSES)
        missing = REAL_ONTOLOGY_CLASS_IDS - handled
        assert not missing, (
            f"Ontology class ID(s) {sorted(missing)} are neither in "
            "RELLIS3D_TO_FOVEAX nor RELLIS3D_EXCLUDED_CLASSES -- would "
            "silently fall through with no explicit handling."
        )

    def test_no_extra_ids_beyond_the_real_ontology(self) -> None:
        handled = set(RELLIS3D_TO_FOVEAX) | set(RELLIS3D_EXCLUDED_CLASSES)
        extra = handled - REAL_ONTOLOGY_CLASS_IDS
        assert not extra, (
            f"RELLIS3D_TO_FOVEAX/RELLIS3D_EXCLUDED_CLASSES reference ID(s) "
            f"{sorted(extra)} that don't exist in the real ontology.yaml -- "
            "likely a guessed ID (e.g. an assumed 'sand' class) rather than "
            "one confirmed against real data."
        )

    def test_mapped_and_excluded_are_disjoint(self) -> None:
        overlap = set(RELLIS3D_TO_FOVEAX) & set(RELLIS3D_EXCLUDED_CLASSES)
        assert not overlap, (
            f"ID(s) {sorted(overlap)} are both mapped to a FOVEAX class and "
            "excluded -- ambiguous, must be exactly one or the other."
        )

    def test_sky_is_excluded_not_mapped(self) -> None:
        assert 7 in RELLIS3D_EXCLUDED_CLASSES
        assert 7 not in RELLIS3D_TO_FOVEAX

    def test_sand_is_not_present(self) -> None:
        # "sand" was assumed as a possible class during early planning but
        # does not exist in the real ontology.yaml -- confirm it never
        # sneaks into the mapping as a guessed/fabricated ID.
        assert 2 not in RELLIS3D_TO_FOVEAX  # no gap-filling guesses either
        assert 11 not in RELLIS3D_TO_FOVEAX
        assert 13 not in RELLIS3D_TO_FOVEAX
        assert 14 not in RELLIS3D_TO_FOVEAX


class TestRellis3DToFoveaxValues:
    """Every mapped value must be a valid FOVEAX class ID."""

    def test_all_values_are_valid_foveax_class_ids(self) -> None:
        valid_ids = set(FOVEAX_CLASSES)
        for raw_id, foveax_id in RELLIS3D_TO_FOVEAX.items():
            assert foveax_id in valid_ids, (
                f"RELLIS3D_TO_FOVEAX[{raw_id}] = {foveax_id} is not a valid "
                f"FOVEAX class ID (valid: {sorted(valid_ids)})."
            )

    @pytest.mark.parametrize(
        "raw_id,expected_class_name",
        [
            (0, "UNKNOWN"),          # void
            (1, "DRIVABLE_GROUND"),  # dirt
            (3, "ROUGH_TERRAIN"),    # grass
            (4, "VEGETATION"),       # tree
            (5, "SOLID_OBSTACLE"),   # pole
            (6, "ROUGH_TERRAIN"),    # water
            (8, "VEHICLE"),          # vehicle
            (9, "SOLID_OBSTACLE"),   # object
            (10, "DRIVABLE_GROUND"), # asphalt
            (12, "BUILDING_WALL"),   # building
            (15, "SOLID_OBSTACLE"),  # log
            (17, "PEDESTRIAN"),      # person
            (18, "SOLID_OBSTACLE"),  # fence
            (19, "VEGETATION"),      # bush
            (23, "DRIVABLE_GROUND"), # concrete
            (27, "SOLID_OBSTACLE"),  # barrier
            (31, "ROUGH_TERRAIN"),   # puddle
            (33, "ROUGH_TERRAIN"),   # mud
            (34, "SOLID_OBSTACLE"),  # rubble
        ],
    )
    def test_specific_class_mappings(
        self, raw_id: int, expected_class_name: str
    ) -> None:
        foveax_id = RELLIS3D_TO_FOVEAX[raw_id]
        assert FOVEAX_CLASSES[foveax_id] == expected_class_name
