"""FOVEAX shared semantic mapping configurations.

This module centralizes the FOVEAX class definitions, colors, and the
mapping from raw SemanticKITTI label IDs to FOVEAX simplified classes.
"""

import numpy as np

FOVEAX_CLASSES: dict[int, str] = {
    0: "DRIVABLE_GROUND",
    1: "ROUGH_TERRAIN",
    2: "VEGETATION",
    3: "BUILDING_WALL",
    4: "SOLID_OBSTACLE",
    5: "VEHICLE",
    6: "PEDESTRIAN",
    7: "UNKNOWN",
}

FOVEAX_COLORS: np.ndarray = np.array(
    [
        [46, 204, 113],   # 0 Drivable ground: green
        [241, 196, 15],   # 1 Rough terrain: yellow
        [39, 174, 96],    # 2 Vegetation: dark green
        [142, 68, 173],   # 3 Building/wall: purple
        [231, 76, 60],    # 4 Solid obstacle: red
        [52, 152, 219],   # 5 Vehicle: blue
        [26, 188, 156],   # 6 Pedestrian: cyan
        [108, 117, 125],  # 7 Unknown: grey
    ],
    dtype=np.uint8,
)

NUM_FOVEAX_CLASSES: int = len(FOVEAX_CLASSES)

# Raw SemanticKITTI class ID -> simplified FOVEAX class ID.
SEMANTICKITTI_TO_FOVEAX: dict[int, int] = {
    0:   7,   # unlabeled
    1:   7,   # outlier
    10:  5,   # car
    11:  5,   # bicycle
    13:  5,   # bus
    15:  5,   # motorcycle
    16:  5,   # on-rails
    18:  5,   # truck
    20:  5,   # other-vehicle
    30:  6,   # person
    31:  6,   # bicyclist
    32:  6,   # motorcyclist
    40:  0,   # road
    44:  0,   # parking
    48:  0,   # sidewalk
    49:  1,   # other-ground
    50:  3,   # building
    51:  3,   # fence
    52:  2,   # other-structure
    60:  0,   # lane-marking -> DRIVABLE_GROUND
    70:  2,   # vegetation
    71:  2,   # trunk
    72:  1,   # terrain
    80:  4,   # pole
    81:  4,   # traffic-sign
    99:  4,   # other-object
    252: 5,   # moving-car
    253: 5,   # moving-bicyclist
    254: 6,   # moving-person
    255: 6,   # moving-motorcyclist
    256: 5,   # moving-on-rails
    257: 5,   # moving-bus
    258: 5,   # moving-truck
    259: 5,   # moving-other-vehicle
}

# ---------------------------------------------------------------------------
# RELLIS-3D class ID -> simplified FOVEAX class ID.
#
# Source of truth: C:\dev\data\rellis3d\Rellis_3D_ontology\ontology.yaml,
# read directly from the real downloaded dataset (2026-09-11), NOT assumed
# from documentation. The real 20-class list found there is:
#   0=void, 1=dirt, 3=grass, 4=tree, 5=pole, 6=water, 7=sky, 8=vehicle,
#   9=object, 10=asphalt, 12=building, 15=log, 17=person, 18=fence,
#   19=bush, 23=concrete, 27=barrier, 31=puddle, 33=mud, 34=rubble
# Note: there is no "sand" class in the real data, despite that being
# assumed as a possibility during earlier planning -- not included below.
#
# `sky` (7) is deliberately NOT a key in this dict. It is excluded from the
# 2.5D grid entirely (not a ground/obstacle class at all) via
# RELLIS3D_EXCLUDED_CLASSES below; rellis3d_loader.py drops those points
# before they ever reach a FOVEAX class.
# ---------------------------------------------------------------------------
RELLIS3D_TO_FOVEAX: dict[int, int] = {
    0:  7,   # void -> UNKNOWN. Not a real terrain/obstacle label; same
             #   treatment as SemanticKITTI's "unlabeled" (0 -> UNKNOWN above).
    1:  0,   # dirt -> DRIVABLE_GROUND. This is a trail project: packed dirt
             #   track is the primary drivable surface, not a hazard.
    3:  1,   # grass -> ROUGH_TERRAIN. Natural ground cover, not a standing
             #   obstacle -- treated like SemanticKITTI's "terrain" (72 -> 1
             #   ROUGH_TERRAIN above), not like "vegetation" (trunks/bushes
             #   that actually block a path).
    4:  2,   # tree -> VEGETATION (explicit per spec).
    5:  4,   # pole -> SOLID_OBSTACLE (explicit per spec).
    6:  1,   # water -> ROUGH_TERRAIN. LiDAR alone can't tell a shallow
             #   puddle-like body from something deeper; bucketed with the
             #   other low-traction/cautionary ground surfaces (puddle, mud)
             #   rather than assumed safely DRIVABLE_GROUND.
    8:  5,   # vehicle -> VEHICLE (explicit per spec).
    9:  4,   # object -> SOLID_OBSTACLE. Generic catch-all class in the
             #   ontology; grouped with the other unidentified-obstacle IDs.
    10: 0,   # asphalt -> DRIVABLE_GROUND (explicit per spec).
    12: 3,   # building -> BUILDING_WALL (explicit per spec).
    15: 4,   # log -> SOLID_OBSTACLE (explicit per spec).
    17: 6,   # person -> PEDESTRIAN (explicit per spec).
    18: 4,   # fence -> SOLID_OBSTACLE. NOTE: this intentionally differs from
             #   SEMANTICKITTI_TO_FOVEAX's fence (51 -> 3 BUILDING_WALL
             #   above). The RELLIS-3D task spec explicitly groups fence with
             #   pole/barrier/log as a thin blocking obstacle, not a
             #   structural wall material -- kept as specified rather than
             #   forced to match the SemanticKITTI table.
    19: 2,   # bush -> VEGETATION (explicit per spec).
    23: 0,   # concrete -> DRIVABLE_GROUND (explicit per spec).
    27: 4,   # barrier -> SOLID_OBSTACLE (explicit per spec).
    31: 1,   # puddle -> ROUGH_TERRAIN (explicit per spec).
    33: 1,   # mud -> ROUGH_TERRAIN (explicit per spec).
    34: 4,   # rubble -> SOLID_OBSTACLE. NOTE: the task spec lists "rubble"
             #   in two conflicting places -- once grouped with
             #   "mud/puddle/rubble -> ROUGH_TERRAIN" and once grouped with
             #   "pole, barrier, fence, log, building, object, rubble ->
             #   SOLID_OBSTACLE". Resolved as SOLID_OBSTACLE: rubble is
             #   debris/rock piles that physically block a path, unlike
             #   mud/puddle which are low-traction but still passable ground
             #   surfaces -- classifying it as ROUGH_TERRAIN would wrongly
             #   signal "drivable with caution" for what is actually a
             #   blocking obstacle. Flagged explicitly rather than silently
             #   picking one interpretation.
}

# Classes excluded from the FOVEAX grid entirely (not mapped to any FOVEAX
# class -- points with these raw IDs are dropped before remapping).
RELLIS3D_EXCLUDED_CLASSES: frozenset[int] = frozenset({7})  # sky
