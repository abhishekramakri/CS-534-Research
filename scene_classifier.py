"""
scene_classifier.py
Classifies the acoustic scene type and returns pre-optimized acoustic
parameter presets, matching SAMOSA's third parallel perception module.

For the midterm this is rule-based (derived from material distributions +
room volume).  The structure is ready to swap in a Places365 MobileNetV2
classifier later.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Pre-optimized acoustic presets per scene type.
# max_order controls ISM reflection depth; target_rt60 is used for sanity
# checking the simulated result.
# ---------------------------------------------------------------------------
SCENE_PRESETS = {
    "living_room": {
        "max_order":   15,
        "target_rt60": 0.5,
        "description": "Medium absorption, mixed soft/hard surfaces",
    },
    "bedroom": {
        "max_order":   10,
        "target_rt60": 0.4,
        "description": "High absorption from fabrics, small volume",
    },
    "conference": {
        "max_order":   17,
        "target_rt60": 0.7,
        "description": "Large hard-surface room, long reverb tail",
    },
    "office": {
        "max_order":   12,
        "target_rt60": 0.5,
        "description": "Mixed surfaces, moderate size",
    },
    "outdoor": {
        "max_order":   2,
        "target_rt60": 0.1,
        "description": "Minimal reflections",
    },
}

# Materials considered acoustically "soft" (high absorption)
_SOFT_MATERIALS = {
    "carpet_cotton", "carpet_hairy", "ceiling_fibre_absorber",
    "curtains_hung_straight", "curtains_in_folds",
}

# Materials considered acoustically "hard" (low absorption)
_HARD_MATERIALS = {
    "marble_floor", "concrete_floor", "concrete_block_wall",
    "rough_concrete", "brickwork",
}


def classify_scene(
    material_distributions: dict | None = None,
    room_dims: list | None = None,
    best_materials: dict | None = None,
) -> tuple[str, dict]:
    """
    Infer acoustic scene type from material distributions and room geometry.

    Args:
        material_distributions: {"walls": {"plasterboard": 0.8, ...}, ...}
        room_dims:              [width, length, height] in metres
        best_materials:         {"walls": str, "floor": str, "ceiling": str}
                                (used if distributions unavailable)

    Returns:
        (scene_type, preset_dict)
        e.g. ("living_room", {"max_order": 15, "target_rt60": 0.5, ...})
    """
    volume = _volume(room_dims) if room_dims else 50.0  # default mid-size
    softness = _softness_score(material_distributions, best_materials)

    # Decision tree
    if volume < 30 and softness > 0.5:
        scene = "bedroom"
    elif volume > 120:
        scene = "conference"
    elif softness > 0.4:
        scene = "living_room"
    elif softness < 0.15:
        scene = "conference"
    else:
        scene = "office"

    return scene, SCENE_PRESETS[scene]


def _volume(dims):
    if dims is None:
        return 50.0
    return dims[0] * dims[1] * dims[2]


def _softness_score(distributions, best_materials):
    """
    Compute a 0-1 softness score across all surfaces.
    Higher = more absorptive room.
    """
    score = 0.0
    weight = 0.0

    if distributions:
        for surface, dist in distributions.items():
            for mat, frac in dist.items():
                if mat in _SOFT_MATERIALS:
                    score += frac
                elif mat in _HARD_MATERIALS:
                    score -= frac * 0.5
                weight += frac
    elif best_materials:
        for mat in best_materials.values():
            weight += 1.0
            if mat in _SOFT_MATERIALS:
                score += 1.0
            elif mat in _HARD_MATERIALS:
                score -= 0.5

    if weight == 0:
        return 0.3  # neutral default
    return max(0.0, min(1.0, score / weight + 0.3))


if __name__ == "__main__":
    # Quick test with example distributions
    test_dist = {
        "walls":   {"plasterboard": 0.85, "wood_panel": 0.15},
        "floor":   {"carpet_cotton": 0.6, "wood_16mm": 0.4},
        "ceiling": {"ceiling_fibre_absorber": 1.0},
    }
    scene, preset = classify_scene(
        material_distributions=test_dist,
        room_dims=[5.0, 4.0, 2.8],
    )
    print(f"Scene type: {scene}")
    print(f"Preset:     {preset}")
