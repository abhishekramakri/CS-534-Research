"""
acoustics.py
Computes a Room Impulse Response (RIR) from room dimensions and surface materials
using pyroomacoustics.  Supports both named materials (string) and blended
absorption distributions from segmentation.py.
"""

import numpy as np
import pyroomacoustics as pra


# Maps surface type → pyroomacoustics material name (fallback defaults).
DEFAULT_MATERIALS = {
    "walls":   "plasterboard",
    "floor":   "carpet_cotton",
    "ceiling": "ceiling_fibre_absorber",
}

# Octave-band centre frequencies used for absorption blending
OCTAVE_BANDS = [125, 250, 500, 1000, 2000, 4000]

# Absorption coefficients at OCTAVE_BANDS for each material we support.
# Sources: ISO 354 / Vorländer (2008) / Kuttruff (2009).
ABSORPTION_TABLE = {
    "brickwork":              [0.05, 0.04, 0.02, 0.04, 0.05, 0.05],
    "carpet_cotton":          [0.07, 0.31, 0.49, 0.81, 0.66, 0.54],
    "carpet_hairy":           [0.11, 0.14, 0.37, 0.43, 0.27, 0.25],
    "linoleum":               [0.02, 0.02, 0.03, 0.04, 0.04, 0.05],
    "parquet_wood_22mm":      [0.04, 0.04, 0.07, 0.06, 0.06, 0.07],
    "concrete_floor":         [0.01, 0.01, 0.02, 0.02, 0.02, 0.05],
    "marble_floor":           [0.01, 0.01, 0.01, 0.02, 0.02, 0.02],
    "concrete_block_wall":    [0.36, 0.44, 0.31, 0.29, 0.39, 0.25],
    "plasterboard":           [0.15, 0.11, 0.04, 0.04, 0.07, 0.14],
    "wood_panel":             [0.42, 0.21, 0.10, 0.08, 0.06, 0.06],
    "rough_concrete":         [0.02, 0.03, 0.03, 0.03, 0.04, 0.07],
    "ceiling_fibre_absorber": [0.33, 0.44, 0.82, 0.90, 0.92, 0.83],
    "wood_16mm":              [0.18, 0.12, 0.10, 0.09, 0.08, 0.07],
}


def _blend_absorption(distribution: dict) -> list[float]:
    """
    Weighted-average absorption coefficients from a material distribution.
    distribution: {"plasterboard": 0.85, "wood_panel": 0.15}
    Returns list of 6 coefficients at OCTAVE_BANDS.
    """
    blended = np.zeros(len(OCTAVE_BANDS), dtype=float)
    total_weight = 0.0

    for mat_name, fraction in distribution.items():
        if mat_name in ABSORPTION_TABLE:
            blended += fraction * np.array(ABSORPTION_TABLE[mat_name])
            total_weight += fraction
        else:
            # Unknown material — use a mid-range default
            blended += fraction * np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
            total_weight += fraction

    if total_weight > 0:
        blended /= total_weight

    return blended.tolist()


def _resolve_material(surface_key: str, materials: dict, distributions: dict | None):
    """
    Return a scalar or string that pra.make_materials() can wrap.
    - distributions present → scalar mean absorption (float)
    - otherwise            → named material string
    pra.make_materials() handles both via Material(scalar) and Material(string).
    """
    if distributions and surface_key in distributions:
        coeffs = _blend_absorption(distributions[surface_key])
        return float(np.mean(coeffs))   # scalar mean — version-safe
    return materials[surface_key]       # named string


def compute_rir(
    room_dims,
    materials=None,
    source_pos=None,
    listener_pos=None,
    fs=16000,
    max_order=17,
    distributions=None,
):
    """
    Compute a Room Impulse Response.

    Args:
        room_dims:      [width, length, height] in metres
        materials:       dict  {"walls": str, "floor": str, "ceiling": str}
        source_pos:     [x, y, z] of audio source
        listener_pos:   [x, y, z] of listener
        fs:             sample rate (Hz)
        max_order:      image-source-method reflection order
        distributions:  optional per-surface material distributions from segmentation.py
                        e.g. {"walls": {"plasterboard": 0.8, "wood_panel": 0.2}, ...}

    Returns:
        rir: np.ndarray — the room impulse response
        t60: float — reverberation time in seconds
    """
    if materials is None:
        materials = DEFAULT_MATERIALS

    w, l, h = room_dims

    if source_pos is None:
        source_pos = [w * 0.25, l * 0.5, h * 0.5]
    if listener_pos is None:
        listener_pos = [w * 0.75, l * 0.5, h * 0.5]

    wall_param  = _resolve_material("walls",   materials, distributions)
    floor_param = _resolve_material("floor",   materials, distributions)
    ceil_param  = _resolve_material("ceiling", materials, distributions)

    # ShoeBox surface order: east, west, north, south, floor, ceiling
    room_materials = pra.make_materials(
        east=wall_param,
        west=wall_param,
        north=wall_param,
        south=wall_param,
        floor=floor_param,
        ceiling=ceil_param,
    )

    room = pra.ShoeBox(
        room_dims,
        fs=fs,
        materials=room_materials,
        max_order=max_order,
    )

    room.add_source(source_pos)
    room.add_microphone(listener_pos)
    room.compute_rir()

    rir = room.rir[0][0]  # mic 0, source 0

    t60 = pra.experimental.measure_rt60(rir, fs=fs, plot=False)

    return rir, t60


if __name__ == "__main__":
    import time as _time

    print("=== acoustics.py sanity check ===\n")

    configs = [
        {
            "name": "Small furnished room (3x4x2.5m)",
            "dims": [3, 4, 2.5],
            "materials": {"walls": "brickwork", "floor": "carpet_cotton", "ceiling": "ceiling_fibre_absorber"},
        },
        {
            "name": "Medium room, hard surfaces (6x5x3m)",
            "dims": [6, 5, 3],
            "materials": {"walls": "rough_concrete", "floor": "marble_floor", "ceiling": "brickwork"},
        },
        {
            "name": "Large living room (8x6x3m)",
            "dims": [8, 6, 3],
            "materials": {"walls": "brickwork", "floor": "wood_16mm", "ceiling": "ceiling_fibre_absorber"},
        },
        {
            "name": "Blended distribution test (5x4x2.8m)",
            "dims": [5, 4, 2.8],
            "materials": {"walls": "plasterboard", "floor": "parquet_wood_22mm", "ceiling": "ceiling_fibre_absorber"},
            "distributions": {
                "walls":   {"plasterboard": 0.7, "wood_panel": 0.2, "brickwork": 0.1},
                "floor":   {"carpet_cotton": 0.4, "parquet_wood_22mm": 0.6},
                "ceiling": {"ceiling_fibre_absorber": 1.0},
            },
        },
    ]

    for cfg in configs:
        t_start = _time.perf_counter()
        rir, t60 = compute_rir(
            cfg["dims"], cfg["materials"],
            distributions=cfg.get("distributions"),
        )
        elapsed = (_time.perf_counter() - t_start) * 1000

        print(f"{cfg['name']}")
        print(f"  RIR length: {len(rir)} samples ({len(rir)/16000*1000:.1f}ms)")
        print(f"  T60:        {t60:.3f}s")
        print(f"  Compute:    {elapsed:.1f}ms")
        print()
