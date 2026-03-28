"""
acoustics.py
Computes a Room Impulse Response (RIR) from room dimensions and surface materials
using pyroomacoustics. This is the core of the SAMOSA pipeline.
"""

import numpy as np
import pyroomacoustics as pra


# Maps surface type → pyroomacoustics material name.
# These will later be filled in by segmentation.py outputs.
# Full list of available materials: pra.materials_absorption_table.keys()
DEFAULT_MATERIALS = {
    "walls":   "brickwork",
    "floor":   "carpet_cotton",
    "ceiling": "ceiling_fibre_absorber",
}


def compute_rir(
    room_dims,
    materials=None,
    source_pos=None,
    listener_pos=None,
    fs=16000,
    max_order=17,
):
    """
    Compute a Room Impulse Response.

    Args:
        room_dims:    [width, length, height] in meters
        materials:    dict with keys "walls", "floor", "ceiling" → material name string
        source_pos:   [x, y, z] of audio source. Defaults to center of room offset slightly.
        listener_pos: [x, y, z] of listener. Defaults to opposite offset from center.
        fs:           sample rate (Hz)
        max_order:    image source method reflection order (3 is a good tradeoff)

    Returns:
        rir: np.ndarray — the room impulse response
        t60: float — reverberation time in seconds
    """
    if materials is None:
        materials = DEFAULT_MATERIALS

    w, l, h = room_dims

    # Default positions: source and listener on opposite sides of room center
    if source_pos is None:
        source_pos = [w * 0.25, l * 0.5, h * 0.5]
    if listener_pos is None:
        listener_pos = [w * 0.75, l * 0.5, h * 0.5]

    # Build per-surface material object
    # ShoeBox surface order: east, west, north, south, floor, ceiling
    room_materials = pra.make_materials(
        east=materials["walls"],
        west=materials["walls"],
        north=materials["walls"],
        south=materials["walls"],
        floor=materials["floor"],
        ceiling=materials["ceiling"],
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
    ]

    for cfg in configs:
        import time
        t_start = time.perf_counter()
        rir, t60 = compute_rir(cfg["dims"], cfg["materials"])
        elapsed = (time.perf_counter() - t_start) * 1000

        print(f"{cfg['name']}")
        print(f"  RIR length: {len(rir)} samples ({len(rir)/16000*1000:.1f}ms)")
        print(f"  T60:        {t60:.3f}s")
        print(f"  Compute time: {elapsed:.1f}ms")
        print()
