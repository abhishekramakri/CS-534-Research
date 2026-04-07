"""
pipeline.py
End-to-end SAMOSA-style acoustic rendering pipeline.

Three perception modules run in parallel (matching SAMOSA architecture):
  1. Shoebox estimation  (PLY mesh  → room geometry)
  2. Material segmentation (RGB image → per-surface materials)
  3. Scene classification  (materials + dims → acoustic presets)

Then sequential:
  4. RIR synthesis         (geometry + materials + presets → impulse response)
  5. Audio rendering       (dry audio * RIR → reverberant output)

Every stage is timed via profiler.py so we have a latency baseline
for future offloading experiments.
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from profiler import Profiler
from shoebox import estimate_shoebox
from segmentation import classify_materials
from scene_classifier import classify_scene
from acoustics import compute_rir
from render import apply_rir


def run(
    ply_path: str,
    image_path: str,
    input_audio: str,
    output_audio: str,
    seg_method: str = "segformer",
    fs: int = 16000,
    max_order: int | None = None,
    device: str | None = None,
):
    prof = Profiler()

    print("=" * 52)
    print("  SAMOSA Acoustic Rendering Pipeline")
    print("=" * 52)

    # ------------------------------------------------------------------
    # Phase 1: Parallel perception modules
    # ------------------------------------------------------------------
    geo = None
    seg_result = None

    def _run_shoebox():
        nonlocal geo
        with prof.timer("1. Shoebox estimation"):
            geo = estimate_shoebox(ply_path)

    def _run_segmentation():
        nonlocal seg_result
        with prof.timer("2. Material segmentation"):
            seg_result = classify_materials(image_path, method=seg_method, device=device)

    print("\n[Phase 1] Running perception modules in parallel...")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_run_shoebox),
            pool.submit(_run_segmentation),
        ]
        # Wait and re-raise any exceptions
        for f in as_completed(futures):
            f.result()

    w, l, h = geo["room_dims"]
    print(f"  Shoebox:      {w:.2f} x {l:.2f} x {h:.2f} m")
    print(f"  Source:       {[round(v, 2) for v in geo['source_pos']]}")
    print(f"  Listener:    {[round(v, 2) for v in geo['listener_pos']]}")
    print(f"  Floor mat:   {seg_result['floor']}")
    print(f"  Wall mat:    {seg_result['walls']}")
    print(f"  Ceiling mat: {seg_result['ceiling']}")

    # ------------------------------------------------------------------
    # Phase 2: Scene classification (uses outputs from phase 1)
    # ------------------------------------------------------------------
    with prof.timer("3. Scene classification"):
        scene_type, preset = classify_scene(
            material_distributions=seg_result.get("distributions"),
            room_dims=geo["room_dims"],
            best_materials={k: seg_result[k] for k in ("walls", "floor", "ceiling")},
        )

    order = max_order if max_order is not None else preset["max_order"]
    print(f"\n  Scene type:  {scene_type} (max_order={order})")

    # ------------------------------------------------------------------
    # Phase 3: RIR synthesis
    # ------------------------------------------------------------------
    print("\n[Phase 2] Synthesising RIR...")

    with prof.timer("4. RIR synthesis"):
        rir, t60 = compute_rir(
            room_dims=geo["room_dims"],
            materials={k: seg_result[k] for k in ("walls", "floor", "ceiling")},
            source_pos=geo["source_pos"],
            listener_pos=geo["listener_pos"],
            fs=fs,
            max_order=order,
            distributions=seg_result.get("distributions"),
        )

    print(f"  RIR length:  {len(rir)} samples ({len(rir) / fs * 1000:.1f} ms)")
    print(f"  T60:         {t60:.3f} s  (preset target: {preset['target_rt60']:.1f} s)")

    # ------------------------------------------------------------------
    # Phase 4: Audio rendering
    # ------------------------------------------------------------------
    print(f"\n[Phase 3] Rendering audio → {output_audio}")

    with prof.timer("5. Audio rendering"):
        apply_rir(input_audio, rir, output_audio)

    # ------------------------------------------------------------------
    # Latency report
    # ------------------------------------------------------------------
    print(prof.summary())

    return {
        "geometry":      geo,
        "materials":     {k: seg_result[k] for k in ("walls", "floor", "ceiling")},
        "distributions": seg_result.get("distributions"),
        "scene_type":    scene_type,
        "preset":        preset,
        "rir":           rir,
        "t60":           t60,
        "timings":       prof.timings,
    }


def main():
    parser = argparse.ArgumentParser(
        description="SAMOSA-style end-to-end acoustic rendering pipeline",
    )
    parser.add_argument("--ply",       default="testing/modernroom.ply",
                        help="Path to room PLY mesh")
    parser.add_argument("--image",     default="testing/modernroom.png",
                        help="Path to room RGB render")
    parser.add_argument("--input",     default="audio/test.wav",
                        help="Dry input audio")
    parser.add_argument("--output",    default="audio/output_pipeline.wav",
                        help="Reverberant output audio")
    parser.add_argument("--seg-method", default="segformer",
                        choices=["segformer", "heuristic"],
                        help="Segmentation method")
    parser.add_argument("--fs",        type=int, default=16000,
                        help="Sample rate (Hz)")
    parser.add_argument("--max-order", type=int, default=None,
                        help="ISM reflection order (overrides scene preset)")
    parser.add_argument("--device",    default=None,
                        help="Torch device: cuda / cpu / auto")

    args = parser.parse_args()

    run(
        ply_path=args.ply,
        image_path=args.image,
        input_audio=args.input,
        output_audio=args.output,
        seg_method=args.seg_method,
        fs=args.fs,
        max_order=args.max_order,
        device=args.device,
    )


if __name__ == "__main__":
    main()
