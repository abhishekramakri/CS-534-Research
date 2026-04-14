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
from shoebox import estimate_shoebox, estimate_shoebox_from_ply
from segmentation import (
    classify_materials,
    run_mobilenet_timing,
    save_segmentation_cache,
    load_segmentation_cache,
)
from scene_classifier import classify_scene
from acoustics import compute_rir
from render import apply_rir


def init_segmentation(image_path: str, cache_path: str, device: str | None = None):
    """
    One-time SegFormer run: classify materials and save to cache.
    Called when --init-segmentation is passed; does not run the full pipeline.
    """
    print("=" * 52)
    print("  Segmentation Init (SegFormer)")
    print("=" * 52)
    print(f"\nRunning SegFormer on: {image_path}")

    result = classify_materials(image_path, method="segformer", device=device)
    save_segmentation_cache(result, cache_path)

    print("\nMaterials:")
    for s in ("walls", "floor", "ceiling"):
        print(f"  {s:8s}: {result[s]}")
    print("\nDistributions:")
    for s, dist in result["distributions"].items():
        fracs = ", ".join(f"{m}: {v:.0%}" for m, v in dist.items())
        print(f"  {s:8s}: {fracs}")
    print(f"\nCache saved → {cache_path}")


def run(
    scene_dir: str,
    image_path: str,
    input_audio: str,
    output_audio: str,
    seg_method: str = "segformer",
    samosa_mode: bool = False,
    seg_cache: str = "segmentation_cache.json",
    fs: int = 16000,
    max_order: int | None = None,
    device: str | None = None,
    ply_path: str | None = None,
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
            if ply_path is not None:
                geo = estimate_shoebox_from_ply(ply_path)
            else:
                geo = estimate_shoebox(scene_dir)

    def _run_segmentation():
        nonlocal seg_result
        if samosa_mode:
            # SAMOSA emulation: load SegFormer cache for material accuracy,
            # run MobileNetV2 inference to match SAMOSA's ~10ms compute cost.
            # run_mobilenet_timing() returns inference-only ms (preprocessed
            # tensor cached, CUDA warmed up) — record that directly so we
            # don't inflate the number with image loading overhead.
            seg_result = load_segmentation_cache(seg_cache)
            import torch
            if device is None:
                if torch.cuda.is_available():
                    _device = "cuda"
                elif torch.backends.mps.is_available():
                    _device = "mps"
                else:
                    _device = "cpu"
            else:
                _device = device
            inference_ms = run_mobilenet_timing(image_path, _device)
            prof._timings["2. Seg (MobileNetV2 timing, SAMOSA mode)"] = inference_ms
        else:
            with prof.timer("2. Material segmentation"):
                seg_result = classify_materials(image_path, method=seg_method, device=device)

    mode_label = "SAMOSA emulation" if samosa_mode else seg_method
    print(f"\n[Phase 1] Running perception modules in parallel (seg={mode_label})...")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_run_shoebox),
            pool.submit(_run_segmentation),
        ]
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
    parser.add_argument("--scene",     default="scannet/scene0005",
                        help="Path to ScanNet scene dir (contains calibration.txt + images/)")
    parser.add_argument("--ply",       default=None,
                        help="Path to PLY mesh (overrides --scene for geometry; synthetic scenes only)")
    parser.add_argument("--image",     default=None,
                        help="Path to RGB image for segmentation. "
                             "If omitted, uses the first color frame from --scene.")
    parser.add_argument("--input",     default="audio/test.wav",
                        help="Dry input audio")
    parser.add_argument("--output",    default="audio/output_pipeline.wav",
                        help="Reverberant output audio")
    parser.add_argument("--seg-method", default="segformer",
                        choices=["segformer", "heuristic"],
                        help="Segmentation method (normal mode)")
    parser.add_argument("--fs",        type=int, default=16000,
                        help="Sample rate (Hz)")
    parser.add_argument("--max-order", type=int, default=None,
                        help="ISM reflection order (overrides scene preset)")
    parser.add_argument("--device",    default=None,
                        help="Torch device: cuda / cpu / auto")

    # SAMOSA emulation flags
    parser.add_argument("--samosa-mode", action="store_true",
                        help="SAMOSA emulation: use cached SegFormer materials + "
                             "MobileNetV2 timing dummy to match SAMOSA compute profile")
    parser.add_argument("--init-segmentation", action="store_true",
                        help="Run SegFormer once, save segmentation cache, then exit. "
                             "Required before --samosa-mode.")
    parser.add_argument("--seg-cache", default="segmentation_cache.json",
                        help="Path to segmentation cache file (default: segmentation_cache.json)")

    args = parser.parse_args()

    # Auto-pick the first color frame from the scene dir if --image not given
    import glob as _glob
    image_path = args.image
    if image_path is None:
        frames = sorted(_glob.glob(
            os.path.join(args.scene, "images", "*.color.png")
        ))
        if not frames:
            raise FileNotFoundError(
                f"No *.color.png frames found in {args.scene}/images/. "
                "Pass --image explicitly."
            )
        image_path = frames[0]
        print(f"[pipeline] Using image: {image_path}")

    if args.init_segmentation:
        init_segmentation(
            image_path=image_path,
            cache_path=args.seg_cache,
            device=args.device,
        )
        return

    run(
        scene_dir=args.scene,
        image_path=image_path,
        input_audio=args.input,
        output_audio=args.output,
        seg_method=args.seg_method,
        samosa_mode=args.samosa_mode,
        seg_cache=args.seg_cache,
        fs=args.fs,
        max_order=args.max_order,
        device=args.device,
        ply_path=args.ply,
    )


if __name__ == "__main__":
    main()
