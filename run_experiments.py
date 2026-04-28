#!/usr/bin/env python3
"""
run_experiments.py
Runs configs × modes (normal / SAMOSA) for every scene and writes results to CSV.

Two scene sources can be combined in one run:
  1. PLY+image pairs in testing/   → configs A B C D
  2. ScanNet scene directories     → configs A B C only (D requires PLY transfer)

Usage:
    # PLY scenes only
    python run_experiments.py --server 192.168.50.91:9001

    # ScanNet scenes only
    python run_experiments.py --server 192.168.50.91:9001 --scannet-dir /path/to/scannet

    # Both
    python run_experiments.py --server 192.168.50.91:9001 --scannet-dir /path/to/scannet

The server only needs to be started once (no --samosa-mode flag needed):
    python server.py --port 9001 --device cuda
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Config definitions
# ---------------------------------------------------------------------------
CONFIGS = {
    "A": {"offload_start": None, "offload_end": None},   # fully local
    "B": {"offload_start": 2,    "offload_end": 2},
    "C": {"offload_start": 2,    "offload_end": 4},
    "D": {"offload_start": 1,    "offload_end": 4},
}

TESTING_DIR      = Path("testing")
RESULTS_DIR      = Path("results")
AUDIO_INPUT      = "audio/test.wav"
N_RUNS           = 3     # recorded runs per combination
N_WARMUP         = 1     # discarded warmup runs per combination
SCANNET_CONFIGS  = ["A", "B", "C"]   # D excluded — requires PLY transfer to server


def find_pairs(testing_dir: Path) -> list[tuple[str, Path, Path]]:
    """Return [(scene_name, ply_path, image_path)] for each matched PLY+image pair."""
    plys   = {p.stem: p for p in sorted(testing_dir.glob("*.ply"))}
    images = {}
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for img in sorted(testing_dir.glob(ext)):
            if img.stem not in images:
                images[img.stem] = img
    return [(name, plys[name], images[name])
            for name in sorted(plys) if name in images]


def find_scannet_scenes(scannet_dir: Path) -> list[tuple[str, Path, Path]]:
    """
    Return [(scene_name, scene_dir, image_path)] for each ScanNet scene.
    Looks for color frames in color/, rgb/, frames/color/, or the scene dir itself.
    Picks the middle frame as the representative image.
    """
    results = []
    for scene_path in sorted(scannet_dir.iterdir()):
        if not scene_path.is_dir():
            continue
        image_path = None
        for subdir in ("color", "rgb", "frames/color", "images"):
            candidate = scene_path / subdir
            if candidate.is_dir():
                frames = sorted(candidate.glob("*.jpg")) + sorted(candidate.glob("*.png"))
                if frames:
                    image_path = frames[len(frames) // 2]
                    break
        if image_path is None:
            for ext in ("*.jpg", "*.png", "*.jpeg"):
                frames = sorted(scene_path.glob(ext))
                if frames:
                    image_path = frames[len(frames) // 2]
                    break
        if image_path is None:
            print(f"  [warn] No images found in {scene_path.name}, skipping")
            continue
        results.append((scene_path.name, scene_path, image_path))
    return results


def get_timing(timings: dict, *prefixes: str) -> float | None:
    """Return the first timing whose key starts with one of the given prefixes."""
    for prefix in prefixes:
        for k, v in timings.items():
            if k.strip().startswith(prefix):
                return v
    return None


def run_one(scene: str, ply: Path | None, image: Path, config: str,
            samosa: bool, server: str | None, seg_cache: str,
            run_idx: int, scene_dir: Path | None = None) -> dict:
    """Run the pipeline for one combination and return a flat metrics dict.

    ply and scene_dir are mutually exclusive sources for geometry:
      - ply       : pre-built mesh (testing/ pairs, all configs)
      - scene_dir : ScanNet directory (configs A/B/C only; D excluded)
    """
    import pipeline as pl

    cfg = CONFIGS[config]
    offload_start = cfg["offload_start"]
    offload_end   = cfg["offload_end"]
    needs_server  = offload_start is not None

    output_wav = str(RESULTS_DIR / f"{scene}_cfg{config}_{'samosa' if samosa else 'normal'}_r{run_idx}.wav")

    result = pl.run(
        scene_dir     = str(scene_dir) if scene_dir else str(TESTING_DIR / scene),
        image_path    = str(image),
        input_audio   = AUDIO_INPUT,
        output_audio  = output_wav,
        ply_path      = str(ply) if ply else None,
        seg_method    = "segformer",
        samosa_mode   = samosa and (offload_start is None),  # local SAMOSA only for Config A
        seg_cache     = seg_cache,
        server_addr   = server if needs_server else None,
        offload_start = offload_start,
        offload_end   = offload_end,
    )

    timings  = result.get("timings", {})
    power    = result.get("power", {})
    sb       = result.get("server_breakdown")  # None for Config A
    cp       = power.get("client") or {}
    sp       = power.get("server") or {}

    # Server-side per-stage timings
    srv_t    = sb["timings"] if sb else {}

    row = {
        "scene":               scene,
        "config":              config,
        "samosa_mode":         samosa,
        "run":                 run_idx,
        # --- client-side stage latencies ---
        "shoebox_ms":          get_timing(timings, "1."),
        "seg_ms":              get_timing(timings, "2."),           # present only if local (Config A)
        "scene_class_ms":      get_timing(timings, "3."),
        "rir_ms":              get_timing(timings, "4."),
        "audio_ms":            get_timing(timings, "5."),
        "client_total_ms":     sum(timings.values()) if timings else None,
        # --- offload / network ---
        "rtt_ms":              sb["rtt_ms"]             if sb else None,
        "network_overhead_ms": sb["network_overhead_ms"] if sb else None,
        "server_total_ms":     sb["server_total_ms"]    if sb else None,
        # --- server-side stage latencies ---
        "srv_seg_ms":          get_timing(srv_t, "2."),
        "srv_mobilenet_ms":    get_timing(srv_t, "   MobileNetV3"),
        "srv_scene_ms":        get_timing(srv_t, "3."),
        "srv_rir_ms":          get_timing(srv_t, "4."),
        "srv_shoebox_ms":      get_timing(srv_t, "1."),
        # --- client power ---
        "client_avg_mW":       cp.get("avg_mW"),
        "client_peak_mW":      cp.get("peak_mW"),
        "client_energy_mJ":    cp.get("energy_mJ"),
        "client_power_backend": cp.get("backend"),
        # --- server power ---
        "server_avg_mW":       sp.get("avg_mW"),
        "server_peak_mW":      sp.get("peak_mW"),
        "server_energy_mJ":    sp.get("energy_mJ"),
        "server_power_backend": sp.get("backend"),
        # --- audio quality ---
        "t60":                 result.get("t60"),
    }
    return row


def init_seg_caches(scenes: list[tuple[str, Path | None, Path]], server: str | None) -> dict[str, str]:
    """
    Pre-generate segmentation caches for Config A SAMOSA mode.
    Accepts [(scene_name, ply_or_None, image_path)].
    Returns {scene_name: cache_path}.
    """
    import pipeline as pl
    caches = {}
    for scene, _, image in scenes:
        cache_path = str(RESULTS_DIR / f"{scene}_seg_cache.json")
        if not os.path.exists(cache_path):
            print(f"  [init-seg] Generating cache for {scene}...")
            pl.init_segmentation(str(image), cache_path)
        caches[scene] = cache_path
    return caches


def main():
    parser = argparse.ArgumentParser(description="Automated experiment runner")
    parser.add_argument("--server",  default="192.168.50.91:9001",
                        help="Server HOST:PORT (default: 192.168.50.91:9001)")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()),
                        choices=list(CONFIGS.keys()),
                        help="Configs for PLY scenes (default: A B C D)")
    parser.add_argument("--scannet-dir", default=None, metavar="PATH",
                        help="Root directory of ScanNet scenes (each subdirectory is one scene)")
    parser.add_argument("--scannet-configs", nargs="+", default=SCANNET_CONFIGS,
                        choices=list(CONFIGS.keys()),
                        help=f"Configs for ScanNet scenes (default: {' '.join(SCANNET_CONFIGS)})")
    parser.add_argument("--no-ply", action="store_true",
                        help="Skip PLY+image pair scenes from testing/")
    parser.add_argument("--no-samosa", action="store_true",
                        help="Skip SAMOSA mode runs")
    parser.add_argument("--runs",    type=int, default=N_RUNS,
                        help=f"Recorded runs per combination (default: {N_RUNS})")
    parser.add_argument("--warmup",  type=int, default=N_WARMUP,
                        help=f"Warmup runs to discard (default: {N_WARMUP})")
    parser.add_argument("--output",  default="results/metrics.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    modes = [False, True] if not args.no_samosa else [False]

    # --- Collect scenes from both sources ---
    # Each entry: (scene_name, ply_or_None, image, allowed_configs, scene_dir_or_None)
    all_scenes: list[tuple[str, Path | None, Path, list[str], Path | None]] = []

    if not args.no_ply:
        pairs = find_pairs(TESTING_DIR)
        if not pairs:
            print("No PLY+image pairs found in testing/", file=sys.stderr)
            if args.scannet_dir is None:
                sys.exit(1)
        else:
            print(f"PLY scenes ({len(pairs)}): {[p[0] for p in pairs]}")
            for name, ply, img in pairs:
                all_scenes.append((name, ply, img, args.configs, None))

    if args.scannet_dir:
        scannet_scenes = find_scannet_scenes(Path(args.scannet_dir))
        if not scannet_scenes:
            print(f"No ScanNet scenes found in {args.scannet_dir}", file=sys.stderr)
        else:
            # Enforce configs A/B/C — D requires PLY transfer which ScanNet can't do
            valid_sn_configs = [c for c in args.scannet_configs if c != "D"]
            if "D" in args.scannet_configs:
                print("  [warn] Config D excluded for ScanNet (requires PLY transfer)")
            print(f"ScanNet scenes ({len(scannet_scenes)}): {[s[0] for s in scannet_scenes]}")
            for name, sdir, img in scannet_scenes:
                all_scenes.append((name, None, img, valid_sn_configs, sdir))

    if not all_scenes:
        print("No scenes to run.", file=sys.stderr)
        sys.exit(1)

    # Pre-generate seg caches (needed for Config A + SAMOSA)
    seg_caches: dict[str, str] = {}
    if not args.no_samosa:
        needs_cache = [(name, ply, img) for name, ply, img, cfgs, _ in all_scenes if "A" in cfgs]
        if needs_cache:
            print("\nInitialising segmentation caches for SAMOSA mode...")
            seg_caches = init_seg_caches(needs_cache, args.server)
    # Fallback paths for scenes that don't need a cache (won't be used)
    for name, *_ in all_scenes:
        seg_caches.setdefault(name, f"results/{name}_seg_cache.json")

    # Build full run list: (scene, ply, image, config, samosa, r, scene_dir)
    runs = []
    for name, ply, img, cfgs, sdir in all_scenes:
        for config in cfgs:
            for samosa in modes:
                for r in range(args.warmup + args.runs):
                    runs.append((name, ply, img, config, samosa, r, sdir))

    total = len(runs)
    print(f"\nTotal runs: {total}  ({args.warmup} warmup + {args.runs} recorded each)\n")

    csv_path = Path(args.output)
    csv_path.parent.mkdir(exist_ok=True)

    fieldnames = [
        "scene", "config", "samosa_mode", "run",
        "shoebox_ms", "seg_ms", "scene_class_ms", "rir_ms", "audio_ms", "client_total_ms",
        "rtt_ms", "network_overhead_ms", "server_total_ms",
        "srv_seg_ms", "srv_mobilenet_ms", "srv_scene_ms", "srv_rir_ms", "srv_shoebox_ms",
        "client_avg_mW", "client_peak_mW", "client_energy_mJ", "client_power_backend",
        "server_avg_mW", "server_peak_mW", "server_energy_mJ", "server_power_backend",
        "t60",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for i, (scene, ply, image, config, samosa, r, sdir) in enumerate(runs):
            is_warmup = r < args.warmup
            recorded_r = r - args.warmup
            src_label = "ScanNet" if sdir else "PLY"
            label = (f"[{i+1}/{total}] {scene} ({src_label}) Config {config} "
                     f"{'SAMOSA' if samosa else 'normal':6s} "
                     f"{'(warmup)' if is_warmup else f'run {recorded_r+1}'}")
            print(label, end=" ... ", flush=True)

            t0 = time.perf_counter()
            try:
                row = run_one(scene, ply, image, config, samosa,
                              args.server, seg_caches[scene], recorded_r,
                              scene_dir=sdir)
                elapsed = time.perf_counter() - t0
                print(f"{row['client_total_ms']:.0f} ms  ({elapsed:.1f}s wall)")

                if not is_warmup:
                    writer.writerow(row)
                    f.flush()

            except Exception as exc:
                elapsed = time.perf_counter() - t0
                print(f"FAILED ({elapsed:.1f}s): {exc}")

    print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    main()
