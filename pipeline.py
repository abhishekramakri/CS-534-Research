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
import socket as _socket
from concurrent.futures import ThreadPoolExecutor, as_completed

from power_monitor import PowerMonitor
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


def _run_remote(server_addr: str, payload: dict) -> dict:
    """Send payload to the offload server and return its response."""
    from net_utils import send_msg, recv_msg
    host, port_str = server_addr.rsplit(":", 1)
    with _socket.create_connection((host, int(port_str)), timeout=120) as sock:
        send_msg(sock, payload)
        resp = recv_msg(sock)
    if resp.get("error"):
        raise RuntimeError(f"Server error: {resp['error']}")
    return resp


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
    server_addr: str | None = None,
    offload_start: int | None = None,
    offload_end: int | None = None,
):
    prof = Profiler()
    power = PowerMonitor()
    offloading = bool(server_addr and offload_start is not None)

    print("=" * 52)
    print("  SAMOSA Acoustic Rendering Pipeline")
    print("=" * 52)

    geo = None
    seg_result = None
    scene_type = None
    preset = None
    rir = None
    t60 = None

    if offloading:
        # --------------------------------------------------------------
        # OFFLOAD PATH: headset runs stages 1..(offload_start-1),
        # server runs offload_start..offload_end, headset finishes.
        # Audio rendering (stage 5) always runs on headset.
        # --------------------------------------------------------------
        print(f"\n[Offload] Stages {offload_start}–{offload_end} → {server_addr}")
        print(f"[Phase 1] Running local pre-offload stages...")
        power.start()

        # Local stage 1 (shoebox) — only if we go before offload_start
        if offload_start > 1:
            with prof.timer("1. Shoebox estimation"):
                if ply_path is not None:
                    geo = estimate_shoebox_from_ply(ply_path)
                else:
                    geo = estimate_shoebox(scene_dir)

        # Local stage 2 (segmentation) — only if offload starts at 3+
        if offload_start > 2:
            with prof.timer("2. Material segmentation"):
                seg_result = classify_materials(image_path, method=seg_method, device=device)

        # Local stage 3 (scene classification) — only if offload starts at 4
        if offload_start > 3:
            with prof.timer("3. Scene classification"):
                scene_type, preset = classify_scene(
                    material_distributions=seg_result.get("distributions"),
                    room_dims=geo["room_dims"],
                    best_materials={k: seg_result[k] for k in ("walls", "floor", "ceiling")},
                )

        # Build server payload
        payload: dict = {
            "stage_start": offload_start,
            "stage_end":   offload_end,
            "params": {
                "fs":          fs,
                "max_order":   max_order,
                "seg_method":  seg_method,
                "samosa_mode": samosa_mode,
                "device":      device,
                "image_ext":   os.path.splitext(image_path)[1] or ".png",
            },
        }

        if offload_start == 1:
            with open(ply_path, "rb") as f:
                payload["ply_bytes"] = f.read()

        if offload_start <= 2:
            with open(image_path, "rb") as f:
                payload["image_bytes"] = f.read()

        if offload_start >= 2 and geo is not None:
            payload["geo"] = {
                k: list(v) if hasattr(v, "tolist") else v
                for k, v in geo.items()
                if k in ("room_dims", "source_pos", "listener_pos")
            }

        if offload_start >= 3 and seg_result is not None:
            payload["seg"] = seg_result

        if offload_start >= 4 and preset is not None:
            payload["preset"] = preset

        # Remote call — timer captures full headset-perceived round-trip
        rtt_label = f"Server round-trip (stages {offload_start}–{offload_end})"
        print(f"[Phase 2] Sending to server ({server_addr})...")
        with prof.timer(rtt_label):
            resp = _run_remote(server_addr, payload)

        # Unpack server outputs
        if offload_start == 1:
            geo = resp["geo"]
        if offload_end >= 2:
            seg_result = resp["seg"]
        if offload_end >= 3:
            scene_type = resp["scene_type"]
            preset = resp["preset"]
        if offload_end >= 4:
            import numpy as _np
            rir = _np.asarray(resp["rir"])
            t60 = resp["t60"]

        # Remaining local stages after offload window
        if offload_end < 3:
            with prof.timer("3. Scene classification"):
                scene_type, preset = classify_scene(
                    material_distributions=seg_result.get("distributions"),
                    room_dims=geo["room_dims"],
                    best_materials={k: seg_result[k] for k in ("walls", "floor", "ceiling")},
                )

        if offload_end < 4:
            order = max_order if max_order is not None else preset["max_order"]
            print("\n[Phase 3] Synthesising RIR (local)...")
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
        else:
            order = max_order if max_order is not None else (preset or {}).get("max_order", "?")

    else:
        # --------------------------------------------------------------
        # LOCAL PATH: original parallel execution, unchanged
        # --------------------------------------------------------------
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
        power.start()

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(_run_shoebox), pool.submit(_run_segmentation)]
            for f in as_completed(futures):
                f.result()

        with prof.timer("3. Scene classification"):
            scene_type, preset = classify_scene(
                material_distributions=seg_result.get("distributions"),
                room_dims=geo["room_dims"],
                best_materials={k: seg_result[k] for k in ("walls", "floor", "ceiling")},
            )

        order = max_order if max_order is not None else preset["max_order"]
        print(f"\n  Scene type:  {scene_type} (max_order={order})")
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

    # ------------------------------------------------------------------
    # Print geometry / material summary (both paths)
    # ------------------------------------------------------------------
    if geo is not None:
        w, l, h = geo["room_dims"]
        print(f"  Shoebox:      {w:.2f} x {l:.2f} x {h:.2f} m")
        print(f"  Source:       {[round(v, 2) for v in geo['source_pos']]}")
        print(f"  Listener:    {[round(v, 2) for v in geo['listener_pos']]}")
    if seg_result is not None:
        print(f"  Floor mat:   {seg_result['floor']}")
        print(f"  Wall mat:    {seg_result['walls']}")
        print(f"  Ceiling mat: {seg_result['ceiling']}")
    if scene_type is not None:
        print(f"  Scene type:  {scene_type}")
    if rir is not None:
        print(f"\n  RIR length:  {len(rir)} samples ({len(rir) / fs * 1000:.1f} ms)")
        print(f"  T60:         {t60:.3f} s")
    if preset is not None and "target_rt60" in preset:
        print(f"  (preset target: {preset['target_rt60']:.1f} s)")

    # ------------------------------------------------------------------
    # Stage 5: Audio rendering (always local)
    # ------------------------------------------------------------------
    print(f"\n[Phase {'3' if offloading else '3'}] Rendering audio → {output_audio}")

    with prof.timer("5. Audio rendering"):
        apply_rir(input_audio, rir, output_audio)

    client_power = power.stop(duration_ms=prof.total_ms)

    # ------------------------------------------------------------------
    # Latency report
    # ------------------------------------------------------------------
    print(prof.summary())

    if offloading:
        server_timings = resp.get("timings", {})
        server_total = sum(server_timings.values())
        network_overhead = prof._timings[rtt_label] - server_total
        rtt_ms = prof._timings[rtt_label]
        print(f"  Server breakdown (stages {offload_start}–{offload_end}):")
        for name, ms in server_timings.items():
            print(f"    {name:<40s} {ms:7.1f} ms")
        print(f"    {'Server total':<40s} {server_total:7.1f} ms")
        print(f"    {'Network overhead':<40s} {network_overhead:7.1f} ms")
        print(f"    (round-trip wall-clock:               {rtt_ms:7.1f} ms)")
        print()

    # ------------------------------------------------------------------
    # Power report
    # ------------------------------------------------------------------
    server_power = resp.get("power") if offloading else None
    if client_power or server_power:
        print("=" * 52)
        print("  Power Usage")
        print("=" * 52)
        if client_power:
            e = f"  {client_power['energy_mJ']:.1f} mJ" if "energy_mJ" in client_power else ""
            print(f"  Client  [{client_power['backend']}]")
            print(f"    Avg {client_power['avg_mW']:,.0f} mW  "
                  f"Peak {client_power['peak_mW']:,.0f} mW{e}")
        if server_power:
            e = f"  {server_power['energy_mJ']:.1f} mJ" if "energy_mJ" in server_power else ""
            print(f"  Server  [{server_power['backend']}]")
            print(f"    Avg {server_power['avg_mW']:,.0f} mW  "
                  f"Peak {server_power['peak_mW']:,.0f} mW{e}")
        print("=" * 52)
        print()

    _server_breakdown = None
    if offloading:
        _server_breakdown = {
            "timings":            server_timings,
            "server_total_ms":    server_total,
            "network_overhead_ms": network_overhead,
            "rtt_ms":             rtt_ms,
        }

    return {
        "geometry":        geo,
        "materials":       {k: seg_result[k] for k in ("walls", "floor", "ceiling")} if seg_result else None,
        "distributions":   seg_result.get("distributions") if seg_result else None,
        "scene_type":      scene_type,
        "preset":          preset,
        "rir":             rir,
        "t60":             t60,
        "timings":         prof.timings,
        "power":           {"client": client_power, "server": server_power},
        "server_breakdown": _server_breakdown,
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

    # Offloading flags
    parser.add_argument("--server", default=None, metavar="HOST:PORT",
                        help="Offload server address, e.g. 192.168.1.100:9000")
    parser.add_argument("--offload", nargs=2, type=int, metavar=("START", "END"),
                        default=None,
                        help="Pipeline stage range to run on server (1-4). "
                             "E.g. --offload 2 4 (Config C), --offload 1 4 (Config D)")

    args = parser.parse_args()

    # Validate offload arguments
    if args.offload is not None:
        s, e = args.offload
        if not (1 <= s <= e <= 4):
            parser.error("--offload START END must satisfy 1 <= START <= END <= 4")
        if s == 1 and args.ply is None:
            parser.error("--offload 1 ... requires --ply "
                         "(ScanNet scene dirs cannot be transferred over the network)")
        if args.server is None:
            parser.error("--offload requires --server HOST:PORT")
        if args.samosa_mode and s > 2:
            parser.error("--samosa-mode with --offload START > 2 would run segmentation "
                         "locally, which requires a --seg-cache on the headset")

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
        server_addr=args.server,
        offload_start=args.offload[0] if args.offload else None,
        offload_end=args.offload[1] if args.offload else None,
    )


if __name__ == "__main__":
    main()
