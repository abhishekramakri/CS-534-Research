"""
server.py
TCP server that runs pipeline stages on behalf of the headset client.

Usage:
    python server.py --port 9000 --device cuda

The server accepts one connection at a time. Each connection carries a single
request/response pair (stage range + inputs → stage outputs + timings).
"""

import argparse
import os
import socket
import tempfile

from acoustics import compute_rir
from net_utils import recv_msg, send_msg
from profiler import Profiler
from scene_classifier import classify_scene
from segmentation import classify_materials
from shoebox import estimate_shoebox_from_ply


def handle_request(req: dict, device: str) -> dict:
    prof = Profiler()
    stage_start = req["stage_start"]
    stage_end = req["stage_end"]
    params = req["params"]
    temp_files: list[str] = []

    geo = seg = scene_type = preset = rir = t60 = None

    try:
        # ----------------------------------------------------------------
        # Stage 1: Shoebox estimation
        # ----------------------------------------------------------------
        if stage_start == 1:
            with prof.timer("1. Shoebox estimation"):
                tmp = tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
                tmp.write(req["ply_bytes"])
                tmp.close()
                temp_files.append(tmp.name)
                geo = estimate_shoebox_from_ply(tmp.name)
            geo_out = {k: list(geo[k]) if hasattr(geo[k], "tolist") else geo[k]
                       for k in ("room_dims", "source_pos", "listener_pos")}
        else:
            geo = req.get("geo")  # slim dict from client (already plain lists)
            geo_out = None

        # ----------------------------------------------------------------
        # Stage 2: Material segmentation
        # ----------------------------------------------------------------
        if stage_start <= 2 <= stage_end:
            img_ext = params.get("image_ext", ".png")
            with prof.timer("2. Material segmentation"):
                tmp = tempfile.NamedTemporaryFile(suffix=img_ext, delete=False)
                tmp.write(req["image_bytes"])
                tmp.close()
                temp_files.append(tmp.name)
                seg = classify_materials(
                    tmp.name,
                    method=params.get("seg_method", "segformer"),
                    device=device,
                )
        else:
            seg = req.get("seg")

        # ----------------------------------------------------------------
        # Stage 3: Scene classification
        # ----------------------------------------------------------------
        if stage_start <= 3 <= stage_end:
            with prof.timer("3. Scene classification"):
                scene_type, preset = classify_scene(
                    material_distributions=seg.get("distributions"),
                    room_dims=geo["room_dims"],
                    best_materials={k: seg[k] for k in ("walls", "floor", "ceiling")},
                )
        else:
            scene_type = req.get("scene_type")
            preset = req.get("preset")

        # ----------------------------------------------------------------
        # Stage 4: RIR synthesis
        # ----------------------------------------------------------------
        if stage_start <= 4 <= stage_end:
            order = params.get("max_order") or preset["max_order"]
            with prof.timer("4. RIR synthesis"):
                rir, t60 = compute_rir(
                    room_dims=geo["room_dims"],
                    materials={k: seg[k] for k in ("walls", "floor", "ceiling")},
                    source_pos=geo["source_pos"],
                    listener_pos=geo["listener_pos"],
                    fs=params.get("fs", 16000),
                    max_order=order,
                    distributions=seg.get("distributions"),
                )

    finally:
        for path in temp_files:
            try:
                os.unlink(path)
            except OSError:
                pass

    resp: dict = {"error": None, "timings": prof.timings}
    if geo_out is not None:
        resp["geo"] = geo_out
    if seg is not None and stage_end >= 2:
        resp["seg"] = seg
    if scene_type is not None and stage_end >= 3:
        resp["scene_type"] = scene_type
        resp["preset"] = preset
    if rir is not None and stage_end >= 4:
        resp["rir"] = rir   # ndarray — net_utils encodes as base64
        resp["t60"] = t60
    return resp


def serve(host: str, port: int, device: str) -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"[server] Listening on {host}:{port}  device={device}")
    print("[server] Waiting for connections (Ctrl-C to stop)...")

    while True:
        conn, addr = srv.accept()
        print(f"[server] Connection from {addr[0]}:{addr[1]}")
        try:
            req = recv_msg(conn)
            stages = f"{req.get('stage_start')}–{req.get('stage_end')}"
            print(f"[server] Running stages {stages}")
            resp = handle_request(req, device)
            timings_str = "  ".join(
                f"{k}: {v:.1f}ms" for k, v in resp.get("timings", {}).items()
            )
            print(f"[server] Done — {timings_str}")
        except Exception as exc:
            print(f"[server] Error: {exc}")
            resp = {"error": str(exc), "timings": {}}
        send_msg(conn, resp)
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline offload server — runs acoustic pipeline stages remotely"
    )
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=9000,
                        help="TCP port (default: 9000)")
    parser.add_argument("--device", default=None,
                        help="Torch device: cuda / cpu (default: auto-detect)")
    args = parser.parse_args()

    device = args.device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    serve(args.host, args.port, device)


if __name__ == "__main__":
    main()
