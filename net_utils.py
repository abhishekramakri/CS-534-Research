"""
net_utils.py
TCP framing and JSON serialization for the pipeline offloading protocol.

Protocol: 8-byte big-endian uint64 length prefix + UTF-8 JSON body.
numpy ndarrays and raw bytes blobs are encoded inline as base64 JSON objects.
"""

import base64
import json
import socket
import struct

import numpy as np


class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                "__ndarray__": True,
                "dtype": str(obj.dtype),
                "shape": list(obj.shape),
                "b64": base64.b64encode(obj.tobytes()).decode("ascii"),
            }
        if isinstance(obj, bytes):
            return {"__bytes__": True, "b64": base64.b64encode(obj).decode("ascii")}
        return super().default(obj)


def _hook(d: dict):
    if d.get("__ndarray__"):
        raw = base64.b64decode(d["b64"])
        return np.frombuffer(raw, dtype=d["dtype"]).reshape(d["shape"]).copy()
    if d.get("__bytes__"):
        return base64.b64decode(d["b64"])
    return d


def pack(payload: dict) -> bytes:
    body = json.dumps(payload, cls=_Encoder).encode("utf-8")
    return struct.pack(">Q", len(body)) + body


def unpack(data: bytes) -> dict:
    return json.loads(data.decode("utf-8"), object_hook=_hook)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed before all bytes received")
        buf.extend(chunk)
    return bytes(buf)


def send_msg(sock: socket.socket, payload: dict) -> None:
    sock.sendall(pack(payload))


def recv_msg(sock: socket.socket) -> dict:
    header = _recv_exact(sock, 8)
    length = struct.unpack(">Q", header)[0]
    body = _recv_exact(sock, length)
    return unpack(body)
