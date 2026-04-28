"""
segmentation.py
Classifies room surface materials from an RGB image using semantic segmentation.

Primary path:  SegFormer-b0 (ADE20K, 150 classes) — pixel-level surface
               identification, then map ADE20K classes to pyroomacoustics
               material names.  Outputs both best-match strings (compatible
               with acoustics.compute_rir) AND per-surface confidence
               distributions for analysis.

Fallback path: Heuristic region split + colour-based material guess
               (no ML model required, for quick testing).
"""

import warnings
import numpy as np
from PIL import Image

# -----------------------------------------------------------------------
# ADE20K class → (surface, pyroomacoustics material)
# Only classes relevant to indoor room acoustics are listed.
# -----------------------------------------------------------------------
ADE20K_TO_MATERIAL = {
    # floor surfaces
    3:  ("floor",   "wood_16mm"),              # "floor, flooring"
    28: ("floor",   "carpet_cotton"),           # "rug, carpet, carpeting"

    # wall surfaces
    0:  ("walls",   "plasterboard"),            # "wall"
    8:  ("walls",   "glass_window"),            # "windowpane"
    14: ("walls",   "wooden_door"),             # "door"
    18: ("walls",   "curtains_densely_woven"),  # "curtain" — high absorption

    # ceiling surfaces
    5:  ("ceiling", "ceiling_fibre_absorber"),  # "ceiling"
}

# Which ADE20K class IDs belong to each surface
SURFACE_CLASS_IDS = {
    "floor":   {3, 28},
    "walls":   {0, 8, 14, 18},
    "ceiling": {5},
}

# Used when segmentation finds nothing or no model is available
FALLBACK_MATERIALS = {
    "walls":   "plasterboard",
    "floor":   "wood_16mm",
    "ceiling": "ceiling_fibre_absorber",
}


# -----------------------------------------------------------------------
# SegFormer inference
# -----------------------------------------------------------------------

# Module-level cache so the model is only loaded once per process.
_segformer_cache: dict = {}


def _get_segformer(device: str):
    """Load (or return cached) SegFormer processor + model on the given device."""
    if _segformer_cache.get("device") != device:
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        model_id = "nvidia/segformer-b0-finetuned-ade-512-512"
        processor = SegformerImageProcessor.from_pretrained(model_id)
        model = SegformerForSemanticSegmentation.from_pretrained(model_id)
        model.to(device).eval()
        _segformer_cache["processor"] = processor
        _segformer_cache["model"] = model
        _segformer_cache["device"] = device
    return _segformer_cache["processor"], _segformer_cache["model"]


def _run_segformer(img_array: np.ndarray, device: str, max_dim: int = 512):
    """
    Run SegFormer-b0 (ADE20K) on an HxWx3 uint8 array.
    Returns seg_map [H,W] (at original resolution).

    The image is downscaled to max_dim before inference (SegFormer-b0 was
    trained at 512x512).  The segmentation map is upsampled back to the
    original size.  This is significantly faster for high-res inputs.
    """
    import torch
    import torch.nn.functional as F

    processor, model = _get_segformer(device)

    H_orig, W_orig = img_array.shape[:2]

    # Downscale to max_dim on the longest side (matches training resolution)
    scale = min(max_dim / H_orig, max_dim / W_orig, 1.0)
    if scale < 1.0:
        H_new, W_new = int(H_orig * scale), int(W_orig * scale)
        pil_img = Image.fromarray(img_array).resize((W_new, H_new), Image.BILINEAR)
    else:
        pil_img = Image.fromarray(img_array)

    inputs = processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits                       # (1, 150, H/4, W/4)

    # Upsample back to original resolution for pixel-accurate surface areas
    upsampled = F.interpolate(
        logits, size=(H_orig, W_orig), mode="bilinear", align_corners=False
    )
    seg_map = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()   # (H, W)
    return seg_map


def _distributions_from_segmap(seg_map: np.ndarray) -> tuple[dict, dict]:
    """
    Given a per-pixel ADE20K class map, compute:
      best_materials: {"walls": str, "floor": str, "ceiling": str}
      distributions:  {"walls": {mat: frac, ...}, ...}
    """
    best = {}
    dists = {}

    for surface, class_ids in SURFACE_CLASS_IDS.items():
        # Build a mask of all pixels that belong to this surface
        mask = np.isin(seg_map, list(class_ids))
        total = mask.sum()

        if total == 0:
            best[surface] = FALLBACK_MATERIALS[surface]
            dists[surface] = {FALLBACK_MATERIALS[surface]: 1.0}
            continue

        # Count sub-class pixels → material fractions
        dist: dict[str, float] = {}
        for cid in class_ids:
            count = ((seg_map == cid) & mask).sum()
            if count > 0 and cid in ADE20K_TO_MATERIAL:
                mat_name = ADE20K_TO_MATERIAL[cid][1]
                dist[mat_name] = dist.get(mat_name, 0.0) + float(count) / float(total)

        if not dist:
            dist = {FALLBACK_MATERIALS[surface]: 1.0}

        dists[surface] = dist
        best[surface] = max(dist, key=dist.get)

    return best, dists


# -----------------------------------------------------------------------
# Heuristic fallback (no ML model needed)
# -----------------------------------------------------------------------
def _heuristic_classify(img_array: np.ndarray,
                        floor_frac: float,
                        ceiling_frac: float) -> tuple[dict, dict]:
    """
    Region-split + colour heuristic.  Very rough, but runs instantly
    and needs zero dependencies beyond numpy/PIL.
    """
    H, W = img_array.shape[:2]
    floor_top = int(H * (1 - floor_frac))
    ceil_bot = int(H * ceiling_frac)

    floor_region = img_array[floor_top:, :, :]
    ceil_region = img_array[:ceil_bot, :, :]
    wall_region = img_array[ceil_bot:floor_top, :, :]

    def _brightness(region):
        return region.mean() / 255.0

    def _warmth(region):
        r, g, b = region[:, :, 0].mean(), region[:, :, 1].mean(), region[:, :, 2].mean()
        return (r - b) / 255.0

    # Floor
    fb = _brightness(floor_region)
    fw = _warmth(floor_region)
    if fb < 0.3:
        floor_mat = "wood_16mm"     # dark → wood
    elif fw > 0.05 and fb < 0.6:
        floor_mat = "wood_16mm"     # warm mid-tone → wood
    elif fb > 0.7:
        floor_mat = "marble_floor"          # very bright → marble/tile
    else:
        floor_mat = "carpet_cotton"         # mid-brightness, cool → carpet

    # Walls — default plasterboard, override if very dark (concrete/brick)
    wb = _brightness(wall_region)
    wall_mat = "plasterboard" if wb > 0.3 else "brickwork"

    # Ceiling — almost always light
    ceil_mat = "ceiling_fibre_absorber"

    best = {"walls": wall_mat, "floor": floor_mat, "ceiling": ceil_mat}
    dists = {s: {m: 1.0} for s, m in best.items()}
    return best, dists


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------
def classify_materials(
    image_path: str,
    method: str = "segformer",
    floor_frac: float = 0.35,
    ceiling_frac: float = 0.15,
    device: str | None = None,
) -> dict:
    """
    Classify room surface materials from a rendered RGB image.

    Args:
        image_path:    Path to .png / .jpg room render.
        method:        "segformer" (default) or "heuristic".
        floor_frac:    Bottom fraction of image height treated as floor
                       (heuristic mode only).
        ceiling_frac:  Top fraction of image height treated as ceiling
                       (heuristic mode only).
        device:        "cuda" / "cpu" / None (auto-detect).

    Returns:
        {
            "walls":          str,   # best pyroomacoustics material name
            "floor":          str,
            "ceiling":        str,
            "distributions":  {      # per-surface confidence fractions
                "walls":   {"plasterboard": 0.85, "wood_panel": 0.15},
                "floor":   {"wood_16mm": 0.6, "carpet_cotton": 0.4},
                "ceiling": {"ceiling_fibre_absorber": 1.0},
            },
        }
    """
    img = np.array(Image.open(image_path).convert("RGB"))

    if method == "segformer":
        try:
            import torch
        except ImportError:
            warnings.warn("torch not installed, falling back to heuristic", stacklevel=2)
            method = "heuristic"

    if method == "segformer":
        import torch
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        seg_map = _run_segformer(img, device)
        best, dists = _distributions_from_segmap(seg_map)
    else:
        best, dists = _heuristic_classify(img, floor_frac, ceiling_frac)

    return {**best, "distributions": dists}


# -----------------------------------------------------------------------
# SAMOSA emulation utilities
# -----------------------------------------------------------------------

_mobilenet_cache: dict = {}


def _ensure_mobilenet_loaded(device: str) -> None:
    """Load and warm MobileNetV3 into the module cache if not already done."""
    import torch
    import torchvision.transforms as T
    from torchvision.models.segmentation import (
        deeplabv3_mobilenet_v3_large,
        DeepLabV3_MobileNet_V3_Large_Weights,
    )
    if _mobilenet_cache.get("device") == device:
        return
    model = deeplabv3_mobilenet_v3_large(
        weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    )
    model.to(device).eval()
    dummy = torch.zeros(1, 3, 520, 520, device=device)
    with torch.no_grad():
        model(dummy)
    _mobilenet_cache["model"] = model
    _mobilenet_cache["device"] = device
    _mobilenet_cache["transform"] = T.Compose([
        T.Resize((520, 520)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _mobilenet_forward(pil_img: Image.Image, device: str) -> float:
    """Preprocess pil_img and run one timed forward pass. Returns ms."""
    import time
    import torch
    _ensure_mobilenet_loaded(device)
    tensor = _mobilenet_cache["transform"](pil_img).unsqueeze(0).to(device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = _mobilenet_cache["model"](tensor)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


def run_mobilenet_timing(image_path: str, device: str) -> float:
    """
    Run DeepLabV3-MobileNetV3 and return inference-only ms.
    Image loading and preprocessing are NOT included in the returned time.
    """
    # Cache the preprocessed tensor by (path, device) so repeated local calls are fast.
    cache_key = (image_path, device)
    _ensure_mobilenet_loaded(device)
    if _mobilenet_cache.get("tensor_key") != cache_key:
        img = Image.open(image_path).convert("RGB")
        import torch
        tensor = _mobilenet_cache["transform"](img).unsqueeze(0).to(device)
        _mobilenet_cache["tensor"] = tensor
        _mobilenet_cache["tensor_key"] = cache_key

    import time, torch
    tensor = _mobilenet_cache["tensor"]
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = _mobilenet_cache["model"](tensor)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


def run_mobilenet_timing_from_pil(pil_img: Image.Image, device: str) -> float:
    """
    Same as run_mobilenet_timing but accepts an already-decoded PIL image.
    Preprocessing (resize + normalize + GPU transfer) IS included — use this
    on the server where the image arrives as bytes and disk I/O should be avoided.
    """
    return _mobilenet_forward(pil_img, device)


def save_segmentation_cache(result: dict, cache_path: str):
    """
    Save a classify_materials() result dict to JSON for later use in
    SAMOSA emulation mode.  Only strings and floats are stored (no numpy).
    """
    import json

    # distributions values are plain Python floats already; just serialise
    serialisable = {
        "walls":   result["walls"],
        "floor":   result["floor"],
        "ceiling": result["ceiling"],
        "distributions": result.get("distributions", {}),
    }
    with open(cache_path, "w") as f:
        json.dump(serialisable, f, indent=2)


def load_segmentation_cache(cache_path: str) -> dict:
    """
    Load a previously saved segmentation cache.
    Raises FileNotFoundError with a helpful message if the file is missing.
    """
    import json
    import os

    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Segmentation cache not found: {cache_path}\n"
            "Run with --init-segmentation first to generate it."
        )
    with open(cache_path) as f:
        return json.load(f)


if __name__ == "__main__":
    import os, sys

    img_path = os.path.join(os.path.dirname(__file__), "testing", "modernroom.png")
    method = sys.argv[1] if len(sys.argv) > 1 else "segformer"

    print(f"Classifying materials (method={method}) ...")
    result = classify_materials(img_path, method=method)

    print("\nBest materials:")
    for s in ("walls", "floor", "ceiling"):
        print(f"  {s:8s}: {result[s]}")

    print("\nDistributions:")
    for s, dist in result["distributions"].items():
        fracs = ", ".join(f"{m}: {v:.0%}" for m, v in dist.items())
        print(f"  {s:8s}: {fracs}")
