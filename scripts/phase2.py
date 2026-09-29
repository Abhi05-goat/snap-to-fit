"""
Phase 2 - Depth inference (T+1:00 - 1:40)

Thin wrapper around UniDepthV2: load the model once, run metric-depth
inference on a photo, downscale first (matches the paper's training
resolution range), and cache the raw 3D points to disk so repeated calls
during calibration/measurement debugging are instant.

Requires a GPU machine with UniDepth installed (see requirements.txt).
Nothing in this file runs without `unidepth` + `torch` importable, so it's
meant to be run on your machine, not in a sandbox without a GPU.

    from scripts.phase2 import load_model, infer

    model = load_model()                       # once, at app/script startup
    result = infer(model, "data/sofa_01.jpg")   # cached after first call
    points = result["points"]                   # (H, W, 3) metres, camera frame
"""

from __future__ import annotations

import os

# Windows fix: numpy and torch each bundle their own Intel OpenMP runtime;
# loading both in one process can crash LAPACK calls (np.linalg.svd in
# phase3.py) with a garbled "init_gesdd failed init" + bogus MemoryError.
# Must be set before numpy/torch are imported anywhere -- this file is the
# common entry point for both, so it's set here first.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "data" / ".depth_cache"

# Target megapixels to downscale to before inference. The plan calls for
# ~0.3-0.6 MP (matches UniDepth's training resolution range) -- 0.5 is the
# midpoint and a safe default; drop to 0.3 for the ViT-S fallback if VRAM
# or speed is tight (see the plan's fallback ladder, item 3).
DEFAULT_TARGET_MP = 0.5

DEFAULT_CHECKPOINT = "lpiccinelli/unidepth-v2-vits14"  # ViT-S; swap to vitb14 if you have headroom


@dataclass
class DepthResult:
    points: np.ndarray          # (H, W, 3) float32, metres, camera frame, at RESIZED resolution
    depth: np.ndarray           # (H, W) float32, metres
    original_size: tuple[int, int]   # (W, H) of the source photo
    resized_size: tuple[int, int]    # (W, H) actually fed to the model
    scale_factor: float         # resized / original (apply to tap pixel coords)

    def to_dict(self) -> dict:
        return {
            "points": self.points,
            "depth": self.depth,
            "original_size": self.original_size,
            "resized_size": self.resized_size,
            "scale_factor": self.scale_factor,
        }


_MODEL_CACHE = {}


def load_model(checkpoint: str = DEFAULT_CHECKPOINT, device: str | None = None):
    """Load a UniDepthV2 checkpoint once. Safe to call repeatedly -- returns
    the cached instance after the first call (per checkpoint+device)."""
    import torch
    from unidepth.models import UniDepthV2

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    key = (checkpoint, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    t0 = time.time()
    model = UniDepthV2.from_pretrained(checkpoint)
    model = model.to(device).eval()
    print(f"[phase2] loaded {checkpoint} on {device} in {time.time() - t0:.1f}s")

    _MODEL_CACHE[key] = model
    return model


def _resize_for_inference(image: Image.Image, target_mp: float) -> tuple[Image.Image, float]:
    """Downscale to ~target_mp megapixels, preserving aspect ratio. Returns
    the resized image and the scale_factor (resized_dim / original_dim) to
    apply to tap pixel coordinates."""
    w, h = image.size
    current_mp = (w * h) / 1_000_000
    if current_mp <= target_mp:
        return image, 1.0
    scale_factor = (target_mp / current_mp) ** 0.5
    new_w, new_h = max(1, round(w * scale_factor)), max(1, round(h * scale_factor))
    resized = image.resize((new_w, new_h), Image.LANCZOS)
    return resized, scale_factor


def _cache_key(image_path: Path, checkpoint: str, target_mp: float) -> Path:
    stat = image_path.stat()
    raw = f"{image_path.resolve()}|{stat.st_mtime}|{stat.st_size}|{checkpoint}|{target_mp}"
    digest = hashlib.sha1(raw.encode()).hexdigest()[:16]
    return CACHE_DIR / f"{image_path.stem}_{digest}.npz"


def infer(
    model,
    image_path: str,
    checkpoint: str = DEFAULT_CHECKPOINT,
    target_mp: float = DEFAULT_TARGET_MP,
    force: bool = False,
) -> DepthResult:
    """Run (or load cached) UniDepthV2 inference on one photo.

    `checkpoint` is only used to key the cache (so switching model size
    invalidates it) -- the model itself is whatever you passed as `model`.
    """
    import torch

    image_path = Path(image_path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_key(image_path, checkpoint, target_mp)

    if cache_path.exists() and not force:
        data = np.load(cache_path)
        return DepthResult(
            points=data["points"],
            depth=data["depth"],
            original_size=tuple(data["original_size"]),
            resized_size=tuple(data["resized_size"]),
            scale_factor=float(data["scale_factor"]),
        )

    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (W, H)
    resized, scale_factor = _resize_for_inference(image, target_mp)

    rgb = torch.from_numpy(np.array(resized)).permute(2, 0, 1)  # (3, H, W)
    device = next(model.parameters()).device
    rgb = rgb.to(device)

    t0 = time.time()
    with torch.no_grad():
        predictions = model.infer(rgb)
    elapsed = time.time() - t0

    points = predictions["points"].squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)  # (H, W, 3)
    depth = predictions["depth"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)            # (H, W)

    result = DepthResult(
        points=points,
        depth=depth,
        original_size=original_size,
        resized_size=resized.size,
        scale_factor=scale_factor,
    )

    np.savez_compressed(
        cache_path,
        points=result.points,
        depth=result.depth,
        original_size=np.array(result.original_size),
        resized_size=np.array(result.resized_size),
        scale_factor=np.array(result.scale_factor),
    )

    print(f"[phase2] {image_path.name}: {original_size} -> {resized.size} "
          f"({elapsed:.2f}s inference, cached to {cache_path.name})")
    return result


def taps_to_resized(taps_px: list, scale_factor: float) -> list[tuple[int, int]]:
    """Map tap coordinates given in ORIGINAL-photo pixels (per the measure.py
    contract) to indices into the resized `points`/`depth` arrays."""
    return [(round(x * scale_factor), round(y * scale_factor)) for x, y in taps_px]


def point_at(points: np.ndarray, x: int, y: int, window: int = 5) -> np.ndarray:
    """Median 3D point in a `window`x`window` box around (x, y) in the
    RESIZED image -- robust to single-pixel depth noise. Used for every tap
    (sheet corners, box corners, space taps) in Phases 3-5."""
    h, w, _ = points.shape
    half = window // 2
    x0, x1 = max(0, x - half), min(w, x + half + 1)
    y0, y1 = max(0, y - half), min(h, y + half + 1)
    patch = points[y0:y1, x0:x1].reshape(-1, 3)
    valid = np.isfinite(patch).all(axis=1) & (patch[:, 2] > 0)
    if not valid.any():
        raise ValueError(f"no valid depth points in the {window}x{window} window around ({x}, {y})")
    return np.median(patch[valid], axis=0)


# ---------------------------------------------------------------------------
# Sanity check (Phase 2's "pick two pixels with a known real distance and
# compare" step). Run this once per photo after your first successful infer,
# before trusting the pipeline further.
#
#   python scripts/phase2.py data/sofa_01.jpg 100,200 400,200 63.0
#
# (two pixel taps in ORIGINAL-image coords, then the true distance in cm
# between whatever those two pixels are pointing at -- measure it with your
# tape measure)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 5:
        print(__doc__)
        print("Sanity check usage:")
        print("  python scripts/phase2.py <image_path> <x1,y1> <x2,y2> <true_distance_cm>")
        sys.exit(1)

    img_path, p1_str, p2_str, true_cm_str = sys.argv[1:5]
    p1 = tuple(int(v) for v in p1_str.split(","))
    p2 = tuple(int(v) for v in p2_str.split(","))
    true_cm = float(true_cm_str)

    m = load_model()
    r = infer(m, img_path)
    rp1 = taps_to_resized([p1], r.scale_factor)[0]
    rp2 = taps_to_resized([p2], r.scale_factor)[0]
    a = point_at(r.points, *rp1)
    b = point_at(r.points, *rp2)
    measured_cm = float(np.linalg.norm(a - b) * 100)
    err_pct = 100 * abs(measured_cm - true_cm) / true_cm

    print(f"tapped points (original px): {p1} -> {p2}")
    print(f"raw model distance:          {measured_cm:.1f} cm")
    print(f"true distance:                {true_cm:.1f} cm")
    print(f"raw error (pre-A4-calibration): {err_pct:.1f}%")
    print("(this is the UNCALIBRATED distance -- Phase 3's scale factor corrects this)")