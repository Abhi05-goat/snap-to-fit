"""SAM (MobileSAM) point-prompt segmentation.

Turns a single click into a precise pixel mask, replacing hand-tapped boxes/
polygons for anything that's a solid, visually-distinct surface in the
photo: the item being measured, and a floor patch. (The A4 calibration
sheet stays manual taps -- SAM's mask boundary isn't precise enough on
something that small and low-contrast; see measure.py's module docstring.
A doorway *opening* is also manual taps -- it's empty space, not a
segmentable object.)

    from sam_utils import segment_point, mask_to_quad_px

    image_rgb = np.array(Image.open(path).convert("RGB"))
    mask = segment_point(image_rgb, (x, y))       # bool (H, W), full image res
    corners = mask_to_quad_px(mask)                 # 4 (x, y) corners, valid edge-walk order
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "mobile_sam.pt"

_PREDICTOR_CACHE = {}


def load_predictor(device: str | None = None):
    """Lazy singleton -- load the MobileSAM checkpoint once per process."""
    import torch
    from mobile_sam import SamPredictor, sam_model_registry

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device in _PREDICTOR_CACHE:
        return _PREDICTOR_CACHE[device]

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"MobileSAM checkpoint not found at {CHECKPOINT_PATH} -- "
            "download it from https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
        )

    sam = sam_model_registry["vit_t"](checkpoint=str(CHECKPOINT_PATH))
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)
    _PREDICTOR_CACHE[device] = predictor
    return predictor


def segment_points(
    image_rgb: np.ndarray, points_xy: list[tuple[float, float]], predictor=None
) -> list[np.ndarray]:
    """Run SAM with one foreground-point prompt per entry in `points_xy`,
    against a single shared image embedding (computed once, since that's the
    expensive part -- calling this with all of a request's clicks at once is
    much cheaper than segmenting one point at a time on the same image).
    `points_xy` are in the image's own pixel space. Returns one highest-
    scoring boolean mask (H, W) per point, same order as `points_xy`.
    """
    predictor = predictor or load_predictor()
    predictor.set_image(image_rgb)
    masks_out = []
    for point_xy in points_xy:
        point_coords = np.array([[point_xy[0], point_xy[1]]], dtype=np.float32)
        point_labels = np.array([1])
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        best = int(np.argmax(scores))
        masks_out.append(masks[best].astype(bool))
    return masks_out


def segment_point(image_rgb: np.ndarray, point_xy: tuple[float, float], predictor=None) -> np.ndarray:
    """Single-point convenience wrapper around segment_points, for one-off/
    debug use. Prefer segment_points when segmenting >1 point on the same
    image (e.g. measure_item's sheet + item clicks) -- it shares the (slow)
    image embedding across points instead of recomputing it per call.
    """
    return segment_points(image_rgb, [point_xy], predictor=predictor)[0]


def largest_contour(mask: np.ndarray):
    import cv2

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("SAM mask is empty -- the click didn't land on a segmentable region")
    return max(contours, key=cv2.contourArea)


def mask_to_quad_px(mask: np.ndarray) -> np.ndarray:
    """Largest contour in `mask` -> its 4 actual corners, in a valid
    walk-the-quadrilateral order (consecutive points are real edges, not
    diagonals -- that's all downstream code needs, not any particular
    starting corner or winding direction).

    For a planar rectangular surface in the photo (the A4 sheet, a floor
    patch) under perspective, this is a general skewed quadrilateral, not an
    upright/rotated rectangle -- cv2.minAreaRect's fitted-rectangle corners
    don't line up with a trapezoid's own corners, which was quietly wrong
    for anything with real perspective skew (a floor patch spanning a
    couple of metres; less visible on the small, near-frontal A4 sheet).

    Deliberately NOT reordered by a "top-left/top-right/..." sum/diff
    heuristic: that trick assumes a roughly axis-aligned rectangle, and
    silently produces an INVALID order (connecting diagonal corners as if
    they were adjacent) for a sheet or floor patch photographed at a steep
    angle, where it appears heavily rotated/diamond-shaped in frame -- caught
    on a real photo where it turned calibrate()'s edge lengths into
    diagonals, collapsing the scale fit entirely. The convex hull's own
    walk order is already valid; approxPolyDP just needs to land on exactly
    4 points (searching increasing simplification tolerances since a single
    fixed epsilon isn't always enough to collapse hull noise to 4 corners).
    """
    import cv2

    contour = largest_contour(mask)
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    for frac in (0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25):
        approx = cv2.approxPolyDP(hull, frac * peri, True).reshape(-1, 2)
        if len(approx) == 4:
            return approx.astype(np.float64)
    raise ValueError(
        f"could not simplify the SAM mask's outline to a clean 4-cornered "
        f"quadrilateral (got {len(approx)} corners) -- try clicking closer "
        "to the middle of the floor patch"
    )
