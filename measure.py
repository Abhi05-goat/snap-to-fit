"""
Item/space measurement, SAM-assisted.

    def measure_item(image_path: str, sheet_px: list, item_click: list) -> dict: ...
    def measure_space(image_path: str, sheet_px: list, kind: str, space_input: list) -> dict: ...

sheet_px: the 4 A4-sheet-corner taps, ORIGINAL-image pixel coords -- still
manual, NOT SAM. This is the one place precision matters most (every
downstream measurement is scaled by it) and where manual tapping was never
the actual problem: a human reads off 4 corners of a small flat rectangle
in a couple of clicks, accurately, every time. SAM was tried here too, but
its mask boundary on a small, low-contrast sheet lands a few pixels off the
true edge -- fine for a big object, but enough to visibly distort the
sheet's edge-length ratios and, through them, the scale factor everything
else depends on. Keep this one manual.

item_click: a single [x, y] click ON the item. SAM segments it and the
mask's pixel *extent* (a tight box around its true silhouette, not the mask
itself) replaces the hand-drawn box_px -- see measure_item's docstring for
why the mask's interior is used for localization but not for the actual
point cloud.

space_input depends on kind:
  kind="floor":   a single [x, y] click on the floor patch -- SAM segments
                  it and its corners become the 3 taps _extract_space_dims
                  expects.
  kind="opening": still the original 4-point manual taps
                  ([left_bottom, right_bottom, top, floor_below_top]). A
                  doorway opening is empty space, not a segmentable object
                  -- there's nothing for a point-prompted segmenter to
                  click "on".

Everything below requires a GPU + UniDepth installed (scripts/phase2.py) --
this is meant to run on your machine with real photos, not in a sandbox.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from sam_utils import mask_to_quad_px, segment_points
from scripts.phase2 import infer, load_model, point_at, taps_to_resized
from scripts.phase3 import calibrate

# sigma_pct floor -- Phase 8/10 in the plan retunes this from observed
# real error. 0.08 is the plan's placeholder; SIGMA_FLOOR in confidence.py
# (teammate's file) is the one that actually gets tuned in Phase 10.
BASE_SIGMA = 0.08

# Height above the fitted floor plane a point needs to clear to count as
# "the item" rather than floor/paper noise (Phase 4, step 2).
HEIGHT_THRESHOLD_CM = 3.0

# Foreground/background depth split (roadblock fix, see HANDOFF doc section 6).
# An open-frame item (wire shoe rack) lets the camera see the wall/door
# behind it through the gaps; those background pixels clear the height
# filter too (a wall is "above the floor"), inflating L/W/H. Below this
# many points on either side of a detected depth gap, treat it as noise
# rather than a real foreground/background boundary and keep everything.
MIN_CLUSTER_POINTS = 30


def _get_model():
    """Lazy singleton -- load once per process, not once per call."""
    global _MODEL
    try:
        return _MODEL
    except NameError:
        _MODEL = load_model()
        return _MODEL


def _load_rgb(image_path: str) -> np.ndarray:
    return np.array(Image.open(image_path).convert("RGB"))


def _find_depth_gap(
    values: np.ndarray,
    n_bins: int = 40,
    lookahead: int = 5,
    jump_ratio: float = 5.0,
) -> float | None:
    """Find where the histogram of `values` jumps to a much denser regime
    a bit further out -- the signature of a background cluster (a wall)
    starting up behind a foreground item -- and return the split point
    just before that jump, or None if there's no such jump.

    History of two rejected approaches, both caught by real data
    (tests/test_measure.py's far-background/solid-item cases plus a real
    shoerack photo -- see the roadblock in the handoff doc):
      1. Otsu/variance-maximizing split: its between-class variance ratio
         is high for ANY split near the middle of a plain continuous
         distribution (a uniform distribution split at its median gives
         0.75 by construction) -- it spuriously chopped a single solid
         item's own height range in half, no background involved.
      2. A literal empty (zero-count) bin: too strict for real, noisy
         depth data. On a real photo the item's own points tapered to a
         nonzero minimum before the wall's density spike (edge
         bleed/noise smears a few points into the gap), so no bin was
         ever exactly zero.
      3. A relative-valley threshold (bin count <= X% of the flanking
         peaks): comparing against the SMALLER flanking peak still failed
         when the item's own density gently tapers off before the wall
         spike -- the taper's tail is only ~5-6x below the item's own
         (much smaller) peak, not below whatever fixed fraction, even
         though the wall spike right after it is 15-20x higher again.

    What actually works: look `lookahead` bins ahead of each bin for the
    densest bin nearby, and flag the bin whose nearby-future density is
    `jump_ratio`x-or-more higher than its own count. A wall's sudden
    spike clears this by a wide margin (~20x on the real shoerack photo)
    while an item's own internal density variation, or noise in a flat
    distribution, does not (~1-4x) -- this is what actually distinguishes
    "about to hit an unrelated denser surface" from "still tapering off
    within the same object".
    """
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return None

    hist, edges = np.histogram(values, bins=n_bins, range=(lo, hi))
    n = len(hist)

    best_ratio = jump_ratio
    best_i = None
    for i in range(n - 1):
        window = hist[i + 1 : i + 1 + lookahead]
        if len(window) == 0:
            break
        ratio = float(window.max()) / max(float(hist[i]), 1.0)
        if ratio > best_ratio:
            best_ratio = ratio
            best_i = i

    if best_i is None:
        return None
    return float(edges[best_i + 1])


def _drop_far_background(kept: np.ndarray, h_kept: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split candidate item points into a near cluster and a far cluster by
    distance from the camera (camera is at the origin in this frame -- see
    Calib/plane_n orientation in scripts/phase3.py), and drop the far
    cluster if a real depth gap separates them.

    Why distance-from-camera and not a naive median filter: an open-frame
    item (wire shoe rack) has far more background pixels visible through
    its gaps than item pixels, so the median of ALL candidate points is
    background, not the item. Gap detection doesn't care which side has
    more points -- it finds the empty space itself.
    """
    camera_dist = np.linalg.norm(kept, axis=1)
    threshold = _find_depth_gap(camera_dist)
    if threshold is None:
        return kept, h_kept
    near = camera_dist <= threshold
    far = ~near
    if near.sum() >= MIN_CLUSTER_POINTS and far.sum() >= MIN_CLUSTER_POINTS:
        return kept[near], h_kept[near]
    return kept, h_kept


def _extract_item_dims(
    box_points_cm: np.ndarray,
    plane_p_cm: np.ndarray,
    plane_n: np.ndarray,
    height_threshold_cm: float = HEIGHT_THRESHOLD_CM,
) -> tuple[float, float, float, int]:
    """Pure geometry, Phase 4 steps 2-4: box points (already in cm, already
    scaled) -> (L_cm, W_cm, H_cm, n_points_kept). Raises ValueError if too
    few points survive the height filter. Separated out from measure_item so
    it's testable with synthetic point clouds, no GPU/model required.
    """
    # 2. Height above the plane; keep points with h > threshold (drops floor + paper).
    h = (box_points_cm - plane_p_cm) @ plane_n
    kept = box_points_cm[h > height_threshold_cm]
    h_kept = h[h > height_threshold_cm]

    if len(kept) < 10:
        raise ValueError("no points found above the floor plane inside the box")

    # 2b. Drop a far background cluster visible through an open-frame item
    # (see the roadblock in the handoff doc: a wire rack's gaps let the
    # camera see the wall/door behind it, and those points clear the height
    # filter too since a wall is "above the floor").
    kept, h_kept = _drop_far_background(kept, h_kept)

    # 3. Project onto the plane, PCA on the 2D footprint for the two axes.
    #    Build an orthonormal (u, v) basis spanning the plane.
    arbitrary = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(arbitrary, plane_n)) > 0.9:
        arbitrary = np.array([0.0, 1.0, 0.0])
    u = np.cross(plane_n, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(plane_n, u)

    centered = kept - plane_p_cm
    footprint_2d = np.stack([centered @ u, centered @ v], axis=1)  # (N, 2)

    cov = np.cov(footprint_2d, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]  # descending variance
    axes = eigvecs[:, order]

    projected = footprint_2d @ axes  # (N, 2), columns = principal axes

    # 4. L, W = 2nd-98th percentile spread along each axis (sorted L >= W).
    #    H = 98th percentile of h.
    spreads = []
    for col in range(2):
        lo, hi = np.percentile(projected[:, col], [2, 98])
        spreads.append(float(hi - lo))
    L_cm, W_cm = sorted(spreads, reverse=True)
    H_cm = float(np.percentile(h_kept, 98))

    return L_cm, W_cm, H_cm, len(kept)


def measure_item(image_path: str, sheet_px: list, item_click: list) -> dict:
    """Item L/W/H from a single click on the item + the A4 calibration.

    item_click: [x, y], ORIGINAL-image pixel coords, turned into a SAM mask
    (see module docstring). The mask's pixel *extent* (a tight box around
    its true silhouette, not the mask itself) is what replaces the
    hand-drawn box_px: for a solid item this is already a big precision win
    over eyeballing a box by hand, and it feeds the exact same box-crop +
    depth-gap background filter as before. (Using the mask's interior
    directly, instead of just its extent, was tried and made things worse
    for open/mesh items like a wire shoe rack: SAM fills in the
    visually-enclosed shelf-plane area, gaps and all, and the gaps are
    exactly where the wall/floor behind the item shows through -- turning a
    well-localized click into a mask that's *more* background-contaminated
    than a plain box over the same footprint.)
    """
    try:
        model = _get_model()
        depth_result = infer(model, image_path)
        calib = calibrate(image_path, sheet_px, depth_result=depth_result)

        image_rgb = _load_rgb(image_path)
        item_mask = segment_points(image_rgb, [item_click])[0]

        # 1. Raw points inside the item mask's tight pixel extent x scale.
        ys, xs = np.where(item_mask)
        if len(xs) == 0:
            return {"ok": False, "error": "SAM mask is empty -- the click didn't land on a segmentable region"}
        x1, y1 = taps_to_resized([[int(xs.min()), int(ys.min())]], depth_result.scale_factor)[0]
        x2, y2 = taps_to_resized([[int(xs.max()), int(ys.max())]], depth_result.scale_factor)[0]
        xlo, xhi = sorted((x1, x2))
        ylo, yhi = sorted((y1, y2))
        xhi = min(xhi, depth_result.points.shape[1])
        yhi = min(yhi, depth_result.points.shape[0])

        box_points_raw = depth_result.points[ylo:yhi, xlo:xhi].reshape(-1, 3)
        box_points_raw = box_points_raw[np.isfinite(box_points_raw).all(axis=1)]
        if len(box_points_raw) == 0:
            return {"ok": False, "error": "no valid depth points inside the segmented item box"}

        box_points_cm = box_points_raw * calib.scale * 100.0
        plane_p_cm = calib.plane_p * 100.0

        L_cm, W_cm, H_cm, n_kept = _extract_item_dims(box_points_cm, plane_p_cm, calib.plane_n)

        sigma_pct = float(np.sqrt(BASE_SIGMA**2 + calib.scale_cv**2))

        # 6. Debug image -- best-effort, never fails the measurement itself.
        debug_img = None
        try:
            debug_img = _save_debug_overlay_item(image_path, item_click, item_mask, sheet_px, kept_count=n_kept)
        except Exception:
            pass

        return {
            "ok": True,
            "L_cm": round(L_cm, 1),
            "W_cm": round(W_cm, 1),
            "H_cm": round(H_cm, 1),
            "sigma_pct": round(sigma_pct, 3),
            "debug_img": debug_img,
        }

    except Exception as e:  # 7. never crash -- always ok:false on failure
        return {"ok": False, "error": str(e)}


def _extract_space_dims(kind: str, tap_points_cm: list) -> tuple[float, float | None, float | None]:
    """Pure geometry, Phase 5: tap points (already in cm, already scaled) ->
    (width_cm, depth_cm, height_cm) with the unused one as None, per the
    contract. Separated out from measure_space so it's testable without a
    GPU/model.
    """
    if kind == "floor":
        c1, c2, c3 = tap_points_cm
        width_cm = float(np.linalg.norm(c1 - c2))
        depth_cm = float(np.linalg.norm(c2 - c3))
        return width_cm, depth_cm, None
    else:  # opening
        left_bottom, right_bottom, top, floor_below_top = tap_points_cm
        width_cm = float(np.linalg.norm(left_bottom - right_bottom))
        height_cm = float(np.linalg.norm(top - floor_below_top))
        return width_cm, None, height_cm


def measure_space(image_path: str, sheet_px: list, kind: str, space_input: list) -> dict:
    """Floor width/depth, or opening width/height, from the A4 calibration
    plus either a single SAM-segmented click (floor) or manual taps (opening).

    kind="floor":   space_input = [x, y], a single click on the floor patch.
                    SAM's mask -> corners [c1, c2, c3] (its own contract:
                    width=|c1-c2|, depth=|c2-c3|).
    kind="opening": space_input = [left_bottom, right_bottom, top, floor_below_top],
                    the original 4 manual taps -- an opening is empty space,
                    not something a point click can segment (see module
                    docstring). width=|left_bottom-right_bottom|, height=|top-floor_below_top|.
    """
    try:
        if kind not in ("floor", "opening"):
            return {"ok": False, "error": f"unknown kind {kind!r}, expected 'floor' or 'opening'"}

        model = _get_model()
        depth_result = infer(model, image_path)
        calib = calibrate(image_path, sheet_px, depth_result=depth_result)

        if kind == "floor":
            image_rgb = _load_rgb(image_path)
            floor_mask = segment_points(image_rgb, [space_input])[0]
            quad = mask_to_quad_px(floor_mask)  # 4 corners, valid edge-walk order
            taps_px = [quad[0].tolist(), quad[1].tolist(), quad[2].tolist()]
        else:  # opening -- space_input is already the 4 manual taps
            taps_px = space_input

        resized_taps = taps_to_resized(taps_px, depth_result.scale_factor)
        tap_points_raw = [point_at(depth_result.points, x, y) for x, y in resized_taps]
        tap_points_cm = [p * calib.scale * 100.0 for p in tap_points_raw]

        width_cm, depth_cm, height_cm = _extract_space_dims(kind, tap_points_cm)
        sigma_pct = float(np.sqrt(BASE_SIGMA**2 + calib.scale_cv**2))

        debug_img = None
        try:
            debug_img = _save_debug_overlay_space(image_path, taps_px)
        except Exception:
            pass

        return {
            "ok": True,
            "kind": kind,
            "width_cm": round(width_cm, 1),
            "depth_cm": round(depth_cm, 1) if depth_cm is not None else None,
            "height_cm": round(height_cm, 1) if height_cm is not None else None,
            "sigma_pct": round(sigma_pct, 3),
            "debug_img": debug_img,
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Debug overlays -- best-effort visual sanity checks, saved next to the
# source photo. Cut-list item #1 if you're behind schedule: these can be
# deleted/stubbed without affecting measure_item/measure_space's return
# values, since both wrap the call in try/except and fall back to None.
# ---------------------------------------------------------------------------

def _debug_out_path(image_path: str, suffix: str) -> str:
    from pathlib import Path
    p = Path(image_path)
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)
    return str(out_dir / f"{p.stem}_{suffix}.png")


def _save_debug_overlay_item(
    image_path: str, item_click: list, item_mask: np.ndarray, sheet_px: list, kept_count: int
) -> str:
    import matplotlib.pyplot as plt
    from PIL import Image

    img = Image.open(image_path)
    fig, ax = plt.subplots(figsize=(8, 8 * img.height / img.width))
    ax.imshow(img)
    ax.imshow(np.where(item_mask, 1.0, np.nan), alpha=0.4, cmap="autumn")
    ax.plot(*item_click, "b*", markersize=14, label="item click")
    sx = [p[0] for p in sheet_px] + [sheet_px[0][0]]
    sy = [p[1] for p in sheet_px] + [sheet_px[0][1]]
    ax.plot(sx, sy, "o-", color="lime", linewidth=2, markersize=6, label="sheet taps")
    ax.set_title(f"item mask ({kept_count} pts kept above floor)")
    ax.legend(loc="upper right", fontsize=8)
    ax.axis("off")
    out_path = _debug_out_path(image_path, "debug")
    fig.savefig(out_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    return out_path


def _save_debug_overlay_space(image_path: str, taps_px: list) -> str:
    import matplotlib.pyplot as plt
    from PIL import Image

    img = Image.open(image_path)
    fig, ax = plt.subplots(figsize=(8, 8 * img.height / img.width))
    ax.imshow(img)
    xs = [p[0] for p in taps_px]
    ys = [p[1] for p in taps_px]
    ax.plot(xs, ys, "o-", color="lime", linewidth=2, markersize=8, label="space taps")
    ax.set_title("space taps")
    ax.legend(loc="upper right", fontsize=8)
    ax.axis("off")
    out_path = _debug_out_path(image_path, "debug")
    fig.savefig(out_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    return out_path
