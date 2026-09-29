"""
Phase 3 - Calibration: scale and floor plane (T+1:40 - 2:40)

Turns the 4 tapped A4-sheet corners into:
  1. a scale factor (uncalibrated model units -> real centimetres)
  2. a floor plane fit (so Phase 4/5 can tell "above the floor" from "on it")

Everything downstream (measure_item, measure_space) depends on this being
right -- the plan is explicit: "fix this before moving on." Run
`python scripts/phase3.py <image_path> "x1,y1 x2,y2 x3,y3 x4,y4"` on a real
photo once you have UniDepth working, and check the printed scale_cv and
edge lengths before writing a single line of Phase 4/5 code.
"""

from __future__ import annotations

import os

# Windows: NumPy and PyTorch each bundle their own copy of the Intel OpenMP
# runtime (libiomp5md.dll); when both load in the same process this can
# throw duplicate-library errors. Harmless to set defensively even though
# it turned out not to be the cause of the MemoryError below (see
# _fit_plane's docstring for the actual bug and fix).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dataclasses import dataclass

import numpy as np

# Works both when this file is run directly (`python scripts/phase3.py`,
# where only scripts/ itself is on sys.path) and when it's imported as a
# package (`from scripts.phase3 import ...`, from measure.py or the test
# suite at the repo root, where scripts.phase2 is the correct path).
try:
    from scripts.phase2 import point_at, taps_to_resized
except ImportError:
    from phase2 import point_at, taps_to_resized

# Real A4 dimensions, cm. Long edge / short edge.
A4_LONG_CM = 29.7
A4_SHORT_CM = 21.0
# 3.5% tolerance for hand-tapped sheet corners. Real A4 measurement by eye is ±3-4%;
# 3.0% was overly strict and hit on real photos due to sub-pixel tap precision.
EDGE_TOLERANCE_PCT = 3.5


@dataclass
class Calib:
    scale: float          # multiply raw model-unit distances by this to get metres
    plane_n: np.ndarray   # (3,) unit normal, oriented toward the camera
    plane_p: np.ndarray   # (3,) a point on the plane (its centroid)
    scale_cv: float       # coefficient of variation of the 4 edge-length ratios;
                           # a quality signal -- high scale_cv means the sheet
                           # taps or its depth are noisy, distrust this photo
    sheet_edges_cm: np.ndarray  # (4,) the sheet's 4 edges after scaling, cm --
                                  # should land within EDGE_TOLERANCE_PCT of
                                  # [29.7, 21.0, 29.7, 21.0]. Diagnostic only.


def _edge_lengths(corners: np.ndarray) -> np.ndarray:
    """4 corners (already ordered clockwise from top-left) -> 4 edge lengths,
    walking the quadrilateral: 0-1, 1-2, 2-3, 3-0."""
    return np.array([np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)])


def compute_scale(sheet_points_raw: np.ndarray) -> tuple[float, float]:
    """sheet_points_raw: (4, 3) the 4 sheet corners' 3D points, in the model's
    raw (uncalibrated) units. Returns (scale, scale_cv).

    A4 alternates long/short/long/short edges going around any rectangle,
    but WHICH pair (edges 0&2, or edges 1&3) is the long pair depends on
    how the sheet happens to be rotated relative to whichever corner you
    tapped first -- there's no fixed "top edge is always long" rule, that
    was a bad assumption in an earlier version of this function (it hardcoded
    top=long/right=short, which only holds for one specific paper rotation
    and broke -- with a very distinctive symptom: edges alternating between
    ~25% short and ~50% long of their assigned target -- on a real photo
    where the sheet's short edge happened to be first in tap order).

    Fix: don't assume which pair is which. A rectangle's OPPOSITE edges
    (0&2, 1&3) are equal by construction, so compare the two pairs' average
    measured length and assign the larger pair to A4_LONG_CM, the smaller
    to A4_SHORT_CM, whichever tap order you happened to use.
    """
    measured = _edge_lengths(sheet_points_raw)  # model units, order: edge0-1, 1-2, 2-3, 3-0
    pair_02 = (measured[0] + measured[2]) / 2  # opposite edges, should be ~equal
    pair_13 = (measured[1] + measured[3]) / 2

    if pair_02 >= pair_13:
        true_cm = np.array([A4_LONG_CM, A4_SHORT_CM, A4_LONG_CM, A4_SHORT_CM])
    else:
        true_cm = np.array([A4_SHORT_CM, A4_LONG_CM, A4_SHORT_CM, A4_LONG_CM])
    true_m = true_cm / 100.0

    ratios = true_m / measured  # model-unit -> metres, per edge
    scale = float(np.mean(ratios))
    scale_cv = float(np.std(ratios) / np.mean(ratios))
    return scale, scale_cv


def _fit_plane(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares plane through points `p`: (normal, centroid).

    Deliberately NOT `np.linalg.svd(p - centroid)` on the full (N, 3) array.
    A real photo's floor region can be tens of thousands of points, and
    there's a known NumPy/OpenBLAS bug on Windows where SVD's `gesdd`
    driver overflows its own workspace-size calculation for very tall,
    skinny matrices (huge N, 3 columns) -- it throws a nonsensical
    MemoryError on an array that's a few hundred KB, nowhere near actually
    running out of memory. Synthetic tests never caught this because they
    used at most ~2000 points; a real photo blew past whatever the
    threshold is.

    The fix is also just a better approach: the plane normal is the
    eigenvector of the smallest eigenvalue of the 3x3 covariance matrix
    p_centered.T @ p_centered -- mathematically identical to SVD's
    smallest right-singular-vector, but eigh() here only ever operates on
    a fixed 3x3 matrix no matter how large N is. Sidesteps the buggy
    LAPACK path entirely and is faster besides.
    """
    centroid = p.mean(axis=0)
    centered = p - centroid
    cov = centered.T @ centered  # 3x3, size independent of N
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalue order
    normal = eigvecs[:, 0]  # smallest eigenvalue's eigenvector = plane normal
    return normal, centroid


def fit_floor_plane(
    points_in_quad: np.ndarray,
    scale: float,
    camera_pos: np.ndarray = None,
    ransac_iters: int = 200,
    inlier_tolerance_m: float = 0.02,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """points_in_quad: (N, 3) raw points inside the tapped sheet quadrilateral
    (i.e. floor points -- the sheet sits flat on the floor). Scales them,
    then fits the plane in two stages and orients the normal toward the
    camera (origin, since points are in the camera frame).

    The plan's original spec was "SVD, trim the worst 10% by residual,
    refit." That works when the outliers are mild, but a single un-trimmed
    SVD pass is itself a least-squares fit -- a handful of points that
    leak in from well above the floor (e.g. the item's own edge, or noise)
    can dominate the variance and tilt that FIRST fit before any trimming
    happens, so residual-to-a-bad-plane ends up trimming the wrong 10%
    (this actually broke on a synthetic 5%-outlier test case while writing
    tests/test_measure.py -- see test_fit_floor_plane_trims_outliers).

    So this does a quick RANSAC pass first (fit a plane from 3 random
    points, count inliers within a fixed physical tolerance, repeat, keep
    the best-supported plane) to get a robust *inlier set*, then does one
    ordinary least-squares SVD refit on just those inliers -- same "trim,
    then refit" idea as the plan, just with a trim step that isn't itself
    thrown off by the outliers it's supposed to be removing.

    Note on the threshold: it has to be a FIXED tolerance (`inlier_tolerance_m`,
    default 2cm -- a reasonable "close enough to call it the floor" for a
    monocular depth estimate), not one recomputed per-candidate from that
    candidate's own residuals. An earlier version did the latter and it was
    silently broken: a badly-tilted candidate plane can inflate its own
    residual spread enough that its self-derived threshold swallows nearly
    every point, "winning" the inlier count by cheating rather than by
    fitting well. Caught by test_fit_floor_plane_trims_outliers.

    Returns (plane_n, plane_p).
    """
    pts = points_in_quad * scale
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 10:
        raise ValueError(f"only {len(pts)} valid floor points inside the sheet quad -- too few to fit a plane")

    rng = np.random.default_rng(seed)
    best_inliers = None
    best_count = -1
    for _ in range(min(ransac_iters, len(pts) * (len(pts) - 1))):
        idx = rng.choice(len(pts), 3, replace=False)
        a, b, c = pts[idx]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:  # degenerate (near-collinear) sample, skip
            continue
        n = n / norm
        residuals = np.abs((pts - a) @ n)
        inliers = residuals <= inlier_tolerance_m
        if inliers.sum() > best_count:
            best_count = inliers.sum()
            best_inliers = inliers

    if best_inliers is None or best_count < 10:
        # RANSAC found nothing usable (degenerate geometry) -- fall back to
        # the plan's original one-shot SVD + 10%-trim as a last resort.
        normal, centroid = _fit_plane(pts)
        residuals = np.abs((pts - centroid) @ normal)
        keep = residuals <= np.percentile(residuals, 90)
        normal, centroid = _fit_plane(pts[keep])
    else:
        normal, centroid = _fit_plane(pts[best_inliers])

    # Orient toward the camera: camera is at the origin in camera-frame
    # coordinates (or at camera_pos if given), so the normal should point
    # from the plane back toward the camera.
    cam = camera_pos if camera_pos is not None else np.zeros(3)
    if np.dot(normal, cam - centroid) < 0:
        normal = -normal

    return normal, centroid


def calibrate(image_path: str, sheet_px: list, depth_result=None, model=None) -> Calib:
    """End-to-end: given the 4 sheet-corner taps (original-image pixel
    coords, per the contract), run/reuse inference, compute scale, and fit
    the floor plane. Raises on failure -- callers (measure_item/measure_space)
    should catch and turn into {"ok": false, "error": ...}.
    """
    try:
        from scripts.phase2 import infer, load_model
    except ImportError:
        from phase2 import infer, load_model

    if depth_result is None:
        model = model or load_model()
        depth_result = infer(model, image_path)

    resized_taps = taps_to_resized(sheet_px, depth_result.scale_factor)
    corners_raw = np.array([point_at(depth_result.points, x, y) for x, y in resized_taps])

    scale, scale_cv = compute_scale(corners_raw)

    # Points "inside the tapped quadrilateral" for the plane fit: crop to the
    # taps' bounding box in the resized image and let the plane fit's own
    # 10%-trim handle anything that leaks in from outside the quad edges.
    xs = [p[0] for p in resized_taps]
    ys = [p[1] for p in resized_taps]
    x0, x1 = max(0, min(xs)), min(depth_result.points.shape[1], max(xs))
    y0, y1 = max(0, min(ys)), min(depth_result.points.shape[0], max(ys))
    region = depth_result.points[y0:y1, x0:x1].reshape(-1, 3)

    plane_n, plane_p = fit_floor_plane(region, scale)
    sheet_edges_cm = _edge_lengths(corners_raw) * scale * 100.0

    return Calib(
        scale=scale,
        plane_n=plane_n,
        plane_p=plane_p,
        scale_cv=scale_cv,
        sheet_edges_cm=sheet_edges_cm,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python scripts/phase3.py <image_path> \"x1,y1 x2,y2 x3,y3 x4,y4\"")
        print("(4 sheet-corner taps, clockwise from top-left, original-image pixel coords)")
        sys.exit(1)

    image_path, taps_str = sys.argv[1], sys.argv[2]
    sheet_px = [[int(v) for v in pair.split(",")] for pair in taps_str.split()]
    if len(sheet_px) != 4:
        print(f"expected 4 tapped corners, got {len(sheet_px)}")
        sys.exit(1)

    calib = calibrate(image_path, sheet_px)

    print(f"scale:      {calib.scale:.6f}")
    print(f"scale_cv:   {calib.scale_cv:.4f}  (quality signal -- lower is better)")
    print(f"plane_n:    {calib.plane_n}")
    print(f"plane_p:    {calib.plane_p}")
    print()
    # Same orientation-detection as compute_scale(): don't assume which
    # pair of opposite edges is the long one, figure it out from the
    # actual scaled lengths (see compute_scale's docstring for why).
    pair_02 = (calib.sheet_edges_cm[0] + calib.sheet_edges_cm[2]) / 2
    pair_13 = (calib.sheet_edges_cm[1] + calib.sheet_edges_cm[3]) / 2
    if pair_02 >= pair_13:
        target_cm = [A4_LONG_CM, A4_SHORT_CM, A4_LONG_CM, A4_SHORT_CM]
    else:
        target_cm = [A4_SHORT_CM, A4_LONG_CM, A4_SHORT_CM, A4_LONG_CM]
    print("sheet edges after scaling (top, right, bottom, left), vs. true A4:")
    all_ok = True
    for measured, target in zip(calib.sheet_edges_cm, target_cm):
        err_pct = 100 * abs(measured - target) / target
        flag = "OK" if err_pct <= EDGE_TOLERANCE_PCT else "OUT OF TOLERANCE"
        all_ok &= err_pct <= EDGE_TOLERANCE_PCT
        print(f"  {measured:6.2f} cm  (target {target:5.2f} cm, {err_pct:4.1f}% off)  {flag}")
    print()
    if all_ok:
        print(f"all edges within {EDGE_TOLERANCE_PCT}% -- calibration looks good, move on to Phase 4.")
    else:
        print(f"some edges are outside {EDGE_TOLERANCE_PCT}% -- fix this before Phase 4.")
        print("common causes: sheet corners tapped out of order, sheet not flat, blur/glare on the sheet.")