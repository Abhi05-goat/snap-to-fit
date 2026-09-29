"""
Tests for the pure geometry in measure.py / scripts/phase3.py -- everything
here runs on synthetic point clouds, no GPU or UniDepth install required.
This is deliberately what you can run right now, tonight, before you even
have real photos, to sanity-check the math itself.

    python -m pytest tests/test_measure.py -v

What's NOT covered here (needs a real photo + GPU): scripts/phase2.py's
infer()/load_model(), and the full measure_item/measure_space wrappers
(they call infer+calibrate internally). Once you have UniDepth running,
scripts/phase3.py's __main__ CLI and phase2.py's sanity-check CLI are your
real-data checks -- see their docstrings.
"""

import numpy as np
import pytest

from scripts.phase3 import compute_scale, fit_floor_plane, A4_LONG_CM, A4_SHORT_CM
from measure import _extract_item_dims, _extract_space_dims, HEIGHT_THRESHOLD_CM


# ---------------------------------------------------------------------------
# Phase 3: compute_scale
# ---------------------------------------------------------------------------

def test_compute_scale_perfect_sheet():
    """A sheet measured in 'model units' at exactly half true scale should
    yield scale=2.0 and scale_cv=0 (all 4 edges equally off)."""
    # top, right, bottom, left, in model units at half of true metres
    long_m = (A4_LONG_CM / 100) / 2
    short_m = (A4_SHORT_CM / 100) / 2
    corners = np.array([
        [0, 0, 0],
        [long_m, 0, 0],
        [long_m, short_m, 0],
        [0, short_m, 0],
    ])
    scale, scale_cv = compute_scale(corners)
    assert scale == pytest.approx(2.0, rel=1e-6)
    assert scale_cv == pytest.approx(0.0, abs=1e-9)


def test_compute_scale_handles_rotated_sheet_orientation():
    """Regression test: a real photo hit this exact bug. A4's long/short
    edges alternate around the quadrilateral, but WHICH pair (edges 0&2 or
    1&3) is the long one depends on which corner happened to get tapped
    first -- there's no fixed 'first edge is always long' rule. An earlier
    version hardcoded that assumption and broke on a real photo where the
    short edge came first in tap order, producing a very distinctive
    failure signature: edges alternating ~25% under and ~50% over their
    (wrongly) assigned target. This test uses the OPPOSITE orientation
    from test_compute_scale_perfect_sheet (short edge first) and checks
    compute_scale still recovers the correct scale."""
    long_m = (A4_LONG_CM / 100) / 2
    short_m = (A4_SHORT_CM / 100) / 2
    # short, long, short, long -- rotated 90 degrees from the "expected" case
    corners = np.array([
        [0, 0, 0],
        [short_m, 0, 0],
        [short_m, long_m, 0],
        [0, long_m, 0],
    ])
    scale, scale_cv = compute_scale(corners)
    assert scale == pytest.approx(2.0, rel=1e-6)
    assert scale_cv == pytest.approx(0.0, abs=1e-9)


def test_compute_scale_noisy_sheet_has_nonzero_cv():
    """If the 4 edges don't agree on scale (noisy taps/depth), scale_cv
    should be > 0 -- this is the quality signal Phase 3 asks you to check."""
    long_m = (A4_LONG_CM / 100) / 2
    short_m = (A4_SHORT_CM / 100) / 2
    corners = np.array([
        [0, 0, 0],
        [long_m * 1.05, 0, 0],       # 5% noise on the top edge
        [long_m * 1.05, short_m, 0],
        [0, short_m, 0],
    ])
    scale, scale_cv = compute_scale(corners)
    assert scale_cv > 0.01


# ---------------------------------------------------------------------------
# Phase 3: fit_floor_plane
# ---------------------------------------------------------------------------

def test_fit_floor_plane_flat_z0():
    """A perfectly flat set of points at z=0 should fit a plane with
    normal ~[0, 0, 1] (oriented toward the camera at the origin, i.e.
    toward -z from the floor's perspective -- camera looks down at floor
    below it, so normal points toward +z if camera is above)."""
    rng = np.random.default_rng(0)
    xy = rng.uniform(-1, 1, size=(200, 2))
    pts = np.column_stack([xy, np.zeros(200)])  # z=0 plane
    camera_pos = np.array([0, 0, 2.0])  # camera 2m above the floor

    normal, centroid = fit_floor_plane(pts, scale=1.0, camera_pos=camera_pos)

    assert np.linalg.norm(centroid) < 0.1  # centroid near origin
    # normal should be close to [0, 0, 1] (or its negation before orientation
    # -- after orientation toward the camera at +z, it should point +z)
    assert abs(normal[2]) > 0.99
    assert normal[2] > 0  # oriented toward the camera above


def test_fit_floor_plane_handles_realistic_point_counts():
    """Regression test: a real photo's tapped-quad bounding box can easily
    contain 50,000-100,000+ raw points (a few hundred px square at full
    sensor resolution). An earlier version of _fit_plane called
    np.linalg.svd() directly on the full (N, 3) array, which hit a real
    NumPy/OpenBLAS bug on Windows where the SVD 'gesdd' driver's
    workspace-size calculation overflows for very tall, skinny matrices --
    it threw a nonsensical MemoryError on an array that's a few hundred KB.
    All the OTHER synthetic tests in this file use at most ~2000 points, so
    none of them would have caught this; this one deliberately uses a
    six-figure point count to match what a real photo produces."""
    rng = np.random.default_rng(3)
    n = 150_000
    xy = rng.uniform(-1, 1, size=(n, 2))
    pts = np.column_stack([xy, np.zeros(n)])  # flat z=0 plane, 150k points

    normal, centroid = fit_floor_plane(pts, scale=1.0, camera_pos=np.array([0, 0, 2.0]))

    assert abs(normal[2]) > 0.99
    assert np.linalg.norm(centroid) < 0.05


def test_fit_floor_plane_trims_outliers():
    """A plane fit should be robust to ~5% gross outliers (Phase 3's
    'trim the worst 10%, refit' step)."""
    rng = np.random.default_rng(1)
    xy = rng.uniform(-1, 1, size=(190, 2))
    flat = np.column_stack([xy, np.zeros(190)])
    outliers = np.column_stack([
        rng.uniform(-1, 1, size=(10, 2)),
        rng.uniform(2, 5, size=10),  # way off the floor plane
    ])
    pts = np.vstack([flat, outliers])

    normal, _ = fit_floor_plane(pts, scale=1.0, camera_pos=np.array([0, 0, 2.0]))
    assert abs(normal[2]) > 0.95  # still recovers a near-flat plane despite outliers


def test_fit_floor_plane_too_few_points_raises():
    pts = np.zeros((3, 3))
    with pytest.raises(ValueError):
        fit_floor_plane(pts, scale=1.0)


# ---------------------------------------------------------------------------
# Phase 4: _extract_item_dims
# ---------------------------------------------------------------------------

def _box_of_points(L_cm, W_cm, H_cm, n=2000, seed=0, plane_n=None, plane_p=None):
    """Synthetic uniform point cloud filling a box of the given real
    dimensions, sitting on a floor plane, plus a thin layer of floor/paper
    noise at h<=3cm that _extract_item_dims should filter out.

    Default plane_p is offset ~3m from the origin along Y (the camera's
    horizontal viewing direction) -- not [0,0,0] and not along the plane
    normal Z -- because the origin IS the camera (see
    _drop_far_background's docstring in measure.py: camera-frame points,
    camera at the origin). A real item sits meters away roughly level with
    the camera, its height (Z) axis roughly perpendicular to the viewing
    direction, so distance-from-camera stays ~constant across the item's
    own height range. plane_p=[0,0,0] put the "camera" inside the item's
    own point cloud (unrealistic); offsetting along Z instead of Y aliased
    "distance from camera" with "height above the floor" for this box's
    geometry, which spuriously chopped a solid item's height in half --
    both caught by test_extract_item_dims_keeps_solid_item_with_no_background."""
    rng = np.random.default_rng(seed)
    plane_n = np.array([0, 0, 1.0]) if plane_n is None else plane_n
    plane_p = np.array([0, 300.0, 0]) if plane_p is None else plane_p

    n_item = int(n * 0.8)
    n_floor = n - n_item

    # Item points: local frame (u, v, h) then mapped into world via plane_n/plane_p,
    # simple case since plane_n = [0,0,1] means world xyz = u,v,h directly + plane_p.
    u = rng.uniform(-L_cm / 2, L_cm / 2, n_item)
    v = rng.uniform(-W_cm / 2, W_cm / 2, n_item)
    h = rng.uniform(0, H_cm, n_item)
    item_pts = np.column_stack([u, v, h]) + plane_p

    # Floor/paper noise right at the plane (h in [0, 2] cm) -- should be dropped
    # by the h > HEIGHT_THRESHOLD_CM filter.
    fu = rng.uniform(-L_cm, L_cm, n_floor)
    fv = rng.uniform(-W_cm, W_cm, n_floor)
    fh = rng.uniform(0, HEIGHT_THRESHOLD_CM - 0.5, n_floor)
    floor_pts = np.column_stack([fu, fv, fh]) + plane_p

    return np.vstack([item_pts, floor_pts]), plane_n, plane_p


def test_extract_item_dims_recovers_known_box():
    """A synthetic 180 x 90 x 80 cm box should come back close to those
    numbers (within the percentile-trim's expected shrinkage)."""
    pts, plane_n, plane_p = _box_of_points(L_cm=180, W_cm=90, H_cm=80, seed=0)
    L, W, H, n_kept = _extract_item_dims(pts, plane_p, plane_n)

    # 2nd-98th percentile spread is slightly less than the full uniform range
    assert L == pytest.approx(180, rel=0.08)
    assert W == pytest.approx(90, rel=0.08)
    assert H == pytest.approx(80, rel=0.08)
    assert L >= W  # contract: L >= W always
    assert n_kept > 1000  # floor noise was filtered out


def test_extract_item_dims_filters_floor_noise():
    """Points at or below the height threshold should never influence the
    result -- swap in wildly-out-of-range floor noise and dims shouldn't change."""
    pts1, plane_n, plane_p = _box_of_points(L_cm=100, W_cm=50, H_cm=40, seed=2)
    L1, W1, H1, _ = _extract_item_dims(pts1, plane_p, plane_n)

    rng = np.random.default_rng(2)
    n_item = int(2000 * 0.8)
    n_floor = 2000 - n_item
    u = rng.uniform(-50, 50, n_item)
    v = rng.uniform(-25, 25, n_item)
    h = rng.uniform(0, 40, n_item)
    item_pts = np.column_stack([u, v, h]) + plane_p
    # much noisier floor points this time, still below threshold
    fu = rng.uniform(-500, 500, n_floor)
    fv = rng.uniform(-500, 500, n_floor)
    fh = rng.uniform(-10, HEIGHT_THRESHOLD_CM - 0.1, n_floor)
    floor_pts = np.column_stack([fu, fv, fh]) + plane_p
    pts2 = np.vstack([item_pts, floor_pts])

    L2, W2, H2, _ = _extract_item_dims(pts2, plane_p, plane_n)

    assert L1 == pytest.approx(L2, rel=1e-6)
    assert W1 == pytest.approx(W2, rel=1e-6)
    assert H1 == pytest.approx(H2, rel=1e-6)


def test_extract_item_dims_too_few_points_raises():
    pts = np.array([[0, 0, 0.1]] * 5)  # all below height threshold
    with pytest.raises(ValueError):
        _extract_item_dims(pts, np.zeros(3), np.array([0, 0, 1.0]))


def test_extract_item_dims_excludes_far_background():
    """Regression test for the wire-rack roadblock (HANDOFF doc section 6):
    an open/mesh item lets the camera see a large flat wall behind it
    through its gaps. Wall points above the height threshold used to count
    as "item", inflating L/W/H -- and a naive median-depth filter would
    actually pick the WALL as the reference depth here, since the wall has
    more visible points than the sparse wire item. The camera at the
    origin (per Calib's orientation, see scripts/phase3.py) means a far
    wall has much larger norm than the near item, with a real empty gap
    of depth between them (nothing floats in that gap) -- gap detection
    on that distance should isolate the near cluster regardless of how
    many points are on each side."""
    plane_n = np.array([0.0, 0.0, 1.0])
    plane_p = np.array([0.0, 300.0, 0.0])  # item ~3m from the camera (origin)
    pts, _, _ = _box_of_points(L_cm=60, W_cm=20, H_cm=60, seed=5, n=500,
                                plane_n=plane_n, plane_p=plane_p)

    rng = np.random.default_rng(5)
    n_wall = 5000  # far outnumbers the sparse wire-frame item
    wu = rng.uniform(-200, 200, n_wall)
    wh = rng.uniform(5, 100, n_wall)
    # a wall well behind the item (bigger Y -> farther from the camera at
    # the origin, same axis as plane_p's offset), same box footprint, more
    # points than the sparse wire-frame rack
    wall_pts = np.column_stack([wu, np.zeros(n_wall), wh]) + plane_p + np.array([0.0, 300.0, 0.0])

    pts_with_wall = np.vstack([pts, wall_pts])
    L, W, H, n_kept = _extract_item_dims(pts_with_wall, plane_p, plane_n)

    assert L == pytest.approx(60, rel=0.15)
    assert W == pytest.approx(20, rel=0.15)
    assert H == pytest.approx(60, rel=0.15)
    assert n_kept < 1000  # wall cluster (5000 pts) must have been dropped


def test_extract_item_dims_keeps_solid_item_with_no_background():
    """A solid item (no visible background behind it, e.g. a sofa flush
    against a wall) has one continuous depth cluster and no real empty
    gap -- the filter must NOT invent a split and chop the item in half."""
    pts, plane_n, plane_p = _box_of_points(L_cm=180, W_cm=90, H_cm=80, seed=0)
    L, W, H, n_kept = _extract_item_dims(pts, plane_p, plane_n)

    assert L == pytest.approx(180, rel=0.08)
    assert W == pytest.approx(90, rel=0.08)
    assert H == pytest.approx(80, rel=0.08)
    assert n_kept > 1000  # nothing spuriously dropped


# ---------------------------------------------------------------------------
# Phase 5: _extract_space_dims
# ---------------------------------------------------------------------------

def test_extract_space_dims_floor():
    c1 = np.array([0.0, 0.0, 0.0])
    c2 = np.array([210.0, 0.0, 0.0])   # width = 210
    c3 = np.array([210.0, 120.0, 0.0])  # depth = 120
    width, depth, height = _extract_space_dims("floor", [c1, c2, c3])
    assert width == pytest.approx(210.0)
    assert depth == pytest.approx(120.0)
    assert height is None


def test_extract_space_dims_opening():
    left_bottom = np.array([0.0, 0.0, 0.0])
    right_bottom = np.array([82.0, 0.0, 0.0])   # width = 82
    top = np.array([0.0, 0.0, 201.0])
    floor_below_top = np.array([0.0, 0.0, 0.0])  # height = 201
    width, depth, height = _extract_space_dims(
        "opening", [left_bottom, right_bottom, top, floor_below_top]
    )
    assert width == pytest.approx(82.0)
    assert depth is None
    assert height == pytest.approx(201.0)