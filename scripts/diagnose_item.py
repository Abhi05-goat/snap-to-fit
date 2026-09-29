"""
Diagnostic for the wire-rack roadblock: dump the height-above-floor and
camera-distance histograms for an item click, and save an overlay of which
pixels survive height filter vs. the foreground/background depth split, so
you can SEE why a candidate point was kept or dropped instead of guessing
from the final L/W/H numbers alone.

    python scripts/diagnose_item.py data/shoerack.jpeg \\
        "561,879 782,1023 476,1273 261,1060" "740,400"

Does not change measure_item's behavior -- read-only diagnostics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from measure import HEIGHT_THRESHOLD_CM, _find_depth_gap, _load_rgb  # noqa: E402
from sam_utils import segment_points  # noqa: E402
from scripts.phase2 import infer, load_model, taps_to_resized  # noqa: E402
from scripts.phase3 import calibrate  # noqa: E402


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    image_path, sheet_str, click_str = sys.argv[1:4]
    sheet_px = [[int(v) for v in pair.split(",")] for pair in sheet_str.split()]
    item_click = [int(v) for v in click_str.split(",")]

    model = load_model()
    depth_result = infer(model, image_path)
    calib = calibrate(image_path, sheet_px, depth_result=depth_result)

    image_rgb = _load_rgb(image_path)
    item_mask = segment_points(image_rgb, [item_click])[0]
    ys, xs = np.where(item_mask)
    print(f"SAM mask: {item_mask.sum()} px, bbox x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}]")

    x1, y1 = taps_to_resized([[int(xs.min()), int(ys.min())]], depth_result.scale_factor)[0]
    x2, y2 = taps_to_resized([[int(xs.max()), int(ys.max())]], depth_result.scale_factor)[0]
    xlo, xhi = sorted((x1, x2))
    ylo, yhi = sorted((y1, y2))
    xhi = min(xhi, depth_result.points.shape[1])
    yhi = min(yhi, depth_result.points.shape[0])

    box_points_raw = depth_result.points[ylo:yhi, xlo:xhi].reshape(-1, 3)
    box_points_raw = box_points_raw[np.isfinite(box_points_raw).all(axis=1)]
    box_points_cm = box_points_raw * calib.scale * 100.0
    plane_p_cm = calib.plane_p * 100.0

    h = (box_points_cm - plane_p_cm) @ calib.plane_n
    print(f"total points in box: {len(box_points_cm)}")
    print(f"height range: {h.min():.1f} to {h.max():.1f} cm")

    above = box_points_cm[h > HEIGHT_THRESHOLD_CM]
    h_above = h[h > HEIGHT_THRESHOLD_CM]
    print(f"points above {HEIGHT_THRESHOLD_CM} cm floor threshold: {len(above)}")

    camera_dist = np.linalg.norm(above, axis=1)
    threshold = _find_depth_gap(camera_dist)
    print(f"camera-distance range (above-floor points): {camera_dist.min():.1f} to {camera_dist.max():.1f} cm")
    if threshold is None:
        print("no empty depth gap found -- single cluster, filter will keep everything")
    else:
        near = camera_dist <= threshold
        print(f"depth gap found: split at {threshold:.1f} cm")
        print(f"  near cluster (kept if both sides are big enough): {near.sum()} points, "
              f"height {h_above[near].min():.1f}-{h_above[near].max():.1f} cm, "
              f"dist {camera_dist[near].min():.1f}-{camera_dist[near].max():.1f} cm")
        print(f"  far cluster (dropped as background if split fires): {(~near).sum()} points, "
              f"dist {camera_dist[~near].min() if (~near).any() else float('nan'):.1f}-"
              f"{camera_dist[~near].max() if (~near).any() else float('nan'):.1f} cm")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(h, bins=100)
        axes[0].axvline(HEIGHT_THRESHOLD_CM, color="red", linestyle="--", label="height threshold")
        axes[0].set_title("height above floor (cm)")
        axes[0].legend()

        axes[1].hist(camera_dist, bins=100)
        if threshold is not None:
            axes[1].axvline(threshold, color="red", linestyle="--", label="depth gap split")
            axes[1].legend()
        axes[1].set_title("camera distance, above-floor points (cm)")

        out_path = REPO_ROOT / "out" / f"{Path(image_path).stem}_diag.png"
        out_path.parent.mkdir(exist_ok=True)
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        print(f"saved histograms to {out_path}")
    except Exception as e:
        print(f"(plot save skipped: {e})")


if __name__ == "__main__":
    main()
