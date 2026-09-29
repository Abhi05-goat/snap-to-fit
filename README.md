# Snap-to-Fit

A tool to determine if furniture fits in your space using your phone camera.
Real monocular depth (UniDepthV2) + SAM (MobileSAM) segmentation -- no mock
mode, no manual bounding-box dragging.

## How it works

1. Photograph the item/space next to a flat A4 sheet (used as a metric
   scale reference).
2. Tap the sheet's 4 corners (still manual -- see "Why the sheet is manual
   taps" below).
3. Click ONCE on the item, or ONCE on the floor patch, you want measured.
   SAM segments it automatically. (A doorway *opening* is empty space, not
   a segmentable object, so it still takes 4 manual taps -- see
   `measure.py`'s module docstring.)
4. Get L/W/H or width/depth/height, then check it against a space for a
   fit verdict with a confidence tier.

## Setup

See **[SETUP.md](SETUP.md)** for full from-scratch setup (cloning UniDepth,
the GPU/torch install, dependencies, and troubleshooting). Short version,
once UniDepth + this repo's `requirements.txt` are installed:

```bash
uvicorn backend:app --reload --host 0.0.0.0 --port 8000   # terminal 1

cd frontend && npm install && npm run dev                  # terminal 2
```

## Workflow

- Open the web app, upload a photo (or pick one already in `data/`).
- Click the A4 sheet's 4 corners, then click the item (or floor patch).
- Check the SAM mask preview looks right, then measure.
- Go to **Check Fit**, select Placement or Passage, and get a verdict
  (Fits / Tight / Doesn't Fit) with a confidence tier.

## Why the sheet is manual taps

SAM was tried for the A4 sheet too, but its mask boundary on a small,
low-contrast rectangle lands a few pixels off the true edge -- fine for a
big object, but enough to visibly distort the sheet's edge-length ratios
and, through them, the scale factor every measurement in the photo depends
on. A human reads off 4 corners of a small flat rectangle accurately in a
couple of clicks anyway; that was never the error-prone part of the old
workflow (the item's hand-drawn bounding box was). See `measure.py`'s
module docstring for the full reasoning.

## Regression-testing against real photos

```bash
uvicorn backend:app --host 0.0.0.0 --port 8000   # terminal 1
python test_real_data.py                          # terminal 2
```
Replays the recorded clicks in `data/taps.json` through the live HTTP API
and compares against `data/ground_truth.csv`.

## Debug tools

- `scripts/pick_points.py <photo>` -- click points on a new photo, with a
  live SAM mask preview, to get copy-pasteable `sheet_px`/`item_click`.
- `scripts/diagnose_item.py <photo> "<sheet taps>" "<item click>"` -- dumps
  the height/depth-gap histograms behind an item measurement.
