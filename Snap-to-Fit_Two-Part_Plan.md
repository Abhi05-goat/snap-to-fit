# Snap-to-Fit — One-Night MVP Plan (Two-Part Handoff)

**Goal:** by morning, a phone-usable demo: photograph an item plus a space, tap a few points, get *"fits / tight / doesn't fit, with High/Medium/Low confidence."*

**Assumption:** about 8 hours. If you have less, use the cut list at the bottom.

## How the work is split

The pipeline runs left to right, so the split follows the data:

```
PART 1 (you)                                   PART 2 (teammate)
photos → depth → scale → item dims → space dims  ══HANDOFF══►  fit logic → confidence → app → eval/demo
        "pixels → centimetres"                                  "centimetres → verdict → demo"
```

**Sequential in data, parallel in time.** The teammate does not wait for you. At kickoff you both agree on a fixed contract (the JSON shapes and two function signatures below). The teammate builds Part 2 against **mock measurements** while you build the real ones. At handoff (about T+5:00) they change one import from mock to real.

| Owner | Files they own |
|---|---|
| **You (Part 1)** | `measure.py`, `data/` (photos, `ground_truth.csv`), `tests/test_measure.py` |
| **Teammate (Part 2)** | `fit.py`, `confidence.py`, `mock_measure.py`, `app.py`, `eval.py`, `tests/test_fit.py`, `README.md` |

Nobody edits the other person's files. If you need a change, ask.

---

## Decisions made for the night (no debates)

| Topic | Decision |
|---|---|
| Depth model | **UniDepthV2** (start with ViT-S or ViT-B; check the repo README for the exact load call). It outputs metric 3D points directly. |
| Reference object | **A4 sheet** on the floor next to the item (bigger than a card, so less scale noise) |
| Reference detection | **User taps the 4 corners** of the sheet |
| Segmentation | **No SAM.** User taps 2 corners of a box around the item; points above the floor plane inside the box = the item |
| Space measurement | **Tap-to-measure** (tap points, get 3D distances) |
| App | **Gradio** (one Python file, works on a phone browser, provides tap coordinates) |
| Placement | 0° and 90° only |
| Passage | Tilt scan over the 3 possible cross-sections |

---

## The handoff contract (agree in the kickoff, then freeze)

The teammate only ever calls these two functions. All depth, calibration, and plane fitting is hidden behind them.

```python
# measure.py  (Part 1, you)  — mock_measure.py must expose the same two functions

def measure_item(image_path: str, sheet_px: list, box_px: list) -> dict: ...
def measure_space(image_path: str, sheet_px: list, kind: str, taps_px: list) -> dict: ...
```

**Tap conventions (pixel coordinates in the original image):**
- `sheet_px`: 4 points `[[x,y],...]` — the A4 corners, clockwise starting from the top-left as seen in the photo.
- `box_px`: `[x1, y1, x2, y2]` — two opposite corners of a box around the item.
- `kind="floor"`, `taps_px = [c1, c2, c3]`: `width = |c1→c2|`, `depth = |c2→c3|` (c2 is the corner shared by both edges).
- `kind="opening"`, `taps_px = [left_bottom, right_bottom, top, floor_below_top]`: `width = |left_bottom→right_bottom|`, `height = |top→floor_below_top|`.

**Return shapes (JSON-serializable dicts):**

```json
// measure_item
{ "ok": true, "L_cm": 182.0, "W_cm": 88.5, "H_cm": 84.0,
  "sigma_pct": 0.09, "debug_img": "out/sofa_01_debug.png" }

// measure_space, kind = "floor"
{ "ok": true, "kind": "floor", "width_cm": 210.0, "depth_cm": 120.0, "height_cm": null,
  "sigma_pct": 0.09, "debug_img": "out/corner_01_debug.png" }

// measure_space, kind = "opening"
{ "ok": true, "kind": "opening", "width_cm": 82.0, "depth_cm": null, "height_cm": 201.0,
  "sigma_pct": 0.09, "debug_img": "out/door_01_debug.png" }

// any failure
{ "ok": false, "error": "no points found above the floor plane inside the box" }
```

`L_cm ≥ W_cm` always. `sigma_pct` is the estimated relative error (0.09 = ±9%).

**Swap switch:** `app.py` imports from `measure_api.py`, which does:

```python
import os
if os.environ.get("USE_MOCK", "1") == "1":
    from mock_measure import measure_item, measure_space
else:
    from measure import measure_item, measure_space
```

Handoff = run with `USE_MOCK=0`.

---

## Capture protocol (whoever shoots follows it; it's yours in Part 1)

- Use the **main camera**, not ultrawide or portrait mode.
- **Item photo:** 2–3 m away, roughly **45° to the item** so two faces are visible, whole item in frame, A4 sheet **flat on the floor touching the item's base**, same distance from the camera as the item.
- **Floor-space photo:** stand back, A4 sheet on the floor inside the space, corners of the free area visible.
- **Doorway photo:** face the door straight on, A4 sheet flat on the floor at the threshold, full opening visible.
- Avoid mirrors, glass, and strong glare. Decent lighting.

---

## Kickoff (both, T+0:00 – 0:20)

- **Input:** this plan.
- **Work:** agree the contract above. Create the repo and folders. You stub `measure.py` (both functions return the hard-coded example dicts). The teammate writes `mock_measure.py` returning several realistic sample dicts (a chair, a sofa, a cabinet, a narrow door, a wide door, a tight corner, a big corner) so they can test all verdict types.
- **Output:** repo pushed, both stubs importable, contract frozen.

---

# PART 1 — Pixels → centimetres (you)

## Phase 1 — Capture and ground truth (T+0:20 – 1:00)

- **Input:** phone, A4 sheets, tape measure, about 6 items (chair, stool, small table, box, cabinet, suitcase), 2 doorways, 2 floor corners.
- **Work:**
  - First, start the UniDepth environment install and weight download in the background (it takes a while).
  - Shoot every item and space per the capture protocol. 2 photos per item if you can (different angles).
  - Tape-measure each item (L × W × H) and each space (width and depth, or width and height).
  - Physically test 3–4 (item, space) pairs and record fit / no fit.
  - Write `data/ground_truth.csv`. For each photo record: `file, kind, L, W, H` (items) or `file, kind, width, depth_or_height` (spaces), plus a second table `fit_pairs.csv` with `item_file, space_file, scenario, actually_fits`.
- **Output:** 10+ photos in `data/`, `ground_truth.csv`, `fit_pairs.csv`. Push these right away; the teammate needs them later for evaluation.

## Phase 2 — Depth inference (T+1:00 – 1:40)

- **Input:** installed UniDepthV2, photos.
- **Work:**
  - `load_model()` once, `infer(image) -> raw points (H × W × 3, camera frame, metres)`.
  - Downscale big phone photos to about 0.3–0.6 MP before inference (matches the paper's training resolution range); keep the scale factor to map taps back to the resized image.
  - Cache the output per image (`.npy`) so repeated calls are instant.
  - Sanity check on 2–3 photos: pick two pixels with a known real distance and compare.
- **Output:** `infer()` works. A one-line note: seconds per image and VRAM used.

## Phase 3 — Calibration: scale and floor plane (T+1:40 – 2:40)

- **Input:** raw points, the 4 sheet taps per photo.
- **Work:**
  1. Each sheet corner → 3D point (median of the raw points in a 5×5 window).
  2. Compute the 4 edge lengths (two should be 29.7 cm, two 21.0 cm). `scale` = mean of (true ÷ measured). `scale_cv` = spread of those ratios (a quality signal).
  3. Fit a plane (SVD) to the raw points inside the tapped quadrilateral, scaled by `scale`; trim the worst 10% and refit. Orient the normal toward the camera.
  4. Bundle as an internal `Calib(scale, plane_n, plane_p, scale_cv)`.
- **Output:** a script that prints, for every photo, `scale`, `scale_cv`, and the sheet's edges after scaling (they should be within about 3% of 29.7 / 21.0). Fix this before moving on; everything else depends on it.

## Phase 4 — Item dimensions (T+2:40 – 4:00)

- **Input:** raw points, `Calib`, the box taps.
- **Work:** build the real `measure_item`:
  1. Raw points inside the box × `scale`.
  2. Height above the plane: `h = dot(p − plane_p, plane_n)`. Keep points with `h > 3 cm` (drops floor and paper).
  3. Project the kept points onto the plane, run PCA on the 2D footprint to get the two axes.
  4. `L` and `W` = the 2nd–98th percentile spread along each axis (sorted so `L ≥ W`). `H` = 98th percentile of `h`.
  5. `sigma_pct = sqrt(0.08² + scale_cv²)` for now.
  6. Save a debug image (kept points overlaid on the photo).
  7. Wrap everything in try/except and return `{"ok": false, "error": ...}` on failure.
- **Output:** `measure_item` returns valid dicts for every item photo, plus a table of predicted vs. tape-measured L / W / H.
- **Early drop (T+4:00):** push `measure_item` and the table. The teammate can start testing against real numbers.

## Phase 5 — Space dimensions (T+4:00 – 5:00)

- **Input:** raw points, `Calib`, the space taps.
- **Work:** build the real `measure_space`. Each tap → 3D point (median over a 5×5 window, × `scale`), and distance = norm of the difference. `kind="floor"`: width and depth per the contract. `kind="opening"`: width and height per the contract. `sigma_pct` as for items. Same error handling and debug image.
- **Output:** `measure_space` returns valid dicts for every space photo, plus a predicted vs. tape-measured table.

### ▶ HANDOFF (T+5:00)

Push and tell the teammate. The handoff package is:
- `measure.py` with the two real functions
- `data/` photos, `ground_truth.csv`, `fit_pairs.csv`
- `data/taps.json`: the taps you used per photo (`file → {sheet_px, box_px or taps_px}`) so the eval script can replay them
- `data/part1_results.md`: your predicted vs. ground truth tables and known failure notes
- one-line run note (e.g. `USE_MOCK=0 python app.py`)

---

# PART 2 — Centimetres → verdict → demo (teammate)

Phases 6–9 run **at the same time as Part 1**, using `mock_measure.py`.

## Phase 6 — Placement fit (T+0:20 – 1:30)

- **Input:** the contract, mock item and floor dicts.
- **Work:** `fit.py: placement_fit(item, space, margin_cm=15)`:
  - Try the item footprint (L × W) in both orientations against the space footprint (width × depth).
  - Margin per orientation = the smaller of the two axis margins (space − item). Pick the best orientation.
  - Classify: ≥ 15 cm comfortable, 0–15 cm tight, ≤ 0 no fit.
  - Return `{verdict, margin_cm, detail (which orientation), prob_fit, tier}` (probability and tier come from Phase 8; leave placeholders until then).
  - Write `tests/test_fit.py` cases: obvious fit, obvious no-fit, fits only rotated, razor-thin.
- **Output:** placement tests pass.

## Phase 7 — Passage fit (T+1:30 – 2:40)

- **Input:** the contract, mock item and opening dicts.
- **Work:** `passage_fit(item, space, margin_cm=15)`:
  - The item can pass with any of three dimension pairs as its cross-section: (W, H), (L, H), (L, W).
  - For each pair `(w, h)`, scan θ from 0° to 90° in 1° steps. A tilt works if `w·cosθ + h·sinθ ≤ opening_width` and `w·sinθ + h·cosθ ≤ opening_height`.
  - Keep the best clearance (smallest tilt that works, and its margin). Report which pair and which angle.
  - Classify with the same comfortable / tight / no-fit thresholds.
  - Tests: straight-on fits, needs a tilt, needs turning on its side, cannot pass.
- **Output:** passage tests pass.

## Phase 8 — Confidence engine (T+2:40 – 3:40)

- **Input:** a `FitResult` margin plus the `sigma_pct` values from both dicts.
- **Work:** `confidence.py`:
  - `SIGMA_FLOOR = 0.08` (retuned in Phase 10 from real errors). `sigma = max(dict sigma_pct, SIGMA_FLOOR)`.
  - `margin_std = sqrt((sigma_item · item_dim)² + (sigma_space · space_dim)²)` using the dimensions that decided the margin.
  - `prob_fit = Φ(margin / margin_std)` (use `math.erf`).
  - Certainty = `max(prob_fit, 1 − prob_fit)`. Tier: High ≥ 0.90, Medium 0.70–0.90, Low < 0.70.
  - Plug into both fit functions. The output is never a bare yes/no.
  - Tests: a big margin gives High, a +2 cm margin gives Low.
- **Output:** every fit result carries `prob_fit` and `tier`. All tests pass.

## Phase 9 — App (T+3:40 – 5:00)

- **Input:** the fit functions, `measure_api.py` (mock for now).
- **Work:** build `app.py` in Gradio, with `gr.Image(...).select(...)` for taps and `gr.State` for the tap list:
  - **Item flow:** upload photo → tap the 4 sheet corners in order → tap 2 opposite corners of a box around the item → show measured L × W × H.
  - **Space flow:** a mode selector (Placement / Passage). Tap prompts change per mode ("tap the left floor corner…", "tap the top of the doorway…").
  - **Result card:** verdict, margin (cm), orientation or tilt, confidence tier and probability, debug images.
  - **Manual override** fields for item dimensions and space dimensions (also the fallback when `ok: false`).
  - Reset button and a "next tap" prompt at every step. Show `error` text when a measurement fails.
  - Launch with `demo.launch(server_name="0.0.0.0")` and test from a phone on the same Wi-Fi. Check the returned tap coordinates are in original-image pixels (the contract depends on it); rescale if not.
- **Output:** the full flow works on a phone against the mock. When Part 1 lands, only the `USE_MOCK` switch changes.

## Phase 10 — Evaluate and package (T+5:00 – 6:30, after the handoff)

- **Input:** the handoff package, working app.
- **Work:**
  - Switch to `USE_MOCK=0`. Run the app once on a fresh photo to confirm the swap works.
  - `eval.py`: loop over `data/taps.json`, call the real functions, and print item MAE (cm and %), space MAE, and fit-decision accuracy against `fit_pairs.csv`. Also a table of confidence tier vs. whether the verdict was right.
  - Set `SIGMA_FLOOR` from the observed median % error (round up a little) and re-run.
  - Write the 5-minute demo script (one placement, one passage, one deliberate hard case showing Low confidence), the `README.md` with run steps, 3 backup photo sets, and a screen recording of a full successful run.
- **Output:** the eval table, tuned `SIGMA_FLOOR`, `README.md`, backup photos, backup video.

---

# Final merge and rehearsal (both, T+5:00 – 7:30)

- **Input:** everything.
- **Work:**
  - **You (after handoff):** stay on call for `measure.py`. Look at the debug images for the worst 3 errors from the teammate's eval and fix what you can (percentile range, height threshold, window size). Confirm the model loads once at startup and responds in a few seconds. Make sure failures return `ok: false` rather than crash.
  - **Teammate:** re-run `eval.py` after each of your fixes.
  - **Both (T+6:30):** rehearse the demo once, start to finish, on a phone, using the backup photos.
- **Output:** a frozen, tagged commit and a rehearsed demo. Buffer T+7:30 – 8:00 for whatever the rehearsal exposes.

---

## Timeline at a glance

| Time | You (Part 1) | Teammate (Part 2) |
|---|---|---|
| 0:00 – 0:20 | Kickoff (both) | Kickoff (both) |
| 0:20 – 1:00 | P1 Capture and ground truth | P6 Placement fit (from 0:20) |
| 1:00 – 1:40 | P2 Depth inference | P6 → P7 Passage fit (from 1:30) |
| 1:40 – 2:40 | P3 Calibration | P7 Passage fit |
| 2:40 – 4:00 | P4 Item dimensions | P8 Confidence (to 3:40), P9 App starts |
| 4:00 – 5:00 | P5 Space dimensions | P9 App |
| **5:00** | **HANDOFF** | swap `USE_MOCK=0` |
| 5:00 – 6:30 | Fix errors from eval | P10 Evaluate and package |
| 6:30 – 7:30 | Rehearsal | Rehearsal |
| 7:30 – 8:00 | Buffer | Buffer |

---

## Fallback ladder (use in this order)

1. Depth result looks wrong → retake the photo per the capture protocol.
2. Still wrong → use the manual dimension override (the fit logic and UI still demo).
3. Model or GPU trouble → smaller variant (ViT-S) or lower input resolution.
4. Height-based segmentation fails on a piece (dark, glossy, thin legs) → demo a cleaner item and mention it as a known limitation.
5. **If Part 1 isn't done at T+5:00:** hand over what exists (e.g. `measure_item` only). The teammate keeps the mock for the missing function and demos the rest with manual override.

## Cut list (drop in this order if behind)

1. Debug overlay images
2. Passage tilt scan (keep straight-on and the on-its-side check only)
3. `SIGMA_FLOOR` tuning (use a flat 10%)
4. Tap prompt polish

**Never cut:** placement fit, the confidence tier, the manual override, the phone flow.

## Known limitations to state in the demo

- A single view sees only the visible surfaces, so the hidden dimension is estimated from the two visible faces at 45° and can be slightly low.
- Scale comes from one A4 sheet, so accuracy depends on the sheet sitting at about the same distance as the item.
- Mirrors, glass, glossy surfaces, and wide-angle lenses degrade results (the UniDepthV2 paper's failure cases show this).
- Passage is a 2D cross-section check. There is no multi-turn hallway modelling.

## Definition of done

- [ ] Phone → photos → item dimensions on screen (real, not mock)
- [ ] Placement verdict with margin and High/Medium/Low tier
- [ ] Passage verdict with tilt angle and tier
- [ ] Error table (item MAE, space MAE, fit accuracy)
- [ ] README, backup photos, backup video
