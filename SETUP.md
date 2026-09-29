# Setup

Everything needed to run Snap-to-Fit locally, from a fresh clone.

## Prerequisites

- A GPU machine (CUDA). CPU will technically run but depth inference will be
  very slow.
- Python 3.10+ and Node.js 18+.

## 1. Clone this repo

```bash
git clone https://github.com/<your-org>/snap-to-fit.git
cd snap-to-fit
```

The MobileSAM checkpoint (`checkpoints/mobile_sam.pt`, ~39MB) is already
included -- no separate download needed.

## 2. Install UniDepth (not on PyPI, installed from source)

```bash
git clone https://github.com/lpiccinelli-eth/UniDepth.git
cd UniDepth
python -m venv .venv

# Windows:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118
cd ..
```

Adjust the `cu118` suffix if your GPU needs a different CUDA build (see
[pytorch.org](https://pytorch.org/get-started/locally/) for the right index
URL for your setup).

## 3. Install this project's dependencies

With the UniDepth venv still active:

```bash
cd snap-to-fit
pip install -r requirements.txt
```

This installs FastAPI, MobileSAM, OpenCV, and everything else `measure.py`/
`backend.py` need. `pytest` is included too.

## 4. Verify the backend works

```bash
python -m pytest tests/ -v
```

All 22 tests should pass -- these are pure-geometry tests (no GPU needed),
so this is a good first sanity check even before touching the real model.

Then try a real measurement against the bundled demo photos:

```bash
python run_measure.py item data/shoerack.jpeg \
  --sheet "[[561,879],[782,1023],[476,1273],[261,1060]]" \
  --click "[740,400]"
```

First run loads UniDepthV2 (~2s) and MobileSAM, then runs inference. You
should see `"ok": true` with L/W/H values.

## 5. Run the full app

Terminal 1 (backend):
```bash
uvicorn backend:app --host 0.0.0.0 --port 8000
```

Terminal 2 (frontend):
```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**.

## 6. Regression-test against ground truth

```bash
python test_real_data.py
```

Replays the 3 demo photos' recorded clicks (`data/taps.json`) through the
live HTTP API and compares against `data/ground_truth.csv`.

## Troubleshooting

- **`KMP_DUPLICATE_LIB_OK` / OpenMP crash on Windows**: already handled --
  `scripts/phase2.py` and `scripts/phase3.py` set this env var defensively
  at import time.
- **`torch.cuda.is_available()` is False**: your torch build doesn't match
  your CUDA driver. Re-check step 2's `--extra-index-url`.
- **MobileSAM import errors**: make sure you're in the same venv you
  installed UniDepth into (step 2) when you ran `pip install -r
  requirements.txt` in step 3 -- they need to share a torch install.
- **Canvas tap picker in the browser feels imprecise**: use
  `python scripts/pick_points.py <photo>` instead (matplotlib-based, shows a
  live SAM mask preview) and feed its printed coordinates into
  `run_measure.py` -- see the README for the full CLI workflow.

## What's what

See **ARCHITECTURE.md** for how the pipeline actually works end-to-end
(depth inference -> calibration -> segmentation -> measurement -> fit
logic -> confidence scoring), including the reasoning behind each design
decision and answers to common questions about accuracy and trust.
