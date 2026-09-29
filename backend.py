from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from pathlib import Path
import shutil

from measure import measure_item, measure_space
from sam_utils import segment_points
from fit import placement_fit, passage_fit
import numpy as np
from PIL import Image
import base64
from io import BytesIO

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
Path("out").mkdir(exist_ok=True)  # measure.py's debug overlays land here

app = FastAPI(title="Snap-to-Fit API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serves uploaded/demo photos back to the frontend for the tap-picker canvas
# (e.g. GET /data/shoerack.jpeg) -- image_path everywhere else in this API
# is that same server-side "data/..." path, never a binary upload.
app.mount("/data", StaticFiles(directory="data"), name="data")
app.mount("/out", StaticFiles(directory="out"), name="out")  # measure_item/measure_space debug overlays


@app.get("/api/demo_photos")
def api_demo_photos():
    """The bundled demo photos, for the frontend's quick-pick dropdown."""
    return {"paths": sorted(str(p).replace("\\", "/") for p in DATA_DIR.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))}


class ItemRequest(BaseModel):
    image_path: str
    sheet_px: List[List[float]]
    item_click: List[float]


class SpaceRequest(BaseModel):
    image_path: str
    sheet_px: List[List[float]]
    kind: str
    space_input: List  # [x, y] for kind="floor"; 4 taps for kind="opening"


class FitRequest(BaseModel):
    item: dict
    space: dict
    mode: str


class SegmentPreviewRequest(BaseModel):
    image_path: str
    click: List[float]


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    """Save an uploaded photo into data/ and return its server-side path,
    which every other endpoint below expects (image_path is a path on the
    server, not a binary blob -- see measure.py's contract)."""
    DATA_DIR.mkdir(exist_ok=True)
    dest = DATA_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"ok": True, "image_path": str(dest).replace("\\", "/")}


@app.post("/api/segment_preview")
def api_segment_preview(req: SegmentPreviewRequest):
    """Run SAM on one click and return a base64 PNG overlay -- lets the UI
    show the user what got segmented before they commit to a measurement."""
    try:
        image = Image.open(req.image_path).convert("RGB")
        image_rgb = np.array(image)
        mask = segment_points(image_rgb, [tuple(req.click)])[0]

        overlay = image_rgb.copy()
        overlay[mask] = (overlay[mask] * 0.5 + np.array([255, 60, 60]) * 0.5).astype(np.uint8)
        buf = BytesIO()
        Image.fromarray(overlay).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return {"ok": True, "preview_png_base64": b64}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/measure_item")
def api_measure_item(req: ItemRequest):
    return measure_item(req.image_path, req.sheet_px, req.item_click)


@app.post("/api/measure_space")
def api_measure_space(req: SpaceRequest):
    return measure_space(req.image_path, req.sheet_px, req.kind, req.space_input)


@app.post("/api/fit")
def api_fit(req: FitRequest):
    if req.mode == "placement":
        return placement_fit(req.item, req.space)
    elif req.mode == "passage":
        return passage_fit(req.item, req.space)
    else:
        raise HTTPException(status_code=400, detail="Unknown mode")
