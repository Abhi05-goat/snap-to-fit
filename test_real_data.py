"""
Test the FULL real pipeline (backend HTTP API -> measure.py -> sam_utils.py
-> fit.py -> confidence.py) against the 3 real photos in data/, using the
recorded clicks in data/taps.json and comparing to data/ground_truth.csv.

Requires the backend running in another terminal:

    uvicorn backend:app --host 0.0.0.0 --port 8000

Then, in a second terminal:

    python test_real_data.py

This does NOT call measure.py directly -- it goes through the actual HTTP
endpoints (/api/measure_item, /api/measure_space, /api/fit), so it proves
the whole integration (not just the measurement math) works end to end.
"""

import csv
import json

import requests

API_URL = "http://localhost:8000/api"


def load_ground_truth():
    rows = {}
    with open("data/ground_truth.csv", newline="") as f:
        for row in csv.DictReader(f):
            rows[row["file"]] = row
    return rows


def pct_err(pred, truth):
    if pred is None or truth in (None, ""):
        return "n/a"
    truth = float(truth)
    if truth == 0:
        return "n/a"
    return f"{100 * abs(pred - truth) / truth:.1f}%"


def main():
    taps = json.loads(open("data/taps.json").read())
    ground_truth = load_ground_truth()

    print("=" * 70)
    print("REAL DATA TEST -- via HTTP API (backend must be running: uvicorn backend:app)")
    print("=" * 70)

    results = {}  # file -> measured dict, so we can feed into /api/fit next

    for file, tap in taps.items():
        gt = ground_truth.get(file, {})
        print(f"\n--- {file} ---")

        if tap["type"] == "item":
            resp = requests.post(f"{API_URL}/measure_item", json={
                "image_path": file,
                "sheet_px": tap["sheet_px"],
                "item_click": tap["item_click"],
            })
            result = resp.json()
            if not result.get("ok"):
                print(f"  FAILED: {result.get('error')}")
                continue
            for field, key in (("L_cm", "L"), ("W_cm", "W"), ("H_cm", "H")):
                pred = result[field]
                truth = gt.get(key)
                print(f"  {key}: predicted={pred:6.1f} cm   truth={truth:>6} cm   error={pct_err(pred, truth)}")
            print(f"  sigma_pct: {result['sigma_pct']}")
            results[file] = result

        else:  # space
            resp = requests.post(f"{API_URL}/measure_space", json={
                "image_path": file,
                "sheet_px": tap["sheet_px"],
                "kind": tap["kind"],
                "space_input": tap["space_input"],
            })
            result = resp.json()
            if not result.get("ok"):
                print(f"  FAILED: {result.get('error')}")
                continue
            pred_w = result["width_cm"]
            print(f"  width: predicted={pred_w:6.1f} cm   truth={gt.get('width'):>6} cm   error={pct_err(pred_w, gt.get('width'))}")
            second_field = "depth_cm" if tap["kind"] == "floor" else "height_cm"
            label = "depth" if tap["kind"] == "floor" else "height"
            pred2 = result[second_field]
            print(f"  {label}: predicted={pred2:6.1f} cm   truth={gt.get('depth_or_height'):>6} cm   error={pct_err(pred2, gt.get('depth_or_height'))}")
            print(f"  sigma_pct: {result['sigma_pct']}")
            results[file] = result

    # Now run the fit logic on real measurements: shoerack vs doorframe (passage),
    # shoerack vs cupboard floor (placement).
    item = results.get("data/shoerack.jpeg")
    opening = results.get("data/doorframe.jpg")
    floor = results.get("data/cupboard_space.jpg")

    print("\n" + "=" * 70)
    print("FIT CHECKS (real measurements -> /api/fit)")
    print("=" * 70)

    if item and opening:
        resp = requests.post(f"{API_URL}/fit", json={"item": item, "space": opening, "mode": "passage"})
        r = resp.json()
        print(f"\nShoerack through doorframe (passage):")
        print(f"  verdict={r['verdict']}  margin={r['margin_cm']:.1f}cm  "
              f"tier={r['tier']} ({r['prob_fit']*100:.1f}%)")
        print(f"  detail: {r['detail']}")

    if item and floor:
        resp = requests.post(f"{API_URL}/fit", json={"item": item, "space": floor, "mode": "placement"})
        r = resp.json()
        print(f"\nShoerack on cupboard floor (placement):")
        print(f"  verdict={r['verdict']}  margin={r['margin_cm']:.1f}cm  "
              f"tier={r['tier']} ({r['prob_fit']*100:.1f}%)")
        print(f"  detail: {r['detail']}")

    print()


if __name__ == "__main__":
    main()
