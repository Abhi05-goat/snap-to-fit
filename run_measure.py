"""
CLI wrapper around measure_item/measure_space, for pasting in coordinates
from scripts/pick_points.py's output instead of going through the web UI.

Item:
    python run_measure.py item data/shoerack.jpeg \\
        --sheet "[[561,879],[782,1023],[476,1273],[261,1060]]" \\
        --click "[740,400]" \\
        --out out/shoerack_item.json

Space (floor -- 1 click):
    python run_measure.py space data/cupboard_space.jpg --kind floor \\
        --sheet "[[285,745],[620,825],[510,1075],[100,995]]" \\
        --input "[750,700]" \\
        --out out/cupboard_space.json

Space (opening -- 4 manual taps: left_bottom, right_bottom, top, floor_below_top):
    python run_measure.py space data/doorframe.jpg --kind opening \\
        --sheet "[[215,1358],[350,1387],[270,1438],[95,1388]]" \\
        --input "[[140,1295],[605,1345],[390,80],[390,1295]]" \\
        --out out/doorframe_space.json

--out is optional; if given, the full result dict is also saved as JSON so
run_fit.py can load it later without re-running the (slow) depth model.
"""

import argparse
import ast
import json
import sys
from pathlib import Path

from measure import measure_item, measure_space


def parse_points(s: str):
    """Accept either JSON or plain Python list syntax (pick_points.py prints
    the latter -- tuples, no quoted keys)."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return ast.literal_eval(s)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    p_item = sub.add_parser("item")
    p_item.add_argument("image_path")
    p_item.add_argument("--sheet", required=True, help="4 sheet-corner taps, e.g. \"[[x,y],[x,y],[x,y],[x,y]]\"")
    p_item.add_argument("--click", required=True, help="1 click on the item, e.g. \"[x,y]\"")
    p_item.add_argument("--out", default=None, help="optional path to save the result JSON")

    p_space = sub.add_parser("space")
    p_space.add_argument("image_path")
    p_space.add_argument("--kind", required=True, choices=["floor", "opening"])
    p_space.add_argument("--sheet", required=True, help="4 sheet-corner taps")
    p_space.add_argument(
        "--input", required=True,
        help="floor: 1 click \"[x,y]\".  opening: 4 taps \"[[left_bottom],[right_bottom],[top],[floor_below_top]]\""
    )
    p_space.add_argument("--out", default=None)

    args = p.parse_args()

    if args.mode == "item":
        result = measure_item(args.image_path, parse_points(args.sheet), parse_points(args.click))
    else:
        result = measure_space(args.image_path, parse_points(args.sheet), args.kind, parse_points(args.input))

    print(json.dumps(result, indent=2))

    if not result.get("ok"):
        sys.exit(1)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
