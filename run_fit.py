"""
CLI wrapper around fit.py's placement_fit/passage_fit, loading two saved
measurement results from run_measure.py's --out files.

    python run_fit.py out/shoerack_item.json out/doorframe_space.json --mode passage
    python run_fit.py out/shoerack_item.json out/cupboard_space.json --mode placement
"""

import argparse
import json

from fit import passage_fit, placement_fit


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("item_json", help="path saved by: python run_measure.py item ... --out <this>")
    p.add_argument("space_json", help="path saved by: python run_measure.py space ... --out <this>")
    p.add_argument("--mode", required=True, choices=["placement", "passage"])
    args = p.parse_args()

    item = json.loads(open(args.item_json).read())
    space = json.loads(open(args.space_json).read())

    result = passage_fit(item, space) if args.mode == "passage" else placement_fit(item, space)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
