"""
Interactive click picker for a new photo -- get pixel coordinates for the
measure_item/measure_space contract, with a live SAM mask preview so you can
see whether a click actually lands on what you meant before running the
full pipeline.

    python scripts/pick_points.py data/my_photo.jpg

Click the A4 sheet's 4 corners first (clockwise, any starting corner) --
these are always manual (see measure.py's module docstring for why).
Then click ONCE more: on the item, or on the floor patch you want measured.
For a doorway opening, keep clicking the original 4 points instead
(left_bottom, right_bottom, top, floor_below_top) -- an opening is empty
space, nothing for SAM to click "on".

Close the window when done; coordinates print in a copy-pasteable form for
measure_item(image_path, sheet_px, item_click) / measure_space(...).
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from sam_utils import segment_points  # noqa: E402


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/pick_points.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    img = Image.open(image_path).convert("RGB")
    image_rgb = np.array(img)

    print(f"\nOpening {image_path} ({img.width}x{img.height}px)")
    print("Click 4 A4 sheet corners first (sheet_px, clockwise, any starting corner).")
    print("Then click ONCE on the item, or ONCE on a floor patch -- a live SAM mask")
    print("preview overlays after each click past the 4th so you can check it landed")
    print("on the right thing. For a doorway opening, keep clicking the 4 points")
    print("(left_bottom, right_bottom, top, floor_below_top) instead.")
    print()

    fig, ax = plt.subplots(figsize=(10, 10 * img.height / img.width))
    ax.imshow(img)
    ax.set_title("Click 4 sheet corners, then 1 item/floor click. Close window when done.")

    clicked = []
    mask_overlay = [None]

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return
        x, y = round(event.xdata), round(event.ydata)
        clicked.append((x, y))
        ax.plot(x, y, "o", color="lime", markersize=8)
        ax.annotate(str(len(clicked)), (x, y), color="lime", fontsize=12,
                    xytext=(5, 5), textcoords="offset points")

        if len(clicked) > 4:
            if mask_overlay[0] is not None:
                mask_overlay[0].remove()
            mask = segment_points(image_rgb, [(x, y)])[0]
            mask_overlay[0] = ax.imshow(np.where(mask, 1.0, np.nan), alpha=0.4, cmap="autumn")

        fig.canvas.draw()
        print(f"  point {len(clicked)}: ({x}, {y})")

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

    if not clicked:
        print("\nNo points clicked.")
        return

    sheet_px = clicked[:4]
    rest = clicked[4:]
    print(f"\nsheet_px = {sheet_px}")
    if len(rest) == 1:
        print(f"item_click / floor space_input = {list(rest[0])}")
    elif rest:
        print(f"opening space_input = {[list(p) for p in rest]}")


if __name__ == "__main__":
    main()
