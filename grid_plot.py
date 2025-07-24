import argparse
import glob
import math
import os
import random
import re

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--num_img", type=int, required=True)
    parser.add_argument("--coarse_or_fine", choices=["Coarse", "Fine", "Both"])
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--save_as", default=None)
    parser.add_argument(
        "--cols", default=5, type=int, help="Number of columns in the grid"
    )
    parser.add_argument(
        "--pattern", default="*.png", type=str, help="Glob pattern to filter images"
    )
    parser.add_argument(
        "--labels",
        default=None,
        type=str,
        help="Comma-separated list of labels for images",
    )
    args = parser.parse_args()
    return args


def sort_key(file):
    name = os.path.splitext(os.path.basename(file))[0]
    # Match pattern: prefix_number (e.g., foo_10, foo_0.25)
    match = re.match(r"^(.*)_(\d+(?:\.\d+)?)$", name)
    if match:
        prefix, num = match.groups()
        try:
            return (prefix, float(num))
        except ValueError:
            return (prefix, num)
    # Match pattern: just a number (e.g., 0, 0.25, 10.0)
    match_num = re.match(r"^(\d+(?:\.\d+)?)$", name)
    if match_num:
        try:
            return ("", float(match_num.group(1)))
        except ValueError:
            return ("", match_num.group(1))
    # Fallback: sort by name as string
    return (name, -1)


if __name__ == "__main__":
    args = parse_args()
    DIR = args.dir
    coarse_or_fine = args.coarse_or_fine
    NUM_IMAGES = args.num_img

    # Parse labels if provided
    labels = None
    if args.labels:
        labels = [label.strip() for label in args.labels.split(",")]

    if any(os.path.isdir(os.path.join(DIR, x)) for x in os.listdir(DIR)):
        DIRS = [
            os.path.join(DIR, x)
            for x in os.listdir(DIR)
            if os.path.isdir(os.path.join(DIR, x))
        ]
    else:
        DIRS = [DIR]

    for dirs in DIRS:
        save_dir = (
            args.save_dir
            or f"Grids/{coarse_or_fine}/{os.path.basename(os.path.dirname(os.path.dirname(dirs)))}/{os.path.basename(os.path.dirname(dirs))}"
        )
        os.makedirs(save_dir, exist_ok=True)
        if os.path.exists(f"{save_dir}/{os.path.basename(dirs)}.png"):
            continue
        files = glob.glob(f"{dirs}/{args.pattern}")
        files = random.sample(files, NUM_IMAGES)
        images = []
        files_sorted = sorted(files, key=sort_key)
        for image in files_sorted:
            images.append(Image.open(os.path.join(dirs, f"{image}")))

        cols = args.cols
        rows = math.ceil(NUM_IMAGES / cols)
        fig = plt.figure(figsize=(cols, rows))
        grid = ImageGrid(
            fig,
            111,
            nrows_ncols=(rows, cols),
        )
        for i in range(len(grid)):
            if i >= len(images):
                grid[i].axis("off")
                continue
            grid[i].imshow(images[i])
            grid[i].axis("off")
            if labels and i < len(labels):
                grid[i].text(
                    0.5,
                    -0.05,
                    labels[i],
                    fontsize=6,
                    ha="center",
                    va="top",
                    transform=grid[i].transAxes,
                    wrap=True,
                )

        plt.savefig(
            f"{save_dir}/{args.save_as or os.path.basename(dirs)}.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()
