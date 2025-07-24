#!/bin/env bash

DIR="$PWD/Outputs/5" # Dir should be the directory containing images or the immediate parent directory that contains directories containing images
NUM_IMG=10 # Number of images sampled for grid
COLS=10 # Number of columns in grid
COARSE_OR_FINE="Fine" # Used if save_as/file_name is not given
PATTERN="*.png" # Pattern of images to select for the grid
SAVE_AS="FluxSpace - Human Edits" # File save name
SAVE_DIR="assets" # Directory to store grid
LABELS="" # Labels for the Images

uv run grid_plot.py \
    --dir "$DIR" \
    --num_img $NUM_IMG \
    --cols $COLS \
    --coarse_or_fine "$COARSE_OR_FINE" \
    --pattern "$PATTERN" \
    --save_as "$SAVE_AS" \
    --save_dir "$SAVE_DIR" \
    --labels "$LABELS"
