#!/bin/env bash

PRETRAINED_MODEL_NAME_OR_PATH="black-forest-labs/FLUX.1-dev" # Huggingface Model Name or Path
PIPELINE_PATH="pipelines/pipeline_fluxspace_flux.py" # Custom Pipeline Path - Or pipelines/pipeline_fluxspace_sd3.py
TOKEN="<Insert Huggingface token here>" # Huggingface Access Token

PROMPT="Portrait photo of a man" # Prompt to guide generation
EDIT_PROMPT="eyeglasses" # Edit Prompt to guide editing
EDIT_TOKEN="eyeglasses" # Edit token for Fine FluxSpace Editing
EDIT="both" # Edit Type - Coarse, Fine, Both

DTYPE="bfloat16" # Tensor Data Type

COARSE_GUIDANCE_SCALE=0.5 # Coarse Editing Scale
FINE_GUIDANCE_SCALE=5.0 # Fine Editing Scale
THRESHOLDING_MASK=0.5 # Threshold for calculating mask
EDIT_ITERATION=1 # Edit iteration for applying fine FluxSpace Editing

SAVE_DIR="Outputs/" # Save Directory
HEIGHT=1024 # Image Height
WIDTH=1024 # Image Width
GUIDANCE_SCALE=3.5 # Guidance Scale for generation
NUM_INFERENCE_STEPS=30 # Number of inference Steps
NUM_IMAGES_PER_PROMPT=1 # Number of images to be generated per prompt
SEED=0 # Seed to guide generation
MAX_SEQUENCE_LENGTH=512 # Sequence length of text tensor

uv run inference.py \
    --pretrained_model_name_or_path "$PRETRAINED_MODEL_NAME_OR_PATH" \
    --pipeline_path "$PIPELINE_PATH" \
    --prompt "$PROMPT" \
    --token "$TOKEN" \
    --edit_prompt "$EDIT_PROMPT" \
    --edit_token "$EDIT_TOKEN" \
    --edit "$EDIT" \
    --dtype "$DTYPE" \
    --coarse_guidance_scale $COARSE_GUIDANCE_SCALE \
    --fine_guidance_scale $FINE_GUIDANCE_SCALE \
    --thresholding_mask $THRESHOLDING_MASK \
    --edit_iteration $EDIT_ITERATION \
    --save_dir "$SAVE_DIR" \
    --height $HEIGHT \
    --width $WIDTH \
    --guidance_scale $GUIDANCE_SCALE \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --num_images_per_prompt $NUM_IMAGES_PER_PROMPT \
    --max_sequence_length $MAX_SEQUENCE_LENGTH \
    --seed $SEED \
    --offload 

