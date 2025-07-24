import argparse
import inspect
import os
import random
import sys

import torch
from custom_modules.transformer_flux import FluxTransformer2DModel
from custom_modules.transformer_sd3 import SD3Transformer2DModel
from diffusers import DiffusionPipeline

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "half": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "float64": torch.float64,
    "double": torch.float64,
}

TRANSFORMER_MAP = {
    "black-forest-labs/FLUX.1-dev": FluxTransformer2DModel,
    "black-forest-labs/FLUX.1-schell": FluxTransformer2DModel,
    "stabilityai/stable-diffusion-3.5-large": SD3Transformer2DModel,
    "stabilityai/stable-diffusion-3.5-medium": SD3Transformer2DModel,
    "stabilityai/stable-diffusion-3-medium-diffusers": SD3Transformer2DModel,
}


def parse_args():
    parser = argparse.ArgumentParser()
    # Model arguments
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="Path to pretrained model or model ID from https://huggingface.co",
    )
    parser.add_argument(
        "--pipeline_path",
        type=str,
        default="pipelines/pipeline_fluxspace_flux.py",
        help="Custom pipeline path for tokenverse",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        required=True,
        help="Huggingface Token to access gated repositories",
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="To offload models that are not currently being used to the CPU. Useful to decrease memory utilization",
    )
    parser.add_argument(
        "--vae_tiling",
        action="store_true",
        help="""Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images""",
    )
    parser.add_argument(
        "--vae_slicing",
        action="store_true",
        help="""Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """,
    )

    # Prompt arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        required=True,
        help="Prompt to guide Image generation",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="Negative prompt to be used for classifier-free guidance",
    )

    # Device and Dtype
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to store tensors on"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float", "float16", "half", "bfloat16", "double"],
        help="Data type of tensors",
    )

    # Fluxspace arguments
    parser.add_argument(
        "--edit_prompt",
        type=str,
        default=None,
        help="Prompt to guide Image edit",
    )
    parser.add_argument(
        "--edit_prompt_embeds_path",
        type=str,
        default=None,
        help="Prompt Embedding of the edit prompt",
    )
    parser.add_argument(
        "--edit_pooled_prompt_embeds_path",
        type=str,
        default=None,
        help="Pooled prompt embedding of the edit prompt",
    )
    parser.add_argument(
        "--edit_token", type=str, default=None, help="Edit token for mask calculation"
    )
    parser.add_argument(
        "--edit",
        choices=["coarse", "fine", "both"],
        required=True,
        help="Choose coarse to only apply Coarse edit, fine to apply only Fine edit, both to apply both edits",
    )
    parser.add_argument(
        "--fine_guidance_scale",
        type=float,
        default=0.0,
        help="Fine Editing Scale for Latents as per FluxSpace",
    )
    parser.add_argument(
        "--coarse_guidance_scale",
        type=float,
        default=0,
        help="Coarse Editing Scale as per FluxSpace",
    )
    parser.add_argument(
        "--thresholding_mask",
        type=float,
        default=0,
        help="Thresholding mask for attention masking edit",
    )
    parser.add_argument(
        "--edit_iteration",
        type=int,
        default=0,
        help="Edit iteration to start applying FluxSpace Guidance",
    )

    # Inference arguments
    parser.add_argument(
        "--save_dir", default=None, type=str, help="Path to store the generated images"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Distilled Guidance Scale",
    )
    parser.add_argument(
        "--true_cfg_scale",
        type=float,
        default=1.0,
        help="When > 1.0 and a provided `negative_prompt`, enables true classifier-free guidance.",
    )
    parser.add_argument(
        "--height", type=int, default=1024, help="Height of image generated"
    )
    parser.add_argument(
        "--width", type=int, default=1024, help="Width of image generated"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference timesteps for diffusion",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Batch inference for N images",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Max sequence length of T5 embeddings. 512 for Flux, 256/77 for SD3.5, 77 for SD3",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for replicatable generations, -1 for random seed",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Prints Inference arguments on console"
    )

    args = parser.parse_args()
    return args


def load(args):
    transformer_model = TRANSFORMER_MAP[args.pretrained_model_name_or_path]

    transformer = transformer_model.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        token=args.token,
        torch_dtype=args.dtype,
        # device_map='balanced'
    )

    pipe = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=transformer,
        custom_pipeline=args.pipeline_path,
        token=args.token,
        torch_dtype=args.dtype,
        # device_map = 'balanced'
    )

    if args.offload:
        # Moves unused models (text_encoders) to RAM for memory optimization as transformer + text_encoders dont fit within 24GB VRAM
        pipe.enable_model_cpu_offload(device=args.device)
    else:
        # If the transformer + text_encoders fit within 24GB VRAM as in the case of SD3.5, keep all tensors on the same GPU
        pipe = pipe.to(args.device)

    if args.vae_tiling:
        pipe.enable_vae_tiling()

    if args.vae_slicing:
        pipe.enable_vae_slicing()

    return pipe


def call_pipe(pipe, **kwargs):
    sig = inspect.signature(pipe.__call__)
    valid_args = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return pipe(**valid_args).images


def main(args):
    seed = args.seed
    if seed == -1:
        seed = random.randint(0, sys.maxsize)
    generator = torch.Generator().manual_seed(seed)

    default_name = "Outputs"
    save_dir = args.save_dir if args.save_dir else default_name
    os.makedirs(save_dir, exist_ok=True)
    if args.edit == "coarse":
        args.fine_guidance_scale = (
            0.0  # Force Coarse only if fine guidance scale provided
        )
        save_path = os.path.join(
            save_dir,
            f"{args.edit_prompt}_e{args.edit_iteration}_c{args.coarse_guidance_scale}",
        )
    elif args.edit == "fine":
        args.coarse_guidance_scale = (
            0  # Force Fine only if coarse guidance scale provided
        )
        save_path = os.path.join(
            save_dir,
            f"{args.edit_prompt}_e{args.edit_iteration}_t{args.thresholding_mask}_f{args.fine_guidance_scale}",
        )
    elif args.edit == "both":
        save_path = os.path.join(
            save_dir,
            f"{args.edit_prompt}_e{args.edit_iteration}_t{args.thresholding_mask}_f{args.fine_guidance_scale}_c{args.coarse_guidance_scale}",
        )
    else:
        print("Invalid Edit choice. Exiting")
        quit()

    pipe = load(args)

    if args.edit_prompt_embeds_path:
        ep = torch.load(args.edit_prompt_embeds_path).to(args.device, dtype=args.dtype)
    else:
        ep = None

    if args.edit_pooled_prompt_embeds_path:
        epp = torch.load(args.edit_pooled_prompt_embeds_path).to(
            args.device, dtype=args.dtype
        )
    else:
        epp = None

    if args.verbose:
        print("Prompt:", args.prompt)
        print("Seed:", args.seed)
        print("Edit Type:", args.edit)
        print("Edit prompt:", args.edit_prompt)
        print("Edit token:", args.edit_token)
        print("Fine guidance scale:", args.fine_guidance_scale)
        print("Coarse guidance scale:", args.coarse_guidance_scale)
        print("Thresholding_mask:", args.thresholding_mask)
        print("Edit iteration:", args.edit_iteration)

    with torch.inference_mode():
        images = call_pipe(
            pipe,
            **vars(args),
            generator=generator,
            edit_prompt_embeds=ep,
            edit_pooled_prompt_embeds=epp,
        )

    for idx, image in enumerate(images):
        image.save(f"{save_path}_{idx}.png")


if __name__ == "__main__":
    args = parse_args()
    assert args.dtype in DTYPE_MAP, (
        f"Invalid dtype '{args.dtype}' provided. Choose from: {list(DTYPE_MAP.keys())}"
    )
    args.dtype = DTYPE_MAP[args.dtype]
    main(args)
