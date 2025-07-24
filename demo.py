import random
import sys
from dataclasses import dataclass

import gradio as gr
import requests
import torch
from diffusers.utils import load_image
from inference import DTYPE_MAP, call_pipe, load

cached_pipe = None
cached_inv_latents = None
cached_img_latents = None
cached_img_ids = None
old_args = None
old_image = None


@dataclass(frozen=True)
class PipeArgs:
    pretrained_model_name_or_path: str
    pipeline_path: str
    token: str
    dtype: torch.dtype
    device: torch.device
    offload: bool
    vae_tiling: bool
    vae_slicing: bool


def generate_image(
    pretrained_model_name_or_path,
    pipeline_path,
    token,
    prompt,
    negative_prompt,
    real_text,
    real_image,
    inversion_steps,
    gammas,
    start_t,
    stop_t,
    eta,
    edit_prompt,
    edit_tokens,
    true_cfg_scale,
    guidance_scale,
    fluxspace_coarse_scale,
    fluxspace_fine_scale,
    fluxspace_threshold_mask,
    fluxspace_edit_iteration,
    height,
    width,
    num_inference_steps,
    num_images_per_prompt,
    max_sequence_length,
    dtype,
    seed,
    offload,
    vae_tiling,
    vae_slicing,
    save_path,
    edit_type,
):
    global \
        cached_pipe, \
        cached_inv_latents, \
        cached_img_latents, \
        cached_img_ids, \
        old_args, \
        old_image

    device = "cuda" if torch.cuda.is_available() else "cpu"

    valid = validate_hf_token(pretrained_model_name_or_path, token)

    if valid is False:
        return

    load_args = PipeArgs(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        pipeline_path=pipeline_path,
        token=token,
        dtype=DTYPE_MAP[dtype],
        device=device,
        offload=offload,
        vae_tiling=vae_tiling,
        vae_slicing=vae_slicing,
    )

    if old_args != load_args:
        pipe = load(load_args)
    else:
        pipe = cached_pipe

    old_args = load_args
    cached_pipe = pipe

    if edit_type == "Coarse":
        fluxspace_fine_scale = 0
        fluxspace_edit_iteration = 0
        fluxspace_threshold_mask = 0
    elif edit_type == "Fine":
        fluxspace_coarse_scale = 0
    # else:
    # Use both

    inference_args = {
        "prompt": prompt,
        "edit_prompt": edit_prompt,
        "edit_token": edit_tokens,
        "negative_prompt": negative_prompt,
        "guidance_scale": guidance_scale,
        "true_cfg_scale": true_cfg_scale,
        "coarse_guidance_scale": fluxspace_coarse_scale,
        "fine_guidance_scale": fluxspace_fine_scale,
        "thresholding_mask": fluxspace_threshold_mask,
        "edit_iteration": fluxspace_edit_iteration,
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
        "num_images_per_prompt": num_images_per_prompt,
        "max_sequence_length": int(max_sequence_length),
        "seed": seed,
        "save_path": save_path,
    }

    generator = torch.Generator().manual_seed(
        random.randint(0, sys.maxsize) if seed == -1 else seed
    )

    if real_text == "Image":
        if old_image is None or old_image != real_image:
            old_image = real_image
            real_image = load_image(real_image)
            with torch.inference_mode():
                inverted_latents, image_latents, latent_image_ids = pipe.invert(
                    image=real_image,
                    num_inversion_steps=inversion_steps,
                    gamma=gammas,
                )
            cached_inv_latents = inverted_latents
            cached_img_latents = image_latents
            cached_img_ids = latent_image_ids
        else:
            inverted_latents = cached_inv_latents
            image_latents = cached_img_latents
            latent_image_ids = cached_img_ids

        inference_args["inverted_latents"] = inverted_latents
        inference_args["image_latents"] = image_latents
        inference_args["latent_image_ids"] = latent_image_ids

        inference_args["start_timestep"] = start_t
        inference_args["stop_timestep"] = stop_t
        inference_args["eta"] = eta

        print("Inversion Done!")
        if offload:
            pipe.enable_model_cpu_offload(device=device)
    else:
        inference_args["inverted_latents"] = None
        inference_args["image_latents"] = None
        inference_args["latent_image_ids"] = None

    with torch.inference_mode():
        result = call_pipe(pipe, **inference_args, generator=generator)

    if inference_args["save_path"] is not None:
        for idx, image in enumerate(result):
            image.save(f"{inference_args['save_path']}_{idx}.png")

    return result if result else []


def toggle_fields(choice):
    if choice == "Fine":
        return (
            gr.update(visible=False),  # fluxspace_coarse
            gr.update(visible=True),  # fluxspace_fine_group
        )
    elif choice == "Coarse":
        return (
            gr.update(visible=True),
            gr.update(visible=False),
        )
    else:  # Both
        return (
            gr.update(visible=True),
            gr.update(visible=True),
        )


def toggle_edit_mode(choice):
    if choice == "Image":
        return (
            gr.update(visible=True),  # real_image_edit
            gr.update(visible=True),  # real_image_advanced_edit
        )
    else:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
        )


def validate_hf_token(pretrained_model_name_or_path, token):
    if not token.strip():
        gr.Warning("❌ Please enter a Hugging Face token.")
        return False

    headers = {"Authorization": f"Bearer {token.strip()}"}
    model_url = f"https://huggingface.co/{pretrained_model_name_or_path}/resolve/main/model_index.json"

    try:
        response = requests.get(model_url, headers=headers)

        if response.status_code == 200:
            gr.Info("✅ Token is valid and has access to the model.")
        elif response.status_code == 401:
            gr.Warning("❌ Invalid or expired Hugging Face token.")
        elif response.status_code == 403:
            gr.Warning("❌ Token is valid but access is denied (gated model).")
        else:
            gr.Warning(f"⚠️ Unexpected error: {response.status_code} - {response.text}")

    except Exception as e:
        gr.Warning(f"⚠️ Error: {str(e)}")


def display_img(image):
    return gr.update(value=image, visible=True)


css = """
#col-container {
    margin: 0 auto;
    max-width: 1008px;
}
"""
with gr.Blocks(title="FluxSpace", theme=gr.themes.Soft(), css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("""
            # [FluxSpace](https://fluxspace.github.io): Disentangled Semantic Editing in Rectified Flow Transformers
            ## Edit images using FLUX / SD3 pipelines 
        """)

        with gr.Row():
            with gr.Column(scale=1):
                real_text = gr.Radio(
                    choices=["Image", "Text"], label="Image/Text Edit", value="Text"
                )

                with gr.Column(visible=False) as real_image_edit:
                    real_image = gr.File(
                        label="Image Input", file_types=["image"], file_count="single"
                    )
                    img_display = gr.Image(label="Your Image", visible=False)
                    inversion_steps = gr.Slider(
                        label="Inversion Steps", minimum=1, maximum=1000, value=30
                    )

                real_image.upload(
                    fn=display_img, inputs=real_image, outputs=[img_display]
                )

                with gr.Accordion("Advanced setttings", open=False):
                    pretrained_model = gr.Textbox(
                        label="Pretrained Model Name or Path",
                        lines=2,
                        value="black-forest-labs/FLUX.1-dev",
                    )
                    pipeline_path = gr.Textbox(
                        label="Pipeline Path",
                        lines=2,
                        value="pipelines/pipeline_fluxspace_flux.py",
                    )
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        lines=4,
                        placeholder="Leave empty if you dont want Classifier Free Guidance as per FLUX",
                    )
                    true_cfg_scale = gr.Slider(
                        label="Classifier Free Guidance Scale (only if >1.0)",
                        minimum=1.0,
                        maximum=10.0,
                        value=1.0,
                    )
                    guidance_scale = gr.Slider(
                        label="Guidance Scale", minimum=0.0, maximum=10.0, value=3.5
                    )
                    height = gr.Slider(
                        label="Height", minimum=256, maximum=1024, step=64, value=1024
                    )
                    width = gr.Slider(
                        label="Width", minimum=256, maximum=1024, step=64, value=1024
                    )
                    inference_steps = gr.Slider(
                        label="Inference Steps",
                        minimum=1,
                        maximum=1000,
                        step=5,
                        value=30,
                    )
                    seed = gr.Number(label="Seed (-1 for random)", value=0)
                    images_per_prompt = gr.Slider(
                        label="Images per Prompt",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=1,
                    )
                    max_seq_len = gr.Dropdown(
                        label="Max Sequence Length", choices=[77, 256, 512], value=512
                    )
                    dtype = gr.Radio(
                        ["float32", "float16", "bfloat16", "double"],
                        label="Data Type",
                        value="bfloat16",
                    )

                    with gr.Column(visible=False) as real_image_advanced_edit:
                        start_t = gr.Slider(
                            label="Start timestep", minimum=0, maximum=1, value=0
                        )
                        stop_t = gr.Slider(
                            label="Stop timestep", minimum=0, maximum=1, value=0.25
                        )
                        eta = gr.Slider(
                            label="Controller Guidance", minimum=0, maximum=1, value=0.9
                        )
                        gammas = gr.Slider(
                            label="Gammas", minimum=0, maximum=1, value=0.5
                        )

                    with gr.Row():
                        offload = gr.Checkbox(label="Enable CPU Offload", value=True)
                        vae_tiling = gr.Checkbox(label="Enable Vae Tiling", value=True)
                        vae_slicing = gr.Checkbox(
                            label="Enable Vae Slicing", value=True
                        )

                hf_token = gr.Textbox(
                    label="Huggingface Token",
                    placeholder="hf_....",
                    type="password",
                    lines=1,
                )

                prompt = gr.Textbox(
                    label="Prompt", value="An image of a person", lines=2
                )
                edit_prompt = gr.Textbox(label="Edit Prompt", value="male person")

                edit_type = gr.Radio(
                    choices=["Coarse", "Fine", "Both"],
                    label="Edit Type",
                    value="Coarse",
                )

                fluxspace_coarse = gr.Slider(
                    label="FluxSpace Coarse Scale",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.5,
                    visible=True,
                )
                fluxspace_edit_iteration = gr.Slider(
                    label="FluxSpace Edit Iteration",
                    minimum=0,
                    maximum=50,
                    step=1,
                    value=3,
                )

                with gr.Column(visible=False) as fine_group:
                    edit_token = gr.Textbox(label="Edit Token", value="person")
                    fluxspace_fine = gr.Slider(
                        label="FluxSpace Fine Scale",
                        minimum=0.0,
                        maximum=10.0,
                        step=0.1,
                        value=3.0,
                    )
                    fluxspace_threshold_mask = gr.Slider(
                        label="FluxSpace Fine Thresholding Mask",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=0.5,
                    )

                edit_type.change(
                    fn=toggle_fields,
                    inputs=edit_type,
                    outputs=[fluxspace_coarse, fine_group],
                )

                real_text.change(
                    fn=toggle_edit_mode,
                    inputs=real_text,
                    outputs=[real_image_edit, real_image_advanced_edit],
                )

                save_path = gr.Textbox(label="Save path", value="Outputs/output")

            with gr.Column(scale=1):
                gallery = gr.Gallery(label="Generated Images")
                run_btn = gr.Button("Generate")

                with gr.Row():
                    gr.Image(
                        value="Outputs/4/sunglasses_e1_t0.5_f5_c0.5_0.png",
                        label="Sunglasses Edit",
                    )
                    gr.Image(
                        value="Outputs/6/car_red_e2_t0.95_f5_c0_0.png",
                        label="Red Car Edit",
                    )
                with gr.Row():
                    gr.Markdown("*Prompt: Portrait photo of a man → Edit: sunglasses*")
                    gr.Markdown("*Prompt: An image of a car → Edit: red*")

                with gr.Row():
                    gr.Image(value="Outputs/5/3D_Cartoon_Style_e2_t0_f10_c1_0.png")
                    gr.Image(
                        value="Outputs/7/jungle_scenery_e5_t0.5_f8_c0.3_0.png",
                        label="Jungle Edit",
                    )

                with gr.Row():
                    gr.Markdown(
                        "*Prompt: Portrait photo of a man → Edit: 3D Cartoon Style*"
                    )
                    gr.Markdown("*Prompt: An image of a scenery → Edit: jungle*")

        gr.Examples(
            examples=[
                [
                    "Portrait photo of a man",
                    "sunglasses",
                    "sunglasses",
                    0.5,
                    5.0,
                    1,
                    0.5,
                    "Both",
                ],
                ["An image of a car", "red", "red", 0.0, 5.0, 2, 0.95, "Both"],
                [
                    "Portrait photo of a man",
                    "3D Cartoon Style",
                    "3D",
                    1,
                    10,
                    2,
                    0,
                    "Both",
                ],
                ["An image of a scenery", "jungle", "jungle", 0.5, 8.0, 5, 0.3, "Both"],
            ],
            inputs=[
                prompt,
                edit_prompt,
                edit_token,
                fluxspace_coarse,
                fluxspace_fine,
                fluxspace_edit_iteration,
                fluxspace_threshold_mask,
                edit_type,
            ],
            label="Text-Based Edit Examples, Select to generate one of the above examples",
        )

    run_btn.click(
        fn=generate_image,
        inputs=[
            pretrained_model,
            pipeline_path,
            hf_token,
            prompt,
            negative_prompt,
            real_text,
            real_image,
            inversion_steps,
            gammas,
            start_t,
            stop_t,
            eta,
            edit_prompt,
            edit_token,
            true_cfg_scale,
            guidance_scale,
            fluxspace_coarse,
            fluxspace_fine,
            fluxspace_threshold_mask,
            fluxspace_edit_iteration,
            height,
            width,
            inference_steps,
            images_per_prompt,
            max_seq_len,
            dtype,
            seed,
            offload,
            vae_tiling,
            vae_slicing,
            save_path,
            edit_type,
        ],
        outputs=gallery,
    )

if __name__ == "__main__":
    demo.launch(share=False)
