import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

# load both base & refiner
base = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

refiner = StableDiffusionXLImg2ImgPipeline.from_single_file(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
).to("cuda")

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 40
high_noise_frac = 0.8

def invoke(input_text):
    image = base(
        prompt=input_text,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
    image = refiner(
        prompt=input_text,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]
    image.save('generated_image.png')
    return "generated_image.png"
