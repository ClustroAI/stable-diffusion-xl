import torch
from diffusers import AutoPipelineForText2Image

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16, 
    variant="fp16", 
    use_safetensors=True,
    revision=("462165984030d82259a11f4367a4eed129e94a7b")
).to("cuda")

def invoke(input_text):
    image = pipeline(prompt=input_text).images[0]
    image.save('generated_image.png')
    return "generated_image.png"
