import os
import sys
sys.path.append("./")

import torch
from torchvision import transforms
from src.transformer import Transformer2DModel
from src.pipeline import Pipeline
from src.scheduler import Scheduler
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from diffusers import VQModel

device = 'cuda'
dtype = torch.bfloat16
model_path = "Collov-Labs/Monetico"
model = Transformer2DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=dtype)
vq_model = VQModel.from_pretrained(model_path, subfolder="vqvae", torch_dtype=dtype)
text_encoder = CLIPTextModelWithProjection.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=dtype) # better for Monetico
# text_encoder = CLIPTextModelWithProjection.from_pretrained(  #more stable sampling for some cases
#             "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", torch_dtype=dtype
#         )
tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", torch_dtype=dtype)
scheduler = Scheduler.from_pretrained(model_path, subfolder="scheduler", torch_dtype=dtype)
pipe = Pipeline(vq_model, tokenizer=tokenizer, text_encoder=text_encoder, transformer=model, scheduler=scheduler)
pipe.to(device)

steps = 48
CFG = 9
resolution = 512 
negative_prompt = "worst quality, low quality, low res, blurry, distortion, watermark, logo, signature, text, jpeg artifacts, signature, sketch, duplicate, ugly, identifying mark"

prompts = [
    "Two actors are posing for a pictur with one wearing a black and white face paint.",
    "A large body of water with a rock in the middle and mountains in the background.",
    "A white and blue coffee mug with a picture of a man on it.",
    "A statue of a man with a crown on his head.",
    "A man in a yellow wet suit is holding a big black dog in the water.",
    "A white table with a vase of flowers and a cup of coffee on top of it.",
    "A woman stands on a dock in the fog.",
    "A woman is standing next to a picture of another woman."
]

batched_generation = False
num_images = len(prompts) if batched_generation else 1

images = pipe(
    prompt=prompts[:num_images], 
    negative_prompt=[negative_prompt] * num_images,
    height=resolution,
    width=resolution,
    guidance_scale=CFG,
    num_inference_steps=steps
    ).images

output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)
for i, prompt in enumerate(prompts[:num_images]):
    sanitized_prompt = prompt.replace(" ", "_")
    file_path = os.path.join(output_dir, f"{sanitized_prompt}_{resolution}_{steps}_{CFG}.png")
    images[i].save(file_path)
    print(f"The {i+1}/{num_images} image is saved to {file_path}")
