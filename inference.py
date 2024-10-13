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

model_path = "MeissonFlow/Meissonic"
model = Transformer2DModel.from_pretrained(model_path,subfolder="transformer",)
vq_model = VQModel.from_pretrained(model_path, subfolder="vqvae", )
text_encoder = CLIPTextModelWithProjection.from_pretrained(model_path,subfolder="text_encoder",)
tokenizer = CLIPTokenizer.from_pretrained(model_path,subfolder="tokenizer",)
scheduler = Scheduler.from_pretrained(model_path,subfolder="scheduler",)
pipe=Pipeline(vq_model, tokenizer=tokenizer,text_encoder=text_encoder,transformer=model,scheduler=scheduler)

pipe = pipe.to(device)

steps = 48
CFG = 9
resolution = 1024 
negative_prompts = "worst quality, normal quality, low quality, low res, blurry, distortion, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch, duplicate, ugly, monochrome, horror, geometry, mutation, disgusting, bad anatomy, bad proportions, bad quality, deformed, disconnected limbs, out of frame, out of focus, dehydrated, disfigured, extra arms, extra limbs, extra hands, fused fingers, gross proportions, long neck, jpeg, malformed limbs, mutated, mutated hands, mutated limbs, missing arms, missing fingers, picture frame, poorly drawn hands, poorly drawn face, collage, pixel, pixelated, grainy, color aberration, amputee, autograph, bad illustration, beyond the borders, blank background, body out of frame, boring background, branding, cut off, dismembered, disproportioned, distorted, draft, duplicated features, extra fingers, extra legs, fault, flaw, grains, hazy, identifying mark, improper scale, incorrect physiology, incorrect ratio, indistinct, kitsch, low resolution"

prompt = "a beautiful landscape painting with a river and mountains in the background"


image = pipe(prompt=prompt,negative_prompt=negative_prompts,height=resolution,width=resolution,guidance_scale=CFG,num_inference_steps=steps).images[0]

output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)
image.save(output_dir, f"{prompt[:10]}_{resolution}_{steps}_{CFG}.png")

