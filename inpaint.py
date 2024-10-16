import os
import sys
sys.path.append("./")

import argparse
import json
from PIL import Image
from src.transformer import Transformer2DModel
from src.pipeline_inpaint import InpaintPipeline
from src.scheduler import Scheduler
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from diffusers import VQModel

def get_parse_args():
    parser = argparse.ArgumentParser(description="Meissonic Inpaint and Outpaint")
    parser.add_argument("--mode", type=str,default="inpaint", choices=["inpaint", "outpaint"], help="Inpaint or Outpaint")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_parse_args()
    device = 'cuda'

    model_path = "MeissonFlow/Meissonic"
    model = Transformer2DModel.from_pretrained(model_path, subfolder="transformer", )
    vq_model = VQModel.from_pretrained(model_path, subfolder="vqvae", )
    # text_encoder = CLIPTextModelWithProjection.from_pretrained(model_path,subfolder="text_encoder",)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(  # using original text enc for stable sampling
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    )
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", )
    scheduler = Scheduler.from_pretrained(model_path, subfolder="scheduler", )
    
    pipe=InpaintPipeline(vq_model, tokenizer=tokenizer,text_encoder=text_encoder,transformer=model,scheduler=scheduler)
    pipe = pipe.to(device)

    with open(f"./assets/{args.mode}/cases.json", 'r', encoding='utf-8') as file:
        cases = json.load(file)
    item = cases[0]

    steps = 64
    CFG = 9
    resolution = 1024
    negative_prompts = item["negative_prompts"] if "negative_prompts" in item.keys() else "worst quality, low quality, low res, blurry, distortion, watermark, logo, signature, text, jpeg artifacts, signature, sketch, duplicate, ugly, identifying mark"

    image = Image.open(item["input"]).resize((resolution, resolution)).convert("RGB")
    mask = Image.open(item["mask"]).resize((resolution, resolution)).convert("RGB")

    image = pipe(prompt=item["prompt"],negative_prompt=negative_prompts,image =image, mask_image =mask, guidance_scale=CFG, num_inference_steps=steps).images[0]

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    image.save(os.path.join(output_dir, f"{item['prompt'][:10]}_{resolution}_{steps}_{CFG}.png"))