import os
import sys
sys.path.append("./")

import torch
from src.transformer import Transformer2DModel
from src.pipeline import Pipeline
from src.scheduler import Scheduler
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from diffusers import VQModel
import time
import argparse

from torchao.quantization.quant_api import (
    quantize_,
    float8_weight_only, # A8W8 FP8
)

device = 'cuda'

def get_quantization_method(method):
    quantization_methods = {
        'fp8': lambda: float8_weight_only(),
    }
    return quantization_methods.get(method, None)

def load_models(quantization_method=None):
    model_path = "MeissonFlow/Meissonic"
    dtype = torch.float16
    model = Transformer2DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=dtype)
    vq_model = VQModel.from_pretrained(model_path, subfolder="vqvae", torch_dtype=dtype)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        torch_dtype=dtype
    )
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    scheduler = Scheduler.from_pretrained(model_path, subfolder="scheduler")
    
    if quantization_method:
        quant_method = get_quantization_method(quantization_method)
        if quant_method:
            quantize_(model, quant_method())
        else:
            print(f"Unsupported quantization method: {quantization_method}")

    
    pipe = Pipeline(vq_model, tokenizer=tokenizer, text_encoder=text_encoder, transformer=model, scheduler=scheduler)
    return pipe.to(device)

def run_inference(pipe, prompt, negative_prompt, resolution, cfg, steps):
    return pipe(prompt=prompt, negative_prompt=negative_prompt, height=resolution, width=resolution, guidance_scale=cfg, num_inference_steps=steps).images[0]

def main(quantization_method):
    steps = 64
    CFG = 9
    resolution = 1024 
    negative_prompts = "worst quality, low quality, low res, blurry, distortion, watermark, logo, signature, text, jpeg artifacts, signature, sketch, duplicate, ugly, identifying mark"

    prompts = [
        "Two actors are posing for a pictur with one wearing a black and white face paint.",
        "A large body of water with a rock in the middle and mountains in the background.",
        "A white and blue coffee mug with a picture of a man on it.",
        "The sun is setting over a city skyline with a river in the foreground.",
        "A black and white cat with blue eyes.", 
        "Three boats in the ocean with a rainbow in the sky.", 
        "A robot playing the piano.",
        "A cat wearing a hat.",
        "A dog in a jungle.",
    ]

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    pipe = load_models(quantization_method)
    start_time = time.time()
    total_memory_used = 0
    for i, prompt in enumerate(prompts):
        torch.cuda.reset_peak_memory_stats()
        image_start_time = time.time()
        image = run_inference(pipe, prompt, negative_prompts, resolution, CFG, steps)
        image_end_time = time.time()
        image.save(os.path.join(output_dir, f"{prompt[:10]}_{resolution}_{steps}_{CFG}_{quantization_method}.png"))
        
        memory_used = torch.cuda.max_memory_reserved() / (1024 ** 3)  # Convert to GB
        total_memory_used += memory_used
        
        print(f"Image {i+1} time: {image_end_time - image_start_time:.2f} seconds")
        print(f"Image {i+1} max memory used: {memory_used:.2f} GB")
    
    total_time = time.time() - start_time
    avg_memory_used = total_memory_used / len(prompts)
    print(f"Total inference time ({quantization_method}): {total_time:.2f} seconds")
    print(f"Average memory used per image: {avg_memory_used:.2f} GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with specified quantization method.")
    parser.add_argument("--quantization", type=str, choices=['fp8'], 
                        help="Quantization method to use")
    args = parser.parse_args()
    main(args.quantization)
