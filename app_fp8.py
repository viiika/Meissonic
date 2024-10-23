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
import gradio as gr
import time
from torchao.quantization.quant_api import (
    quantize_,
    float8_weight_only,
)

device = 'cuda'

def get_quantization_method(method):
    quantization_methods = {
        'fp8': lambda: float8_weight_only(),
        'none': None
    }
    return quantization_methods.get(method, None)

def load_models(quantization_method='none'):
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
    
    if quantization_method != 'none':
        quant_method = get_quantization_method(quantization_method)
        if quant_method:
            quantize_(model, quant_method())

    pipe = Pipeline(vq_model, tokenizer=tokenizer, text_encoder=text_encoder, transformer=model, scheduler=scheduler)
    return pipe.to(device)

# Global variable to store the pipeline
global_pipe = None
current_quantization = 'none'

def initialize_pipeline(quantization):
    global global_pipe, current_quantization
    if global_pipe is None or current_quantization != quantization:
        global_pipe = load_models(quantization)
        current_quantization = quantization
    return global_pipe

def generate_images(prompt, negative_prompt, seed, randomize_seed, width, height, 
                   guidance_scale, num_inference_steps, quantization_method, batch_size=1, 
                   progress=gr.Progress(track_tqdm=True)):
    if randomize_seed or seed == 0:
        seed = torch.randint(0, MAX_SEED, (1,)).item()
    torch.manual_seed(seed)
    
    # Initialize or update pipeline if needed
    pipe = initialize_pipeline(quantization_method)
    
    # Reset CUDA memory stats
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    # Handle batch generation
    if isinstance(prompt, str):
        prompts = [prompt] * batch_size
    else:
        prompts = prompt[:batch_size]
    
    images = pipe(
        prompt=prompts,
        negative_prompt=[negative_prompt] * batch_size,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    ).images
    
    # Calculate performance metrics
    inference_time = time.time() - start_time
    memory_used = torch.cuda.max_memory_reserved() / (1024 ** 3)  # Convert to GB
    
    performance_info = f"""
    Inference Time: {inference_time:.2f} seconds
    Memory Used: {memory_used:.2f} GB
    Quantization: {quantization_method}
    """
    
    return images[0] if batch_size == 1 else images, seed, performance_info

MAX_SEED = 2**32 - 1
MAX_IMAGE_SIZE = 1024
default_negative_prompt = "worst quality, low quality, low res, blurry, distortion, watermark, logo, signature, text, jpeg artifacts, signature, sketch, duplicate, ugly, identifying mark"

examples = [
    "Two actors are posing for a pictur with one wearing a black and white face paint.",
    "A large body of water with a rock in the middle and mountains in the background.",
    "A white and blue coffee mug with a picture of a man on it.",
    "The sun is setting over a city skyline with a river in the foreground.",
    "A black and white cat with blue eyes.", 
    "Three boats in the ocean with a rainbow in the sky.", 
    "A robot playing the piano.",
    "A cat wearing a hat.",
    "A dog in a jungle."
]

css = """
#col-container {
    margin: 0 auto;
    max-width: 640px;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# Meissonic Text-to-Image Generator (with FP8 Support)")
        
        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            run_button = gr.Button("Run", scale=0, variant="primary")
        
        result = gr.Image(label="Result", show_label=False)
        performance_info = gr.Textbox(label="Performance Metrics", lines=4)
        
        with gr.Accordion("Advanced Settings", open=False):
            quantization = gr.Radio(
                choices=['none', 'fp8'],
                value='none',
                label="Quantization Method",
            )
            negative_prompt = gr.Text(
                label="Negative prompt",
                max_lines=1,
                placeholder="Enter a negative prompt",
                value=default_negative_prompt,
            )
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            
            with gr.Row():
                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                )
                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                )
            
            with gr.Row():
                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=0.0,
                    maximum=20.0,
                    step=0.1,
                    value=9.0,
                )
                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=64,
                )
            
            batch_size = gr.Slider(
                label="Batch Size",
                minimum=1,
                maximum=8,
                step=1,
                value=1,
            )
        
        gr.Examples(examples=examples, inputs=[prompt])
    
    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=generate_images,
        inputs=[
            prompt,
            negative_prompt,
            seed,
            randomize_seed,
            width,
            height,
            guidance_scale,
            num_inference_steps,
            quantization,
            batch_size,
        ],
        outputs=[result, seed, performance_info],
    )

demo.launch()
