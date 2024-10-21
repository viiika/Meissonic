# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import subprocess
import time
import torch
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from diffusers import VQModel
from cog import BasePredictor, Input, Path

from src.transformer import Transformer2DModel
from src.pipeline import Pipeline
from src.scheduler import Scheduler


MODEL_CACHE = "model_cache"
MODEL_URL = (
    f"https://weights.replicate.delivery/default/viiika/Meissonic/{MODEL_CACHE}.tar"
)

os.environ.update(
    {
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": MODEL_CACHE,
        "TORCH_HOME": MODEL_CACHE,
        "HF_DATASETS_CACHE": MODEL_CACHE,
        "TRANSFORMERS_CACHE": MODEL_CACHE,
        "HUGGINGFACE_HUB_CACHE": MODEL_CACHE,
    }
)


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        model_path = f"{MODEL_CACHE}/MeissonFlow/Meissonic"
        model = Transformer2DModel.from_pretrained(model_path, subfolder="transformer")
        vq_model = VQModel.from_pretrained(model_path, subfolder="vqvae")
        text_encoder = CLIPTextModelWithProjection.from_pretrained(  # more stable sampling for some cases
            f"{MODEL_CACHE}/laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        )
        tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        scheduler = Scheduler.from_pretrained(model_path, subfolder="scheduler")
        self.pipe = Pipeline(
            vq_model,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=model,
            scheduler=scheduler,
        ).to("cuda")

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="a photo of an astronaut riding a horse on mars",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default="worst quality, low quality, low res, blurry, distortion, watermark, logo, signature, text, jpeg artifacts, signature, sketch, duplicate, ugly, identifying mark",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=100, default=64
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=0, le=20, default=9
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        torch.manual_seed(seed)

        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=1024,
            width=1024,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        ).images[0]
        output_path = f"/tmp/out.png"
        image.save(output_path)
        return Path(output_path)
