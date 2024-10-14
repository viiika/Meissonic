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


# A racoon wearing a suit smoking a cigar in the style of James Gurney.
# Medieval painting of a rat king.
# Oil portrait of Super Mario as a shaman tripping on mushrooms in a dark and detailed scene.
# A painting of a Persian cat dressed as a Renaissance king, standing on a skyscraper overlooking a city.
# A fluffy owl sits atop a stack of antique books in a detailed and moody illustration.
# A cosmonaut otter poses for a portrait painted in intricate detail by Rembrandt.
# A painting featuring a woman wearing virtual reality glasses and a bird, created by Dave McKean and Ivan Shishkin.
# A hyperrealist portrait of a fairy girl emperor wearing a crown and long starry robes.
# A psychedelic painting of a fantasy space whale.
# A monkey in a blue top hat painted in oil by Vincent van Gogh in the 1800s.
# A queen with red hair and a green and black dress stands veiled in a highly detailed and elegant digital painting.
# An oil painting of an anthropomorphic fox overlooking a village in the moor.
# A digital painting of an evil geisha in a bar.
# Digital painting of a furry deer character on FurAffinity.
# A highly detailed goddess portrait with a focus on the eyes.
# A cute young demon princess in a forest, depicted in digital painting.
# A red-haired queen wearing a green and black dress and veil is depicted in an intricate and elegant digital painting.
prompt = "A racoon wearing a suit smoking a cigar in the style of James Gurney."

image = pipe(prompt=prompt,negative_prompt=negative_prompts,height=resolution,width=resolution,guidance_scale=CFG,num_inference_steps=steps).images[0]

output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)
image.save(output_dir, f"{prompt[:10]}_{resolution}_{steps}_{CFG}.png")

