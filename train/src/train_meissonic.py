# Copyright 2024 The HuggingFace Team and The MeissonFlow Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import logging
import math
import os
import shutil
from contextlib import nullcontext
from pathlib import Path


import sys
sys.path.append(os.getcwd())


# export HF_ENDPOINT=https://hf-mirror.com

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torch.utils.data import DataLoader, Dataset, default_collate
from torchvision import transforms
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

import diffusers.optimization
from diffusers import EMAModel, VQModel 
from src.scheduler import Scheduler
from diffusers.loaders import LoraLoaderMixin
from diffusers.utils import is_wandb_available

from trainer_utils import save_checkpoint
from dataset_utils import HuggingFaceDataset, MSCOCO600KDataset, PickaPicV2Dataset, MyParquetDataset
from dataset_utils import tokenize_prompt, encode_prompt

from torchvision.utils import save_image,make_grid

from src.transformer import Transformer2DModel
from src.pipeline import Pipeline

if is_wandb_available():
    import wandb
    # wandb.login(key="")

logger = get_logger(__name__, log_level="INFO")

import torch._dynamo
torch._dynamo.config.verbose = True

# Optionally suppress errors to fall back to eager execution
torch._dynamo.config.suppress_errors = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_architecture",
        type=str,
        default="Meissonic",
        required=False
    )
    parser.add_argument(
        "--text_encoder_architecture",
        type=str,
        default="open_clip",
        required=False,
        help="The architecture of the text encoder. One of ['CLIP', 'open_clip', 'flan-t5-base','Qwen2-0.5B','gemini-2b',long_CLIP_T5_base','CLIP_T5_base']",
    )
    parser.add_argument(
        "--instance_dataset",
        type=str,
        default=None,
        required=False,
        help="The dataset to use for training. One of ['MSCOCO600K', 'PickaPicV2']",
    )
    parser.add_argument(
        "--training_from_scratch",
        type=bool,
        default=False,
        required=False
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--instance_data_dataset",
        type=str,
        default=None,
        required=False,
        help="A Hugging Face dataset containing the training images",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--instance_data_image", type=str, default=None, required=False, help="A single training image"
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--ema_update_after_step", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="muse_training",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more details"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0003,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--validation_prompts", type=str, nargs="*")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument("--split_vae_encode", type=int, required=False, default=None)
    parser.add_argument("--min_masking_rate", type=float, default=0.0)
    parser.add_argument("--cond_dropout_prob", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", default=50.0, type=float, help="Max gradient norm.", required=False)
    parser.add_argument("--use_lora", action="store_true", help="Fine tune the model using LoRa")
    parser.add_argument("--text_encoder_use_lora", action="store_true", help="Fine tune the model using LoRa")
    parser.add_argument("--lora_r", default=16, type=int)
    parser.add_argument("--lora_alpha", default=32, type=int)
    parser.add_argument("--lora_target_modules", default=["to_q", "to_k", "to_v"], type=str, nargs="+")
    parser.add_argument("--text_encoder_lora_r", default=16, type=int)
    parser.add_argument("--text_encoder_lora_alpha", default=32, type=int)
    parser.add_argument("--text_encoder_lora_target_modules", default=["to_q", "to_k", "to_v"], type=str, nargs="+")
    parser.add_argument("--train_text_encoder", action="store_true")
    parser.add_argument("--image_key", type=str, required=False)
    parser.add_argument("--prompt_key", type=str, required=False)
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--prompt_prefix", type=str, required=False, default=None)

    args = parser.parse_args()

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    num_datasources = sum(
        [x is not None for x in [args.instance_data_dir, args.instance_data_image, args.instance_data_dataset]]
    )

    if num_datasources != 1:
        raise ValueError(
            "provide one and only one of `--instance_data_dir`, `--instance_data_image`, or `--instance_data_dataset`"
        )

    if args.instance_data_dir is not None:
        if not os.path.exists(args.instance_data_dir):
            raise ValueError(f"Does not exist: `--args.instance_data_dir` {args.instance_data_dir}")

    if args.instance_data_image is not None:
        if not os.path.exists(args.instance_data_image):
            raise ValueError(f"Does not exist: `--args.instance_data_image` {args.instance_data_image}")

    if args.instance_data_dataset is not None and (args.image_key is None or args.prompt_key is None):
        raise ValueError("`--instance_data_dataset` requires setting `--image_key` and `--prompt_key`")

    return args

def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )
    # latent_image_ids = latent_image_ids.unsqueeze(0).repeat(batch_size, 1, 1)

    return latent_image_ids.to(device=device, dtype=dtype)

def main(args):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # if args.pretrained_model_architecture == "Meissonic":
    #     from src.pipeline import Pipeline
    # else:
    #     raise ValueError(f"Unknown model architecture: {args.pretrained_model_architecture}")


    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_main_process:
        accelerator.init_trackers("meissonic", config=vars(copy.deepcopy(args)))

    if args.seed is not None:
        set_seed(args.seed)

    if args.text_encoder_architecture == "open_clip":
        if args.resume_from_checkpoint:
            text_encoder = CLIPTextModelWithProjection.from_pretrained( 
                args.resume_from_checkpoint, subfolder="text_encoder", variant=args.variant
            )
            tokenizer = CLIPTokenizer.from_pretrained(
                args.resume_from_checkpoint, subfolder="tokenizer", variant=args.variant
            )
        else:
            text_encoder = CLIPTextModelWithProjection.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="text_encoder", variant=args.variant
            )
            tokenizer = CLIPTokenizer.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="tokenizer", variant=args.variant
            )

    # elif args.text_encoder_architecture == "CLIP_T5_base":
    #     text_encoder_clip = CLIPTextModelWithProjection.from_pretrained(
    #         args.pretrained_model_name_or_path, subfolder="text_encoder", variant=args.variant
    #     )
    #     tokenizer_clip = CLIPTokenizer.from_pretrained(
    #         args.pretrained_model_name_or_path, subfolder="tokenizer", variant=args.variant
    #     )
    #     from transformers import T5Tokenizer, T5ForConditionalGeneration
    #     text_encoder_t5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base",torch_dtype=torch.float16)
    #     tokenizer_t5 = T5Tokenizer.from_pretrained("google/flan-t5-base",torch_dtype=torch.float16)
    #     text_encoder = [text_encoder_clip,text_encoder_t5]
    #     tokenizer = [tokenizer_clip,tokenizer_t5]
    # elif args.text_encoder_architecture == "flan-t5-base":
    #     from transformers import T5Tokenizer, T5ForConditionalGeneration
    #     text_encoder = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base",torch_dtype=torch.float16)
    #     tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base",torch_dtype=torch.float16)
    # elif args.text_encoder_architecture == "gemini-2b":
    #     raise NotImplementedError("Gemini-2b is not yet supported")
    # elif args.text_encoder_architecture == "Qwen2-0.5B":
    #     raise NotImplementedError("Qwen2-0.5B is not yet supported")
    else:
        raise ValueError(f"Unknown text encoder architecture: {args.text_encoder_architecture}")
    
    vq_model = VQModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vqvae", revision=args.revision, variant=args.variant
    )

    if args.train_text_encoder:
        if args.text_encoder_use_lora:
            lora_config = LoraConfig(
                r=args.text_encoder_lora_r,
                lora_alpha=args.text_encoder_lora_alpha,
                target_modules=args.text_encoder_lora_target_modules,
            )
            text_encoder.add_adapter(lora_config)
        if args.text_encoder_architecture == "CLIP_T5_base": # Not support yet. Only support open_clip
            text_encoder[0].train()
            text_encoder[0].requires_grad_(True)
            text_encoder[1].train()
            text_encoder[1].requires_grad_(True)
        else:
            text_encoder.train()
            text_encoder.requires_grad_(True)
    else:
        if args.text_encoder_architecture == "CLIP_T5_base":  # Not support yet. Only support open_clip
            text_encoder[0].eval()
            text_encoder[0].requires_grad_(False)
            text_encoder[1].eval()
            text_encoder[1].requires_grad_(False)
        else:
            text_encoder.eval()
            text_encoder.requires_grad_(False)

    vq_model.requires_grad_(False)

    if args.pretrained_model_architecture == "Meissonic":
        if args.training_from_scratch:
            model = Transformer2DModel( 
                patch_size = 1,
                in_channels = 64,
                num_layers = 14, 
                num_single_layers = 28,
                attention_head_dim = 128, 
                num_attention_heads = 8, 
                joint_attention_dim = 1024,
                pooled_projection_dim = 1024,
                guidance_embeds = False,
                axes_dims_rope = (16, 56, 56),
                downsample= True,
                upsample= True,
            )

            # model_tmp = Transformer2DModel.from_pretrained("LAST_STAGE_CKPT_PATH", low_cpu_mem_usage=False, device_map=None)
            # model.load_state_dict(model_tmp.state_dict(), strict=False)
            # del model_tmp
        else:
            model = Transformer2DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer", low_cpu_mem_usage=False, device_map=None)
    else:
        raise ValueError(f"Unknown model architecture: {args.pretrained_model_architecture}")

    model = torch.compile(model)

    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
        )
        model.add_adapter(lora_config)

    model.train()

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
        if args.train_text_encoder:
            if args.text_encoder_architecture == "CLIP_T5_base": # Not support yet. Only support open_clip
                text_encoder[0].gradient_checkpointing_enable()
                text_encoder[1].gradient_checkpointing_enable()
            else:
                text_encoder.gradient_checkpointing_enable()

    if args.use_ema: # Not verify the robostness of this part
        ema = EMAModel(
            model.parameters(),
            decay=args.ema_decay,
            update_after_step=args.ema_update_after_step,
            model_cls= Transformer2DModel, 
            model_config=model.config,
        )

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            text_encoder_lora_layers_to_save = None

            for model_ in models:
                if isinstance(model_, type(accelerator.unwrap_model(model))):
                    if args.use_lora:
                        transformer_lora_layers_to_save = get_peft_model_state_dict(model_)
                    else:
                        model_.save_pretrained(os.path.join(output_dir, "transformer"))
                elif isinstance(model_, type(accelerator.unwrap_model(text_encoder))):
                    if args.text_encoder_use_lora:
                        text_encoder_lora_layers_to_save = get_peft_model_state_dict(model_)
                    else:
                        model_.save_pretrained(os.path.join(output_dir, "text_encoder"))
                else:
                    raise ValueError(f"unexpected save model: {model_.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            if transformer_lora_layers_to_save is not None or text_encoder_lora_layers_to_save is not None:
                LoraLoaderMixin.save_lora_weights(
                    output_dir,
                    unet_lora_layers=transformer_lora_layers_to_save,
                    text_encoder_lora_layers=text_encoder_lora_layers_to_save,
                )

            if args.use_ema:
                ema.save_pretrained(os.path.join(output_dir, "ema_model"))

    def load_model_hook(models, input_dir):
        transformer = None
        text_encoder_ = None

        # this part is added for keep consistency when add model.compile() in the model
        def adap_compile(ori_dict):#add '_orig_mod.' to each key
            new_dict = {}
            for k,v in ori_dict.items():
                new_dict['_orig_mod.'+k] = v
            return new_dict
            
        while len(models) > 0:
            model_ = models.pop()

            if isinstance(model_, type(accelerator.unwrap_model(model))):
                if args.use_lora:
                    transformer = model_
                else:
                    if args.pretrained_model_architecture == "Meissonic":
                        load_model = Transformer2DModel.from_pretrained(os.path.join(input_dir, "transformer"), low_cpu_mem_usage=False, device_map=None)
                    else:
                        raise ValueError(f"Unknown model architecture: {args.pretrained_model_architecture}")
                    model_.load_state_dict(adap_compile(load_model.state_dict()))
                    del load_model
            elif isinstance(model_, type(accelerator.unwrap_model(text_encoder))):
                if args.text_encoder_use_lora:
                    text_encoder_ = model_
                else:
                    try:
                        load_model = CLIPTextModelWithProjection.from_pretrained(os.path.join(input_dir, "text_encoder"))
                        model_.load_state_dict(load_model.state_dict())
                        # print('finished loading text encoder!')
                    except:
                        print('Not found text-encoder model in current folder. So we download one text encoder from Internet.')
                        load_model = CLIPTextModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
                        model_.load_state_dict(load_model.state_dict())
                    del load_model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        if transformer is not None or text_encoder_ is not None:
            lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)
            LoraLoaderMixin.load_lora_into_text_encoder(
                lora_state_dict, network_alphas=network_alphas, text_encoder=text_encoder_
            )
            LoraLoaderMixin.load_lora_into_transformer(
                lora_state_dict, network_alphas=network_alphas, transformer=transformer
            )

        if args.use_ema:
            load_from = EMAModel.from_pretrained(os.path.join(input_dir, "ema_model"), model_cls=Transformer2DModel)
            ema.load_state_dict(adap_compile(load_from.state_dict()))
            del load_from

    accelerator.register_load_state_pre_hook(load_model_hook)
    accelerator.register_save_state_pre_hook(save_model_hook)

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.adam_weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    if args.train_text_encoder: 
        if args.text_encoder_architecture == "CLIP_T5_base": # Not support yet. Only support open_clip
            optimizer_grouped_parameters.append(
                {"params": text_encoder[0].parameters(), "weight_decay": args.adam_weight_decay}
            )
            optimizer_grouped_parameters.append(
                {"params": text_encoder[1].parameters(), "weight_decay": args.adam_weight_decay}
            )
        else:
            optimizer_grouped_parameters.append(
                {"params": text_encoder.parameters(), "weight_decay": args.adam_weight_decay}
            )

    optimizer = optimizer_cls(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    logger.info("Creating dataloaders and lr_scheduler")

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps


    if args.instance_dataset == "MSCOCO600K":
        dataset = MSCOCO600KDataset(
            data_root=args.instance_data_dir,
            tokenizer=tokenizer,
            size=args.resolution,
            read_code = False,
            text_encoder_architecture=args.text_encoder_architecture
        )
    elif args.instance_dataset == "PickaPicV2":
        dataset = PickaPicV2Dataset(
            data_root=args.instance_data_dir,
            tokenizer=tokenizer,
            size=args.resolution,
            text_encoder_architecture=args.text_encoder_architecture
        )
    elif args.instance_dataset == "Meissonic":
        dataset = MyParquetDataset(
            parquet_dir=args.instance_data_dir,
            tokenizer=tokenizer,
            size=args.resolution,
            text_encoder_architecture=args.text_encoder_architecture
        )
    elif args.instance_dataset == "DATA_TYPE":
        raise NotImplementedError("DATA_TYPE is not yet supported")
        # Some instructions
        # Origanize your text-image pairs in the following way:
        # when apply __getitem__ method, return a dictionary with keys 'image', 'micro_conds' and 'prompt_input_ids'
        # For more details to follow, please refer to the implementation of MyParquetDataset class
    else:
        assert False

    train_dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=default_collate,
        pin_memory=True,
    )
    train_dataloader.num_batches = len(train_dataloader)

    lr_scheduler = diffusers.optimization.get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
    )

    logger.info("Preparing model, optimizer and dataloaders")

    if args.train_text_encoder:
        if args.text_encoder_architecture == "CLIP_T5_base": # Not support yet. Only support open_clip
            model, optimizer, lr_scheduler, train_dataloader, text_encoder[0], text_encoder[1] = accelerator.prepare(
                model, optimizer, lr_scheduler, train_dataloader, text_encoder[0], text_encoder[1]
            )
        else:
            model, optimizer, lr_scheduler, train_dataloader, text_encoder = accelerator.prepare(
            model, optimizer, lr_scheduler, train_dataloader, text_encoder
        )
    else:
        model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
            model, optimizer, lr_scheduler, train_dataloader
        )

    train_dataloader.num_batches = len(train_dataloader)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if not args.train_text_encoder:
        if args.text_encoder_architecture == "CLIP_T5_base": # Not support yet. Only support open_clip
            text_encoder[0].to(device=accelerator.device, dtype=weight_dtype)
            text_encoder[1].to(device=accelerator.device, dtype=weight_dtype)
        else:
            text_encoder.to(device=accelerator.device, dtype=weight_dtype)

    vq_model.to(device=accelerator.device)

    if args.use_ema:
        ema.to(accelerator.device)

    with nullcontext() if args.train_text_encoder else torch.no_grad():
        if args.text_encoder_architecture == "CLIP_T5_base": # Not support yet. Only support open_clip
            _input_ids_tmp_ = tokenize_prompt(tokenizer, "", args.text_encoder_architecture)
            _input_ids_tmp_[0] = _input_ids_tmp_[0].to(accelerator.device, non_blocking=True)
            _input_ids_tmp_[1] = _input_ids_tmp_[1].to(accelerator.device, non_blocking=True)
            empty_embeds, empty_clip_embeds = encode_prompt(
                text_encoder, _input_ids_tmp_, args.text_encoder_architecture
            )
        else:
            empty_embeds, empty_clip_embeds = encode_prompt(
                text_encoder, tokenize_prompt(tokenizer, "", args.text_encoder_architecture).to(accelerator.device, non_blocking=True), args.text_encoder_architecture
            )

        # There is a single image, we can just pre-encode the single prompt
        if args.instance_data_image is not None:
            prompt = os.path.splitext(os.path.basename(args.instance_data_image))[0]
            if args.text_encoder_architecture == "CLIP_T5_base": # Not support yet. Only support open_clip
                _input_ids_tmp_ = tokenize_prompt(tokenizer, prompt, args.text_encoder_architecture)
                _input_ids_tmp_[0] = _input_ids_tmp_[0].to(accelerator.device, non_blocking=True)
                _input_ids_tmp_[1] = _input_ids_tmp_[1].to(accelerator.device, non_blocking=True)
                empty_embeds, empty_clip_embeds = encode_prompt(
                    text_encoder, _input_ids_tmp_, args.text_encoder_architecture
                )
            else:
                encoder_hidden_states, cond_embeds = encode_prompt(
                    text_encoder, tokenize_prompt(tokenizer, prompt, args.text_encoder_architecture).to(accelerator.device, non_blocking=True), args.text_encoder_architecture
                )
            encoder_hidden_states = encoder_hidden_states.repeat(args.train_batch_size, 1, 1)
            cond_embeds = cond_embeds.repeat(args.train_batch_size, 1)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(train_dataloader.num_batches / args.gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs.
    # Note: We are not doing epoch based training here, but just using this for book keeping and being able to
    # reuse the same training loop with other datasets/loaders.
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {args.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = { args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint:
        if resume_from_checkpoint == "latest":
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            if len(dirs) > 0:
                resume_from_checkpoint = os.path.join(args.output_dir, dirs[-1])
            else:
                resume_from_checkpoint = None

        if resume_from_checkpoint is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
        else:
            accelerator.print(f"Resuming from checkpoint {resume_from_checkpoint}")

    if resume_from_checkpoint is None:
        global_step = 0
        first_epoch = 0
    else:
        accelerator.load_state(resume_from_checkpoint)
        global_step = int(os.path.basename(resume_from_checkpoint).split("-")[1])
        first_epoch = global_step // num_update_steps_per_epoch

    # This is to solve the inconsistent tensor device issue
    if args.use_ema:
        ema.shadow_params = [p.to(accelerator.device) for p in ema.shadow_params]

    # As stated above, we are not doing epoch based training here, but just using this for book keeping and being able to
    # reuse the same training loop with other datasets/loaders.
    for epoch in range(first_epoch, num_train_epochs):
        for batch in train_dataloader:
            torch.cuda.empty_cache()
            with torch.no_grad():
                micro_conds = batch["micro_conds"].to(accelerator.device, non_blocking=True)
                pixel_values = batch["image"].to(accelerator.device, non_blocking=True)

                batch_size = pixel_values.shape[0]

                split_batch_size = args.split_vae_encode if args.split_vae_encode is not None else batch_size
                num_splits = math.ceil(batch_size / split_batch_size)
                image_tokens = []
                for i in range(num_splits):
                    start_idx = i * split_batch_size
                    end_idx = min((i + 1) * split_batch_size, batch_size)
                    bs = pixel_values.shape[0]
                    image_tokens.append(
                        vq_model.quantize(vq_model.encode(pixel_values[start_idx:end_idx]).latents)[2][2].reshape(
                            split_batch_size, -1
                        )
                    )
                image_tokens = torch.cat(image_tokens, dim=0)

                batch_size, seq_len = image_tokens.shape

                timesteps = torch.rand(batch_size, device=image_tokens.device)
                mask_prob = torch.cos(timesteps * math.pi * 0.5)
                mask_prob = mask_prob.clip(args.min_masking_rate)

                num_token_masked = (seq_len * mask_prob).round().clamp(min=1)
                batch_randperm = torch.rand(batch_size, seq_len, device=image_tokens.device).argsort(dim=-1)
                mask = batch_randperm < num_token_masked.unsqueeze(-1)

                mask_id = accelerator.unwrap_model(model).config.vocab_size - 1
                input_ids = torch.where(mask, mask_id, image_tokens)
                labels = torch.where(mask, image_tokens, -100)

                if "prompt_input_ids" in batch:
                    with nullcontext() if args.train_text_encoder else torch.no_grad():
                        if args.text_encoder_architecture == "CLIP_T5_base": # Not support yet. Only support open_clip
                            batch["prompt_input_ids"][0] = batch["prompt_input_ids"][0].to(accelerator.device, non_blocking=True)
                            batch["prompt_input_ids"][1] = batch["prompt_input_ids"][1].to(accelerator.device, non_blocking=True)
                            encoder_hidden_states, cond_embeds = encode_prompt(
                                text_encoder, batch["prompt_input_ids"], args.text_encoder_architecture
                            )
                        else:
                            encoder_hidden_states, cond_embeds = encode_prompt(
                                text_encoder, batch["prompt_input_ids"].to(accelerator.device, non_blocking=True), args.text_encoder_architecture
                            )

                if args.cond_dropout_prob > 0.0:
                    assert encoder_hidden_states is not None

                    batch_size = encoder_hidden_states.shape[0]

                    mask = (
                        torch.zeros((batch_size, 1, 1), device=encoder_hidden_states.device).float().uniform_(0, 1)
                        < args.cond_dropout_prob
                    )

                    empty_embeds_ = empty_embeds.expand(batch_size, -1, -1)
                    encoder_hidden_states = torch.where(
                        (encoder_hidden_states * mask).bool(), encoder_hidden_states, empty_embeds_
                    )

                    empty_clip_embeds_ = empty_clip_embeds.expand(batch_size, -1)
                    cond_embeds = torch.where((cond_embeds * mask.squeeze(-1)).bool(), cond_embeds, empty_clip_embeds_)

                bs = input_ids.shape[0]
                vae_scale_factor = 2 ** (len(vq_model.config.block_out_channels) - 1)
                resolution = args.resolution // vae_scale_factor
                input_ids = input_ids.reshape(bs, resolution, resolution)

            if "prompt_input_ids" in batch:
                with nullcontext() if args.train_text_encoder else torch.no_grad():
                    if args.text_encoder_architecture == "CLIP_T5_base": # Not support yet. Only support open_clip
                        batch["prompt_input_ids"][0] = batch["prompt_input_ids"][0].to(accelerator.device, non_blocking=True)
                        batch["prompt_input_ids"][1] = batch["prompt_input_ids"][1].to(accelerator.device, non_blocking=True)
                        encoder_hidden_states, cond_embeds = encode_prompt(
                            text_encoder, batch["prompt_input_ids"],args.text_encoder_architecture
                        )
                    else:
                        encoder_hidden_states, cond_embeds = encode_prompt(
                            text_encoder, batch["prompt_input_ids"].to(accelerator.device, non_blocking=True),args.text_encoder_architecture
                        )

            # Train Step
            with accelerator.accumulate(model):
                codebook_size = accelerator.unwrap_model(model).config.codebook_size

                if args.pretrained_model_architecture == 'Meissonic':
                   
                    if args.resolution == 1024: # only stage 3 and stage 4 do not apply 2*
                        img_ids = _prepare_latent_image_ids(input_ids.shape[0], input_ids.shape[-2],input_ids.shape[-1],input_ids.device,input_ids.dtype)
                    else:
                        img_ids = _prepare_latent_image_ids(input_ids.shape[0],2*input_ids.shape[-2],2*input_ids.shape[-1],input_ids.device,input_ids.dtype)

                    txt_ids = torch.zeros(encoder_hidden_states.shape[1],3).to(device = input_ids.device, dtype = input_ids.dtype)
                   
                    logits = (
                        model(
                            hidden_states=input_ids, # should be (batch size, channel, height, width)
                            encoder_hidden_states=encoder_hidden_states, # should be (batch size, sequence_len, embed_dims)
                            micro_conds=micro_conds, # 
                            pooled_projections=cond_embeds, # should be (batch_size, projection_dim)
                            img_ids = img_ids,
                            txt_ids = txt_ids,
                            # timestep = timesteps * 20,
                            timestep = mask_prob * 1000,
                            # guidance = 9,
                        )
                        .reshape(bs, codebook_size, -1)
                        .permute(0, 2, 1)
                        .reshape(-1, codebook_size)
                    )
                else:
                    raise ValueError(f"Unknown model architecture: {args.pretrained_model_architecture}")

                loss = F.cross_entropy(
                    logits,
                    labels.view(-1),
                    ignore_index=-100,
                    reduction="mean",
                )

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                avg_masking_rate = accelerator.gather(mask_prob.repeat(args.train_batch_size)).mean()

                accelerator.backward(loss)

                if args.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()

                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema.step(model.parameters())

                if (global_step + 1) % args.logging_steps == 0:
                    logs = {
                        "step_loss": avg_loss.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "avg_masking_rate": avg_masking_rate.item(),
                    }
                    accelerator.log(logs, step=global_step + 1)

                    logger.info(
                        f"Step: {global_step + 1} "
                        f"Loss: {avg_loss.item():0.4f} "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.6f}"
                    )

                if (global_step + 1) % args.checkpointing_steps == 0:
                    save_checkpoint(args, accelerator, global_step + 1, logger)

                if (global_step + 1) % args.validation_steps == 0 and accelerator.is_main_process:
                    if args.use_ema:
                        ema.store(model.parameters())
                        ema.copy_to(model.parameters())

                    with torch.no_grad():
                        logger.info("Generating images...")

                        model.eval()

                        if args.train_text_encoder:
                            text_encoder.eval()

                        scheduler = Scheduler.from_pretrained(
                            args.pretrained_model_name_or_path,
                            subfolder="scheduler",
                            revision=args.revision,
                            variant=args.variant,
                            )
                        if args.text_encoder_architecture == "CLIP" or args.text_encoder_architecture == "open_clip":
                            pipe = Pipeline(
                                transformer=accelerator.unwrap_model(model),
                                tokenizer=tokenizer,
                                text_encoder=text_encoder,
                                vqvae=vq_model,
                                scheduler=scheduler,
                            )
                        else:
                            pipe = Pipeline(
                                transformer=accelerator.unwrap_model(model),
                                tokenizer=tokenizer[0],
                                text_encoder=text_encoder[0],
                                vqvae=vq_model,
                                scheduler=scheduler,
                                text_encoder_t5=text_encoder[1],
                                tokenizer_t5=tokenizer[1]
                            )

                      
                            


                        pil_images = pipe(prompt=args.validation_prompts,height=args.resolution,width=args.resolution,guidance_scale=9,num_inference_steps=64).images
                        wandb_images = [
                            wandb.Image(image, caption=args.validation_prompts[i])
                            for i, image in enumerate(pil_images)
                        ]

                        wandb.log({"generated_images": wandb_images}, step=global_step + 1)

                        result=[]
                        for img in pil_images:
                            if not isinstance(img, torch.Tensor):
                                img = transforms.ToTensor()(img)
                            result.append(img.unsqueeze(0))
                        result = torch.cat(result,dim=0)
                        result = make_grid(result, nrow=3)
                        save_image(result,os.path.join(args.output_dir,str(global_step)+'_text2image_1024_CFG-9.png'))

                        
                        # pil_images = pipe(prompt=args.validation_prompts,height=args.resolution,width=args.resolution,guidance_scale=9).images
                        # result=[]
                        # for img in pil_images:
                        #     if not isinstance(img, torch.Tensor):
                        #         img = transforms.ToTensor()(img)
                        #     result.append(img.unsqueeze(0))
                        # result = torch.cat(result,dim=0)
                        # result = make_grid(result, nrow=3)
                        # save_image(result,os.path.join(args.output_dir,str(global_step)+'_text2image_1024_CFG-9.png'))



                        model.train()

                        if args.train_text_encoder:
                            if args.text_encoder_architecture == "CLIP_T5_base": # Not support yet. Only support open_clip
                                text_encoder[0].train()
                                text_encoder[1].trian()
                            else:
                                text_encoder.train()

                    if args.use_ema:
                        ema.restore(model.parameters())

                global_step += 1

            # Stop training if max steps is reached
            if global_step >= args.max_train_steps:
                break
        # End for

    accelerator.wait_for_everyone()

    # Evaluate and save checkpoint at the end of training
    save_checkpoint(args, accelerator, global_step, logger)

    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        if args.use_ema:
            ema.copy_to(model.parameters())
        model.save_pretrained(args.output_dir)

    accelerator.end_training()





if __name__ == "__main__":
    main(parse_args())


