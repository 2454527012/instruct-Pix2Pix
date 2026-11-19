#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Script to fine-tune Stable Diffusion for InstructPix2Pix."""

import argparse
import logging
import math
import os
import shutil
from contextlib import nullcontext
from pathlib import Path

import accelerate
import datasets
import numpy as np
import PIL
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset, Dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    StableDiffusionInstructPix2PixPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.constants import DIFFUSERS_REQUEST_TIMEOUT
# DIFFUSERS_REQUEST_TIMEOUT = 60
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from new_conv_in import ContentAwarePositionGatedConvIn
from process_image import fill_box_with_mean_color
if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.37.0.dev0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "fusing/instructpix2pix-1000-samples": ("input_image", "edit_prompt", "edited_image"),
}
WANDB_TABLE_COL_NAMES = ["original_image", "edited_image", "edit_prompt"]


def log_validation(
    pipeline,
    args,
    accelerator,
    generator,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    original_image = download_image(args.val_image_url)
    edited_images = []
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    with autocast_ctx:
        for _ in range(args.num_validation_images):
            edited_images.append(
                pipeline(
                    args.validation_prompt,
                    image=original_image,
                    num_inference_steps=20,
                    image_guidance_scale=1.5,
                    guidance_scale=7,
                    generator=generator,
                ).images[0]
            )

    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            wandb_table = wandb.Table(columns=WANDB_TABLE_COL_NAMES)
            for edited_image in edited_images:
                wandb_table.add_data(wandb.Image(original_image), wandb.Image(edited_image), args.validation_prompt)
            tracker.log({"validation": wandb_table})


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for InstructPix2Pix.")
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
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )

    parser.add_argument(
        "--edited_image_column",
        type=str,
        default="edited_image",
        help="The column of the dataset containing the edited image.",
    )
    parser.add_argument(
        "--edit_prompt_column",
        type=str,
        default="edit_prompt",
        help="The column of the dataset containing the edit instruction.",
    )
    parser.add_argument(
        "--box_column",
        type=str,
        default="box_column",
        help="The column of the dataset containing the bounding box coordinates.",
    )
    parser.add_argument(
        "--mask_column",
        type=str,
        default="mask_column",
        help="The column of the dataset containing the mask image.",
    )
    parser.add_argument(
        "--val_image_url",
        type=str,
        default=None,
        help="URL to the original image that you would like to edit (used during inference for debugging purposes).",
    )
    parser.add_argument(
        "--validation_prompt", type=str, default=None, help="A prompt that is sampled during training for inference."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="instruct-pix2pix-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
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
        "--conditioning_dropout_prob",
        type=float,
        default=None,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://huggingface.co/papers/2211.09800.",
    )
    parser.add_argument(
        "--online_thumbnail_num_inference_steps",
        type=int,
        default=10,
        help=(
            "Number of no-gradient DDIM steps used to generate the current "
            "model's 512x512 thumbnail prediction for crop conditioning."
        ),
    )
    parser.add_argument(
        "--online_thumbnail_guidance_scale",
        type=float,
        default=3.0,
        help="Text guidance scale for online thumbnail prediction.",
    )
    parser.add_argument(
        "--online_thumbnail_image_guidance_scale",
        type=float,
        default=2.0,
        help="Image guidance scale for online thumbnail prediction.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
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
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
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
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
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
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    if args.online_thumbnail_num_inference_steps <= 0:
        raise ValueError("online_thumbnail_num_inference_steps must be positive.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def convert_to_np(image):
    arr = np.array(image)

    if arr.ndim == 2:
        arr = arr[None, ...]
    else:
        arr = arr.transpose(2, 0, 1)

    return arr


def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True, timeout=DIFFUSERS_REQUEST_TIMEOUT).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def main():
    args = parse_args()
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `hf auth login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    training_generator_seed = (
        args.seed
        if args.seed is not None
        else int(torch.initial_seed() % (2**63 - 1))
    )
    generator = torch.Generator(device=accelerator.device).manual_seed(
        training_generator_seed
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    # Keep online thumbnail sampling state completely separate from the
    # forward-noising scheduler used by the training objective.
    online_thumbnail_scheduler = DDIMScheduler.from_config(
        noise_scheduler.config
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    )

    # InstructPix2Pix uses an additional image for conditioning. To accommodate that,
    # it uses 8 channels (instead of 4) in the first (conv) layer of the UNet. This UNet is
    # then fine-tuned on the custom InstructPix2Pix dataset. This modified UNet is initialized
    # from the pre-trained checkpoints. For the extra channels added to the first layer, they are
    # initialized to zero.
    logger.info("Initializing the InstructPix2Pix UNet from the pretrained UNet.")
    # in_channels = 8
    # in_channels = 14
    # out_channels = unet.conv_in.out_channels
    # unet.register_to_config(in_channels=in_channels)



    # with torch.no_grad():
    #     new_conv_in = nn.Conv2d(
    #         in_channels, out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
    #     )
    #     new_conv_in.weight.zero_()
    #     new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
    #     unet.conv_in = new_conv_in

    ContentAwarePositionGatedConvIn.apply_to_unet(unet)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.eval()
    text_encoder.eval()

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Custom save/load hooks are required because the custom conv_in module is
    # serialized under checkpoint-*/unet rather than Accelerate's default
    # model.safetensors location.
    if version.parse(accelerate.__version__) < version.parse("0.16.0"):
        raise RuntimeError("Resumable training with the custom UNet requires accelerate >= 0.16.0.")

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for model in models:
                unwrap_model(model).save_pretrained(os.path.join(output_dir, "unet"))

                # Prevent duplicate default serialization; load_model_hook
                # restores this exact checkpoint/unet artifact explicitly.
                if weights:
                    weights.pop()

    def load_model_hook(models, input_dir):
        if args.use_ema:
            load_model = EMAModel.from_pretrained(
                os.path.join(input_dir, "unet_ema"),
                UNet2DConditionModel,
            )
            ema_unet.load_state_dict(load_model.state_dict())
            ema_unet.to(accelerator.device)
            del load_model

        unet_dir = os.path.join(input_dir, "unet")
        safetensor_path = os.path.join(unet_dir, "diffusion_pytorch_model.safetensors")
        bin_path = os.path.join(unet_dir, "diffusion_pytorch_model.bin")

        if os.path.isfile(safetensor_path):
            from safetensors.torch import load_file

            state_dict = load_file(safetensor_path, device="cpu")
        elif os.path.isfile(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(
                "Cannot resume custom UNet: expected either "
                f"'{safetensor_path}' or '{bin_path}'."
            )

        while models:
            model = unwrap_model(models.pop())

            # ContentAwarePositionGatedConvIn directly receives four-channel
            # noisy latents. Keep the serialized UNet config aligned with the
            # unchanged custom architecture before loading its exact weights.
            model.register_to_config(in_channels=4)
            model.load_state_dict(state_dict, strict=True)

        logger.info(f"Restored custom UNet weights from {unet_dir}")

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
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

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        # dataset = load_dataset(
        #     "imagefolder",
        #     data_files=data_files,
        #     cache_dir=args.cache_dir,
        # )
        dataset = load_dataset(
            "json",
            data_files=args.train_data_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/main/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if args.edit_prompt_column is None:
        edit_prompt_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        edit_prompt_column = args.edit_prompt_column
        if edit_prompt_column not in column_names:
            raise ValueError(
                f"--edit_prompt_column' value '{args.edit_prompt_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.edited_image_column is None:
        edited_image_column = dataset_columns[2] if dataset_columns is not None else column_names[2]
    else:
        edited_image_column = args.edited_image_column
        if edited_image_column not in column_names:
            raise ValueError(
                f"--edited_image_column' value '{args.edited_image_column}' needs to be one of: {', '.join(column_names)}"
            )

    if args.box_column is None:
        box_column = dataset_columns[2] if dataset_columns is not None else column_names[2]
    else:
        box_column = args.box_column
        if box_column not in column_names:
            raise ValueError(
                f"--box_column' value '{args.box_column}' needs to be one of: {', '.join(column_names)}"
            )
            
    if args.mask_column is None:
        mask_column = dataset_columns[3] if dataset_columns is not None else column_names[3]
    else:
        mask_column = args.mask_column
        if mask_column not in column_names:
            raise ValueError(
                f"--mask_column' value '{args.mask_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(captions):
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    # train_transforms = transforms.Compose(
    #     [
    #         transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
    #         transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
    #     ]
    # )

    # def preprocess_images(examples):
    #     original_images = np.concatenate(
    #         [convert_to_np(image, args.resolution) for image in examples[original_image_column]]
    #     )
    #     edited_images = np.concatenate(
    #         [convert_to_np(image, args.resolution) for image in examples[edited_image_column]]
    #     )
    #     # We need to ensure that the original and the edited images undergo the same
    #     # augmentation transforms.
    #     images = np.stack([original_images, edited_images])
    #     images = torch.tensor(images)
    #     images = 2 * (images / 255) - 1
    #     return train_transforms(images)

    from torchvision.transforms import RandomCrop, functional as Ft
    import xml.etree.ElementTree as ET
    import numpy as np
    import random
    from PIL import Image, ImageChops, ImageDraw, ImageFilter

    crop_pixel_size = 512
    crop_latent_size = crop_pixel_size // 8
    num_crops_per_image = 8

    try:
        image_resize_mode = Image.Resampling.BICUBIC
        mask_resize_mode = Image.Resampling.NEAREST
        box_resize_mode = Image.Resampling.BOX
        affine_transform_mode = Image.Transform.AFFINE
    except AttributeError:
        image_resize_mode = Image.BICUBIC
        mask_resize_mode = Image.NEAREST
        box_resize_mode = Image.BOX
        affine_transform_mode = Image.AFFINE

    def get_letterbox_geometry(image_size, output_size=crop_pixel_size):
        """Return one shared aspect-ratio-preserving resize/padding geometry."""
        image_w, image_h = image_size
        if image_w <= 0 or image_h <= 0:
            raise ValueError(f"Invalid image size for letterboxing: {image_size}")

        scale = min(output_size / image_w, output_size / image_h)
        resized_w = max(1, min(output_size, round(image_w * scale)))
        resized_h = max(1, min(output_size, round(image_h * scale)))
        left = (output_size - resized_w) // 2
        top = (output_size - resized_h) // 2
        return resized_w, resized_h, left, top

    def resize_and_letterbox(image, geometry, resample):
        """Resize an image in pixel space and center it on a black 512x512 canvas."""
        resized_w, resized_h, left, top = geometry
        resized = image.resize((resized_w, resized_h), resample=resample)
        canvas = Image.new(image.mode, (crop_pixel_size, crop_pixel_size), color=0)
        canvas.paste(resized, (left, top))
        return canvas

    def remove_letterbox(image, geometry):
        """Remove the artificial black letterbox border."""
        resized_w, resized_h, left, top = geometry
        return image.crop((left, top, left + resized_w, top + resized_h))

    def simulate_thumbnail_prediction_errors(thumbnail):
        """Approximate common 512px global-restoration errors without inference."""
        thumbnail = thumbnail.convert("RGB")
        width, height = thumbnail.size

        # Most generated thumbnails are only mildly wrong. A small fraction
        # receives medium corruption so the crop branch also learns to repair
        # the less successful global predictions seen at inference time.
        severity_draw = random.random()
        if severity_draw < 0.15:
            severity = random.uniform(0.20, 0.45)
        elif severity_draw < 0.85:
            severity = random.uniform(0.45, 0.90)
        else:
            severity = random.uniform(0.90, 1.25)

        # Apply one small, spatially uniform RGB cast to the whole thumbnail
        # prediction. The final native-resolution composite confines this to
        # the repair region and makes rectangular tone steps similar to real
        # failed global predictions.
        thumbnail_array = np.asarray(
            thumbnail,
            dtype=np.float32,
        )
        channel_gains = np.array(
            [
                random.uniform(
                    1.0 - 0.025 * severity,
                    1.0 + 0.025 * severity,
                )
                for _ in range(3)
            ],
            dtype=np.float32,
        ).reshape(1, 1, 3)
        channel_biases = np.array(
            [
                random.uniform(-4.0 * severity, 4.0 * severity)
                for _ in range(3)
            ],
            dtype=np.float32,
        ).reshape(1, 1, 3)
        thumbnail_array = np.clip(
            thumbnail_array * channel_gains + channel_biases,
            0,
            255,
        ).astype(np.uint8)
        thumbnail = Image.fromarray(thumbnail_array, mode="RGB")

        # Real global predictions often contain broad, non-uniform luminance
        # drift or soft dark/light areas instead of a single constant RGB
        # offset. Create a tiny random field and bicubic-upsample it so only
        # very low spatial frequencies are changed.
        if random.random() < 0.85:
            grid_w = random.randint(3, 5)
            grid_h = random.randint(3, 5)
            luminance = np.random.uniform(
                -1.0,
                1.0,
                size=(grid_h, grid_w, 1),
            ).astype(np.float32)
            chroma = np.random.uniform(
                -1.0,
                1.0,
                size=(grid_h, grid_w, 3),
            ).astype(np.float32)
            field_delta = (
                luminance * (8.0 * severity)
                + chroma * (2.5 * severity)
            )
            field_array = np.clip(
                128.0 + field_delta,
                0,
                255,
            ).astype(np.uint8)
            color_field = Image.fromarray(
                field_array,
                mode="RGB",
            ).resize(
                (width, height),
                resample=image_resize_mode,
            )
            thumbnail = ImageChops.add(
                thumbnail,
                color_field,
                scale=1.0,
                offset=-128,
            )

        # Small translation/scale/rotation errors reproduce displaced cracks,
        # lane markings and road texture at the generated-region boundary.
        if random.random() < 0.70:
            effective_scale = max(width, height) / crop_pixel_size
            translate_x = random.uniform(-4.0, 4.0) * severity * effective_scale
            translate_y = random.uniform(-4.0, 4.0) * severity * effective_scale
            affine_scale = random.uniform(
                1.0 - 0.015 * severity,
                1.0 + 0.015 * severity,
            )
            angle = math.radians(
                random.uniform(-0.40, 0.40) * severity
            )
            cos_angle = math.cos(angle) / affine_scale
            sin_angle = math.sin(angle) / affine_scale
            center_x = (width - 1) * 0.5
            center_y = (height - 1) * 0.5
            affine_data = (
                cos_angle,
                sin_angle,
                center_x
                - cos_angle * center_x
                - sin_angle * center_y
                - translate_x,
                -sin_angle,
                cos_angle,
                center_y
                + sin_angle * center_x
                - cos_angle * center_y
                - translate_y,
            )
            fill_color = thumbnail.resize(
                (1, 1),
                resample=box_resize_mode,
            ).getpixel((0, 0))
            thumbnail = thumbnail.transform(
                (width, height),
                affine_transform_mode,
                affine_data,
                resample=image_resize_mode,
                fillcolor=fill_color,
            )

        # Suppress high-frequency texture by an extra down/up pass. This is
        # independent of the original native->512 resize and approximates a
        # global model that reconstructs layout but misses pavement detail.
        if random.random() < 0.85:
            retain_ratio = max(
                0.55,
                1.0 - random.uniform(0.10, 0.35) * severity,
            )
            reduced_size = (
                max(8, round(width * retain_ratio)),
                max(8, round(height * retain_ratio)),
            )
            thumbnail = thumbnail.resize(
                reduced_size,
                resample=image_resize_mode,
            ).resize(
                (width, height),
                resample=image_resize_mode,
            )

        blur_radius = random.uniform(0.35, 1.50) * severity
        thumbnail = thumbnail.filter(
            ImageFilter.GaussianBlur(radius=blur_radius)
        )

        # Add occasional soft stains or bands. These emulate the diffuse dark
        # blobs and low-frequency striping visible in unsuccessful thumbnail
        # restorations, while avoiding hard synthetic shapes.
        if random.random() < 0.25 + 0.30 * severity:
            artifact_mask = Image.new("L", (width, height), color=0)
            draw = ImageDraw.Draw(artifact_mask)
            artifact_count = 1 if random.random() < 0.8 else 2
            for _ in range(artifact_count):
                shape_type = random.random()
                if shape_type < 0.55:
                    artifact_w = random.uniform(0.10, 0.32) * width
                    artifact_h = random.uniform(0.08, 0.25) * height
                elif shape_type < 0.78:
                    artifact_w = random.uniform(0.45, 0.90) * width
                    artifact_h = random.uniform(0.04, 0.12) * height
                else:
                    artifact_w = random.uniform(0.04, 0.12) * width
                    artifact_h = random.uniform(0.40, 0.85) * height

                left = random.uniform(-0.05, 0.95) * width
                top = random.uniform(-0.05, 0.95) * height
                right = left + artifact_w
                bottom = top + artifact_h
                artifact_alpha = round(
                    random.uniform(0.08, 0.20)
                    * min(severity, 1.0)
                    * 255
                )
                if shape_type < 0.55:
                    draw.ellipse(
                        (left, top, right, bottom),
                        fill=artifact_alpha,
                    )
                else:
                    draw.rectangle(
                        (left, top, right, bottom),
                        fill=artifact_alpha,
                    )

            artifact_mask = artifact_mask.filter(
                ImageFilter.GaussianBlur(
                    radius=random.uniform(5.0, 18.0)
                )
            )
            softened = thumbnail.filter(
                ImageFilter.GaussianBlur(
                    radius=random.uniform(1.5, 4.0) * severity
                )
            )
            thumbnail = Image.composite(
                softened,
                thumbnail,
                artifact_mask,
            )

            mean_color = np.array(
                thumbnail.resize(
                    (1, 1),
                    resample=box_resize_mode,
                ).getpixel((0, 0)),
                dtype=np.float32,
            )
            signed_shift = random.choice((-1.0, 1.0))
            common_shift = signed_shift * random.uniform(15.0, 38.0)
            color_shift = np.random.uniform(-5.0, 5.0, size=3)
            artifact_color = tuple(
                np.clip(
                    mean_color + common_shift + color_shift,
                    0,
                    255,
                ).astype(np.uint8).tolist()
            )
            artifact_layer = Image.new(
                "RGB",
                (width, height),
                artifact_color,
            )
            thumbnail = Image.composite(
                artifact_layer,
                thumbnail,
                artifact_mask,
            )

        return thumbnail, severity

    def add_subtle_boundary_artifacts(
        thumbnail_prior,
        inpaint_mask,
        severity,
    ):
        """Add a faint native-resolution seam on some repair boundaries."""
        if random.random() > 0.35:
            return thumbnail_prior

        native_scale = max(thumbnail_prior.size) / crop_pixel_size

        # Add a subtle repair-side seam only. inpaint_mask uses white=1 for
        # the repair area, therefore mask - eroded(mask) is an inner contour
        # ring. The faint dark/red tint prevents the crop model from relying
        # on a perfectly clean thumbnail-to-known-pixel boundary.
        edge_width = min(
            6,
            max(
                1,
                round(
                    random.uniform(0.35, 0.90)
                    * native_scale
                    * severity
                ),
            ),
        )
        repair_bbox = inpaint_mask.getbbox()
        if repair_bbox is not None:
            repair_width = repair_bbox[2] - repair_bbox[0]
            repair_height = repair_bbox[3] - repair_bbox[1]
            # Preserve a central coarse silhouette even for small defects.
            edge_width = min(
                edge_width,
                max(1, min(repair_width, repair_height) // 4),
            )
        eroded_mask = inpaint_mask.filter(
            ImageFilter.MinFilter(size=2 * edge_width + 1)
        )
        inner_edge_mask = ImageChops.subtract(inpaint_mask, eroded_mask)
        inner_edge_mask = inner_edge_mask.filter(
            ImageFilter.GaussianBlur(
                radius=max(0.5, 0.20 * native_scale)
            )
        )

        mean_color = np.array(
            thumbnail_prior.resize(
                (1, 1),
                resample=box_resize_mode,
            ).getpixel((0, 0)),
            dtype=np.float32,
        )
        edge_type = random.random()
        if edge_type < 0.75:
            neutral_shift = random.choice((-1.0, 1.0)) * random.uniform(15, 30)
            edge_color = tuple(
                np.clip(
                    mean_color + neutral_shift,
                    0,
                    255,
                ).astype(np.uint8).tolist()
            )
        elif edge_type < 0.93:
            edge_color = tuple(
                np.clip(
                    mean_color - random.uniform(35, 75),
                    0,
                    255,
                ).astype(np.uint8).tolist()
            )
        else:
            edge_color = tuple(
                np.clip(
                    mean_color
                    + np.array(
                        [
                            random.uniform(25, 45),
                            random.uniform(-20, -8),
                            random.uniform(-20, -8),
                        ],
                        dtype=np.float32,
                    ),
                    0,
                    255,
                ).astype(np.uint8).tolist()
            )
        corrupted_edge = Image.new(
            "RGB",
            thumbnail_prior.size,
            edge_color,
        )
        subtle_edge = Image.blend(
            thumbnail_prior,
            corrupted_edge,
            alpha=random.uniform(0.02, 0.07) * min(severity, 1.0),
        )
        return Image.composite(
            subtle_edge,
            thumbnail_prior,
            inner_edge_mask,
        )

    def make_letterbox_valid_mask(geometry):
        """Mark real image pixels as 1 and artificial black padding as 0."""
        resized_w, resized_h, left, top = geometry
        valid_mask = Image.new("L", (crop_pixel_size, crop_pixel_size), color=0)
        valid_mask.paste(255, (left, top, left + resized_w, top + resized_h))
        return valid_mask

    def parse_boxes_from_xml(xml_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            boxes = []
            for obj in root.findall("object"):
                bndbox = obj.find("bndbox")
                if bndbox is None:
                    continue

                xmin = int(float(bndbox.find("xmin").text))
                ymin = int(float(bndbox.find("ymin").text))
                xmax = int(float(bndbox.find("xmax").text))
                ymax = int(float(bndbox.find("ymax").text))

                boxes.append((xmin, ymin, xmax, ymax))

            return boxes

        except Exception as e:
            print(f"Warning: failed to parse {xml_path}: {e}")
            return []

    def choose_relative_center_from_xml(
        xml_path,
        image_w,
        image_h,
        box_near_prob=0.8,
        max_retry=50,
    ):
        """
        返回相对中心 (cx_rel, cy_rel)，范围严格在 [0, 1]

        80% box附近:
            0~48 px      : 40%
            48~100 px    : 30%
            100~200 px   : 30%

        20% 全图随机
        """

        boxes = parse_boxes_from_xml(xml_path)

        def sample_radius():
            r = random.random()
            if r < 0.4:
                return random.uniform(0, 48)
            elif r < 0.7:
                return random.uniform(48, 100)
            else:
                return random.uniform(100, 200)

        if len(boxes) > 0 and random.random() < box_near_prob:
            xmin, ymin, xmax, ymax = random.choice(boxes)

            box_cx = (xmin + xmax) * 0.5
            box_cy = (ymin + ymax) * 0.5

            cx, cy = box_cx, box_cy

            for _ in range(max_retry):
                radius = sample_radius()
                theta = random.uniform(0, 2 * np.pi)

                dx = radius * np.cos(theta)
                dy = radius * np.sin(theta)

                cand_x = box_cx + dx
                cand_y = box_cy + dy

                if 0 <= cand_x < image_w and 0 <= cand_y < image_h:
                    cx, cy = cand_x, cand_y
                    break
            else:
                # 极端情况兜底，保证不越界
                cx = np.clip(cx, 0, image_w - 1)
                cy = np.clip(cy, 0, image_h - 1)

        else:
            cx = random.uniform(0, image_w - 1)
            cy = random.uniform(0, image_h - 1)

        # 最后一层保险
        cx = float(np.clip(cx, 0, image_w - 1))
        cy = float(np.clip(cy, 0, image_h - 1))

        cx_rel = cx / max(image_w - 1, 1)
        cy_rel = cy / max(image_h - 1, 1)

        return np.array([cx_rel, cy_rel], dtype=np.float32)   

    def choose_multiple_relative_centers_from_xml(
        xml_path,
        image_w,
        image_h,
        num_centers=8,
        box_near_prob=0.8,
    ):
        """
        调用原来的采样函数多次。

        返回：
            [num_centers, 2]
        """
        centers = []

        for _ in range(num_centers):
            center = choose_relative_center_from_xml(
                xml_path=xml_path,
                image_w=image_w,
                image_h=image_h,
                box_near_prob=box_near_prob,
            )
            centers.append(center)

        return np.stack(centers, axis=0).astype(np.float32)
        

    # 辅助函数：解析 XML，返回所有 box 的列表，每个 box 为 (xmin, ymin, xmax, ymax)
    def preprocess_images(examples):
        edits = examples[edited_image_column]
        box_urls = examples[box_column]
        masks = examples[mask_column]

        processed_originals = []
        processed_masks = []
        processed_edits = []
        processed_inpaint_masks = []
        processed_centers = []
        processed_global_originals = []
        processed_global_edits = []
        processed_global_masks = []
        processed_global_inpaint_masks = []
        processed_global_valid_masks = []

        for edit_path, box_path, mask_path in zip(edits, box_urls, masks):
            large_region_size = random.randint(768, 1280)

            orig, inpaint_mask = fill_box_with_mean_color(
                box_path,
                edit_path,
                large_region_prob=0.5,
                large_region_size=large_region_size,
            )

            orig = orig.convert("RGB")
            edit = Image.open(edit_path).convert("RGB")
            mask = Image.open(mask_path)

            if not isinstance(inpaint_mask, Image.Image):
                inpaint_mask = Image.fromarray(inpaint_mask)

            assert orig.size == edit.size
            assert orig.size == mask.size
            assert orig.size == inpaint_mask.size

            orig_w, orig_h = orig.size
            new_w = (orig_w // 8) * 8
            new_h = (orig_h // 8) * 8

            orig = orig.crop((0, 0, new_w, new_h))
            edit = edit.crop((0, 0, new_w, new_h))
            mask = mask.crop((0, 0, new_w, new_h))
            inpaint_mask = inpaint_mask.crop((0, 0, new_w, new_h))

            # One of the eight expanded samples is trained as a whole-image
            # restoration task. Resize in pixel space before VAE encoding so
            # this branch represents the complete image, not a resized latent.
            letterbox_geometry = get_letterbox_geometry(orig.size)
            global_orig = resize_and_letterbox(orig, letterbox_geometry, image_resize_mode)
            global_edit = resize_and_letterbox(edit, letterbox_geometry, image_resize_mode)
            global_mask = resize_and_letterbox(mask, letterbox_geometry, mask_resize_mode)
            global_inpaint_mask = resize_and_letterbox(inpaint_mask, letterbox_geometry, mask_resize_mode)
            global_valid_mask = make_letterbox_valid_mask(letterbox_geometry)

            centers = choose_multiple_relative_centers_from_xml(
                xml_path=box_path,
                image_w=new_w,
                image_h=new_h,
                num_centers=num_crops_per_image,
                box_near_prob=0.6,
            )

            orig_np = convert_to_np(orig)
            edit_np = convert_to_np(edit)

            mask_np = convert_to_np(mask)
            mask_np = mask_np / 255.0

            inpaint_mask_np = convert_to_np(inpaint_mask)
            inpaint_mask_np = inpaint_mask_np / 255.0

            # Keep the existing mask loading behavior unchanged, but fail with
            # a useful message if the dataset violates the one-channel
            # contract required by [mask, mask, inpaint_mask] packing.
            if mask_np.shape[0] != 1:
                raise ValueError(
                    "Dataset mask must be single-channel before condition "
                    f"packing, but '{mask_path}' has {mask_np.shape[0]} "
                    "channels."
                )
            if inpaint_mask_np.shape[0] != 1:
                raise ValueError(
                    "Inpaint mask must be single-channel before condition "
                    f"packing, but the mask derived from '{box_path}' has "
                    f"{inpaint_mask_np.shape[0]} channels."
                )

            global_orig_np = convert_to_np(global_orig)
            global_edit_np = convert_to_np(global_edit)
            global_mask_np = convert_to_np(global_mask) / 255.0
            global_inpaint_mask_np = convert_to_np(global_inpaint_mask) / 255.0
            global_valid_mask_np = convert_to_np(global_valid_mask) / 255.0

            processed_originals.append(orig_np)
            processed_masks.append(mask_np)
            processed_edits.append(edit_np)
            processed_inpaint_masks.append(inpaint_mask_np)
            processed_centers.append(centers)
            processed_global_originals.append(global_orig_np)
            processed_global_edits.append(global_edit_np)
            processed_global_masks.append(global_mask_np)
            processed_global_inpaint_masks.append(global_inpaint_mask_np)
            processed_global_valid_masks.append(global_valid_mask_np)

        original_images = np.stack(processed_originals)
        mask_images = np.stack(processed_masks)
        edited_images = np.stack(processed_edits)
        inpaint_masks = np.stack(processed_inpaint_masks)
        centers = np.stack(processed_centers)
        global_original_images = np.stack(processed_global_originals)
        global_edited_images = np.stack(processed_global_edits)
        global_mask_images = np.stack(processed_global_masks)
        global_inpaint_masks = np.stack(processed_global_inpaint_masks)
        global_valid_masks = np.stack(processed_global_valid_masks)

        images = np.stack([
            original_images,
            edited_images,
        ])

        images = torch.from_numpy(images)
        images = 2 * (images / 255.0) - 1

        masks = torch.from_numpy(mask_images)
        inpaint_masks = torch.from_numpy(inpaint_masks)
        centers = torch.from_numpy(centers)
        global_images = torch.from_numpy(
            np.stack([global_original_images, global_edited_images])
        )
        global_images = 2 * (global_images / 255.0) - 1
        global_masks = torch.from_numpy(global_mask_images)
        global_inpaint_masks = torch.from_numpy(global_inpaint_masks)
        global_valid_masks = torch.from_numpy(global_valid_masks)

        return (
            images,
            masks,
            inpaint_masks,
            centers,
            global_images,
            global_masks,
            global_inpaint_masks,
            global_valid_masks,
        )
    
    def preprocess_train(examples):
        # Preprocess images.
        (
            preprocessed_images,
            mask_images,
            inpaint_masks,
            centers,
            global_images,
            global_masks,
            global_inpaint_masks,
            global_valid_masks,
        ) = preprocess_images(examples)

        # preprocessed_images contains the native-resolution restoration
        # input and target. The letterboxed whole-image tensors are returned
        # separately for the one global sample in each group of eight.
        original_images, edited_images = preprocessed_images

        # original_images = original_images.reshape(
        #     -1, 3, args.resolution, args.resolution
        # )
        # mask_images = mask_images.reshape(
        #     -1, 3, args.resolution, args.resolution
        # )
        # edited_images = edited_images.reshape(
        #     -1, 3, args.resolution, args.resolution
        # )

        # Collate the preprocessed images into the `examples`.
        examples["original_pixel_values"] = original_images
        examples["mask_pixel_values"] = mask_images
        examples["edited_pixel_values"] = edited_images
        examples["inpaint_masks"] = inpaint_masks
        examples["centers"] = centers
        examples["global_original_pixel_values"] = global_images[0]
        examples["global_edited_pixel_values"] = global_images[1]
        examples["global_mask_pixel_values"] = global_masks
        examples["global_inpaint_masks"] = global_inpaint_masks
        examples["global_valid_masks"] = global_valid_masks
        # Preprocess the captions.
        captions = list(examples[edit_prompt_column])
        examples["input_ids"] = tokenize_captions(captions)

        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        original_pixel_values = torch.stack(
            [example["original_pixel_values"] for example in examples]
        )
        original_pixel_values = original_pixel_values.to(
            memory_format=torch.contiguous_format
        ).float()

        mask_pixel_values = torch.stack(
            [example["mask_pixel_values"] for example in examples]
        )
        mask_pixel_values = mask_pixel_values.to(
            memory_format=torch.contiguous_format
        ).float()

        edited_pixel_values = torch.stack(
            [example["edited_pixel_values"] for example in examples]
        )
        edited_pixel_values = edited_pixel_values.to(
            memory_format=torch.contiguous_format
        ).float()

        inpaint_masks = torch.stack(
            [example["inpaint_masks"] for example in examples]
        )
        inpaint_masks = inpaint_masks.to(
            memory_format=torch.contiguous_format
        ).float()
        input_ids = torch.stack(
            [example["input_ids"] for example in examples]
        )
        
        centers = torch.stack(
            [example["centers"] for example in examples]
        ).float()

        global_original_pixel_values = torch.stack(
            [example["global_original_pixel_values"] for example in examples]
        ).to(memory_format=torch.contiguous_format).float()
        global_edited_pixel_values = torch.stack(
            [example["global_edited_pixel_values"] for example in examples]
        ).to(memory_format=torch.contiguous_format).float()
        global_mask_pixel_values = torch.stack(
            [example["global_mask_pixel_values"] for example in examples]
        ).to(memory_format=torch.contiguous_format).float()
        global_inpaint_masks = torch.stack(
            [example["global_inpaint_masks"] for example in examples]
        ).to(memory_format=torch.contiguous_format).float()
        global_valid_masks = torch.stack(
            [example["global_valid_masks"] for example in examples]
        ).to(memory_format=torch.contiguous_format).float()

        return {
            "original_pixel_values": original_pixel_values,
            "mask_pixel_values": mask_pixel_values,
            "edited_pixel_values": edited_pixel_values,
            "inpaint_masks": inpaint_masks,
            "input_ids": input_ids,
            "centers": centers,
            "global_original_pixel_values": global_original_pixel_values,
            "global_edited_pixel_values": global_edited_pixel_values,
            "global_mask_pixel_values": global_mask_pixel_values,
            "global_inpaint_masks": global_inpaint_masks,
            "global_valid_masks": global_valid_masks,
        }

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    unwrapped_unet = unwrap_model(unet)
    
    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("instruct-pix2pix", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(
        "  Online thumbnail sampling = "
        f"DDIM/{args.online_thumbnail_num_inference_steps} steps, "
        f"text guidance {args.online_thumbnail_guidance_scale}, "
        f"image guidance {args.online_thumbnail_image_guidance_scale}"
    )
    if args.conditioning_dropout_prob is None:
        logger.warning(
            "Online thumbnail prediction uses three-way classifier-free "
            "guidance, but conditioning_dropout_prob is disabled. Existing "
            "checkpoints should have been trained with condition dropout for "
            "the image-only and unconditional branches to be reliable."
        )
    global_step = 0
    online_thumbnail_seed_base = (
        args.seed
        if args.seed is not None
        else int(torch.initial_seed() % (2**31))
    )
    first_epoch = 0
    # Keep a valid dataloader offset when no checkpoint exists (including
    # --resume_from_checkpoint=latest on a fresh output directory).
    resume_step = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            requested_path = args.resume_from_checkpoint
            candidate_paths = [requested_path]
            if not os.path.isabs(requested_path):
                candidate_paths.append(os.path.join(args.output_dir, requested_path))

            checkpoint_path = next(
                (candidate for candidate in candidate_paths if os.path.isdir(candidate)),
                None,
            )
            path = (
                os.path.basename(os.path.normpath(checkpoint_path))
                if checkpoint_path is not None
                else None
            )
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
            checkpoint_path = os.path.join(args.output_dir, path) if path is not None else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(checkpoint_path)
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    # progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar = tqdm(
        total=args.max_train_steps,
        initial=global_step,
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    progress_bar.set_description("Steps")
        
    def crop_latents_by_centers(
        edited_latents_full,
        original_latents_full,
        centers,
        crop_latent_size=64,
    ):
        """
        根据输入 centers 从 edited / original latent 中裁同一个位置。

        输入:
            edited_latents_full:   B, 4, H, W
            original_latents_full: B, 4, H, W
            centers:              B, 2，相对中心坐标，范围 [0, 1]
                                centers[:, 0] = cx_rel
                                centers[:, 1] = cy_rel

        输出:
            edited_crop:   B, 4, crop_latent_size, crop_latent_size
            original_crop: B, 4, crop_latent_size, crop_latent_size
            crop_centers:  B, 2，实际裁剪窗口中心的相对坐标
        """

        bsz, _, h_lat, w_lat = edited_latents_full.shape

        if h_lat < crop_latent_size or w_lat < crop_latent_size:
            raise ValueError(
                f"latent size too small: got H={h_lat}, W={w_lat}, "
                f"but crop_latent_size={crop_latent_size}"
            )

        if centers.shape[0] != bsz or centers.shape[1] != 2:
            raise ValueError(
                f"centers shape should be [B, 2], got {centers.shape}"
            )

        centers = centers.to(
            device=edited_latents_full.device,
            dtype=edited_latents_full.dtype,
        )

        edited_crops = []
        original_crops = []
        actual_centers = []

        half = crop_latent_size / 2.0

        for i in range(bsz):
            cx_rel = float(centers[i, 0].clamp(0, 1).item())
            cy_rel = float(centers[i, 1].clamp(0, 1).item())

            cx = cx_rel * w_lat
            cy = cy_rel * h_lat

            left = int(round(cx - half))
            top = int(round(cy - half))

            left = max(0, min(left, w_lat - crop_latent_size))
            top = max(0, min(top, h_lat - crop_latent_size))

            edited_crop = edited_latents_full[
                i:i + 1,
                :,
                top:top + crop_latent_size,
                left:left + crop_latent_size,
            ]

            original_crop = original_latents_full[
                i:i + 1,
                :,
                top:top + crop_latent_size,
                left:left + crop_latent_size,
            ]

            actual_cx = (left + crop_latent_size / 2.0) / w_lat
            actual_cy = (top + crop_latent_size / 2.0) / h_lat

            edited_crops.append(edited_crop)
            original_crops.append(original_crop)
            actual_centers.append([actual_cx, actual_cy])

        edited_crops = torch.cat(edited_crops, dim=0)
        original_crops = torch.cat(original_crops, dim=0)

        actual_centers = torch.tensor(
            actual_centers,
            device=edited_latents_full.device,
            dtype=edited_latents_full.dtype,
        )

        return edited_crops, original_crops, actual_centers

    def crop_mask_by_centers(
        mask_full,
        centers,
        crop_pixel_size=512,
    ):
        """
        mask_full: B, 1, H, W
        centers:   B, 2, 相对坐标 [0, 1], x/y
        return:    B, 1, 512, 512
        """

        bsz, _, h, w = mask_full.shape
        half = crop_pixel_size / 2.0
        crops = []
        centers = centers.to(device=mask_full.device, dtype=mask_full.dtype)
        for i in range(bsz):
            cx_rel = float(centers[i, 0].clamp(0, 1).item())
            cy_rel = float(centers[i, 1].clamp(0, 1).item())

            cx = cx_rel * w
            cy = cy_rel * h

            left = int(round(cx - half))
            top = int(round(cy - half))

            left = max(0, min(left, w - crop_pixel_size))
            top = max(0, min(top, h - crop_pixel_size))

            crop = mask_full[
                i:i + 1,
                :,
                top:top + crop_pixel_size,
                left:left + crop_pixel_size,
            ]

            crops.append(crop)

        return torch.cat(crops, dim=0)

    @torch.no_grad()
    def predict_online_thumbnail(
        global_image_latents,
        global_mask_latents,
        native_full_latents,
        prompt_embeds,
        null_prompt_embeds,
        global_image_pixels,
        global_inpaint_mask,
        global_valid_mask,
        thumbnail_generator,
    ):
        """Generate the current model's repaired 512px thumbnail.

        This is a real multi-step diffusion prediction made by the live UNet.
        It is intentionally detached from the training graph; only the later
        crop denoising pass contributes gradients.
        """

        batch_size = global_image_latents.shape[0]
        latent_height, latent_width = global_image_latents.shape[-2:]
        device = global_image_latents.device
        dtype = global_image_latents.dtype

        if (latent_height, latent_width) != (
            crop_latent_size,
            crop_latent_size,
        ):
            raise ValueError(
                "Online thumbnail image latents must be 64x64, got "
                f"{(latent_height, latent_width)}."
            )

        null_prompt_embeds = null_prompt_embeds.to(
            device=prompt_embeds.device,
            dtype=prompt_embeds.dtype,
        )
        if null_prompt_embeds.shape[0] == 1 and batch_size > 1:
            null_prompt_embeds = null_prompt_embeds.expand(
                batch_size,
                -1,
                -1,
            )
        if null_prompt_embeds.shape[0] != batch_size:
            raise ValueError(
                "Null prompt batch does not match online thumbnail batch: "
                f"{null_prompt_embeds.shape[0]} vs {batch_size}."
            )

        center_maps = torch.zeros(
            batch_size,
            2,
            latent_height,
            latent_width,
            device=device,
            dtype=dtype,
        )
        zero_image_latents = torch.zeros_like(global_image_latents)
        zero_mask_latents = torch.zeros_like(global_mask_latents)
        zero_full_latents = torch.zeros_like(native_full_latents)

        fixed_noise = torch.randn(
            global_image_latents.shape,
            generator=thumbnail_generator,
            device=device,
            dtype=dtype,
        )
        thumbnail_latents = (
            fixed_noise * online_thumbnail_scheduler.init_noise_sigma
        )
        repair_mask_latent = F.interpolate(
            global_inpaint_mask.float(),
            size=(latent_height, latent_width),
            mode="nearest",
        ).to(device=device, dtype=dtype)

        online_thumbnail_scheduler.set_timesteps(
            args.online_thumbnail_num_inference_steps,
            device=device,
        )
        inference_timesteps = online_thumbnail_scheduler.timesteps

        def run_unet_branch(
            latent_model_input,
            timestep,
            branch_prompt_embeds,
            branch_image_latents,
            branch_mask_latents,
            branch_full_latents,
        ):
            unwrapped_unet.conv_in.set_condition_context(
                original_crop_latents=branch_image_latents,
                mask_embeds=branch_mask_latents,
                original_latents_full=branch_full_latents,
                center_maps=center_maps,
            )
            try:
                with accelerator.autocast():
                    return unwrapped_unet(
                        latent_model_input,
                        timestep,
                        encoder_hidden_states=branch_prompt_embeds,
                        return_dict=False,
                    )[0]
            finally:
                unwrapped_unet.conv_in.clear_condition_context()

        was_training = unwrapped_unet.training
        unwrapped_unet.eval()
        try:
            for timestep_index, timestep in enumerate(inference_timesteps):
                latent_model_input = online_thumbnail_scheduler.scale_model_input(
                    thumbnail_latents,
                    timestep,
                )

                # Run the three InstructPix2Pix CFG branches sequentially.
                # This has the same semantics as a 3B concatenated batch but
                # substantially lowers peak activation memory during training.
                noise_pred_text = run_unet_branch(
                    latent_model_input,
                    timestep,
                    prompt_embeds,
                    global_image_latents,
                    global_mask_latents,
                    native_full_latents,
                )
                noise_pred_image = run_unet_branch(
                    latent_model_input,
                    timestep,
                    null_prompt_embeds,
                    global_image_latents,
                    global_mask_latents,
                    native_full_latents,
                )
                noise_pred_uncond = run_unet_branch(
                    latent_model_input,
                    timestep,
                    null_prompt_embeds,
                    zero_image_latents,
                    zero_mask_latents,
                    zero_full_latents,
                )

                noise_pred = (
                    noise_pred_uncond
                    + args.online_thumbnail_guidance_scale
                    * (noise_pred_text - noise_pred_image)
                    + args.online_thumbnail_image_guidance_scale
                    * (noise_pred_image - noise_pred_uncond)
                )

                thumbnail_latents = online_thumbnail_scheduler.step(
                    noise_pred,
                    timestep,
                    thumbnail_latents,
                    return_dict=False,
                )[0]

                # Preserve known pixels throughout sampling, matching the
                # first-stage inference pipeline. The same initial noise is
                # used to construct the known latent at every next timestep.
                if timestep_index < len(inference_timesteps) - 1:
                    next_timestep = inference_timesteps[timestep_index + 1]
                    if torch.is_tensor(next_timestep):
                        next_timestep = next_timestep.reshape(1).to(
                            device=device,
                            dtype=torch.long,
                        ).expand(batch_size)
                    else:
                        next_timestep = torch.full(
                            (batch_size,),
                            int(next_timestep),
                            device=device,
                            dtype=torch.long,
                        )
                    known_latents = online_thumbnail_scheduler.add_noise(
                        original_samples=global_image_latents,
                        noise=fixed_noise,
                        timesteps=next_timestep,
                    )
                else:
                    known_latents = global_image_latents

                thumbnail_latents = (
                    repair_mask_latent * thumbnail_latents
                    + (1.0 - repair_mask_latent) * known_latents
                )
        finally:
            unwrapped_unet.conv_in.clear_condition_context()
            unwrapped_unet.train(was_training)

        with accelerator.autocast():
            thumbnail_pixels = vae.decode(
                thumbnail_latents / vae.config.scaling_factor,
                return_dict=False,
            )[0]

        thumbnail_pixels = thumbnail_pixels.clamp(-1.0, 1.0)
        global_inpaint_mask = global_inpaint_mask.to(
            device=device,
            dtype=thumbnail_pixels.dtype,
        )
        global_valid_mask = global_valid_mask.to(
            device=device,
            dtype=thumbnail_pixels.dtype,
        )
        global_image_pixels = global_image_pixels.to(
            device=device,
            dtype=thumbnail_pixels.dtype,
        )

        # Restore known pixels exactly and keep the artificial letterbox black
        # in normalized [-1, 1] pixel space.
        thumbnail_pixels = (
            global_inpaint_mask * thumbnail_pixels
            + (1.0 - global_inpaint_mask) * global_image_pixels
        )
        thumbnail_pixels = (
            global_valid_mask * thumbnail_pixels
            - (1.0 - global_valid_mask)
        )
        return thumbnail_pixels.detach()

    @torch.no_grad()
    def build_native_crop_condition(
        thumbnail_pixels,
        original_pixels,
        native_inpaint_mask,
    ):
        """Upsample the online thumbnail and composite it in pixel space."""

        native_height, native_width = original_pixels.shape[-2:]
        resized_width, resized_height, left, top = get_letterbox_geometry(
            (native_width, native_height)
        )
        thumbnail_content = thumbnail_pixels[
            :,
            :,
            top:top + resized_height,
            left:left + resized_width,
        ]
        upsampled_thumbnail = F.interpolate(
            thumbnail_content.float(),
            size=(native_height, native_width),
            mode="bicubic",
            align_corners=False,
        ).to(dtype=original_pixels.dtype)
        upsampled_thumbnail = upsampled_thumbnail.clamp(-1.0, 1.0)

        native_inpaint_mask = (
            native_inpaint_mask >= 0.5
        ).to(
            device=original_pixels.device,
            dtype=original_pixels.dtype,
        )
        crop_condition_pixels = (
            native_inpaint_mask * upsampled_thumbnail
            + (1.0 - native_inpaint_mask) * original_pixels
        )
        return crop_condition_pixels.detach()

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0

        if (
            args.resume_from_checkpoint
            and epoch == first_epoch
            and resume_step > 0
        ):
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader,
                resume_step,
            )
        else:
            active_dataloader = train_dataloader


        # for step, batch in enumerate(train_dataloader):
            # 恢复训练时跳过已经完成的 dataloader step
            # if (
            #     args.resume_from_checkpoint
            #     and epoch == first_epoch
            #     and step < resume_step
            # ):
            #     if step % args.gradient_accumulation_steps == 0:
            #         progress_bar.update(1)
            #     continue


        for step, batch in enumerate(
            active_dataloader,
            start=resume_step if epoch == first_epoch else 0,
        ):
            # =====================================================
            # 1. 完整图片移到 GPU
            # =====================================================

            mask_full = batch[
                "mask_pixel_values"
            ].to(
                device=accelerator.device,
                dtype=weight_dtype,
            )

            inpaint_masks_full = batch[
                "inpaint_masks"
            ].to(
                device=accelerator.device,
                dtype=weight_dtype,
            )

            global_inpaint_masks = batch[
                "global_inpaint_masks"
            ].to(
                device=accelerator.device,
                dtype=weight_dtype,
            )

            global_valid_masks = batch[
                "global_valid_masks"
            ].to(
                device=accelerator.device,
                dtype=weight_dtype,
            )

            input_ids = batch["input_ids"].to(
                accelerator.device
            )

            # [B, 4, 2]
            all_centers = batch["centers"].to(
                device=accelerator.device,
                dtype=weight_dtype,
            )

            full_bsz = mask_full.shape[0]

            if all_centers.shape != (
                full_bsz,
                num_crops_per_image,
                2,
            ):
                raise ValueError(
                    f"centers shape should be "
                    f"[B, {num_crops_per_image}, 2], "
                    f"got {all_centers.shape}"
                )

            # =====================================================
            # 2. 完整图片 VAE 编码一次
            #
            # edited 和 original 合并后，只调用一次 vae.encode。
            # =====================================================
            with torch.no_grad():
                with accelerator.autocast():
                    edited_pixel_values = batch[
                        "edited_pixel_values"
                    ].to(
                        device=accelerator.device,
                        dtype=weight_dtype,
                    )

                    latents_full = (
                        vae.encode(edited_pixel_values)
                        .latent_dist.sample()
                        * vae.config.scaling_factor
                    )

                    del edited_pixel_values

                    original_pixel_values = batch[
                        "original_pixel_values"
                    ].to(
                        device=accelerator.device,
                        dtype=weight_dtype,
                    )

                    original_latents_full = (
                        vae.encode(original_pixel_values)
                        .latent_dist.mode()
                        * vae.config.scaling_factor
                    )

                    # The whole-image branch is resized and letterboxed at
                    # pixel level in preprocessing, then encoded by the VAE.
                    # original_latents_full remains the raw mean-filled global
                    # condition for the one whole-image sample in each group.
                    global_edited_pixel_values = batch[
                        "global_edited_pixel_values"
                    ].to(
                        device=accelerator.device,
                        dtype=weight_dtype,
                    )
                    global_target_latents = (
                        vae.encode(global_edited_pixel_values)
                        .latent_dist.sample()
                        * vae.config.scaling_factor
                    )
                    del global_edited_pixel_values

                    global_original_pixel_values = batch[
                        "global_original_pixel_values"
                    ].to(
                        device=accelerator.device,
                        dtype=weight_dtype,
                    )
                    global_original_image_embeds = (
                        vae.encode(global_original_pixel_values)
                        .latent_dist.mode()
                        * vae.config.scaling_factor
                    )

                    global_mask_pixel_values = batch[
                        "global_mask_pixel_values"
                    ].to(
                        device=accelerator.device,
                        dtype=weight_dtype,
                    )
                    # Keep the mask-condition network shape unchanged while
                    # exposing the actual repair region to the model.  The
                    # pseudo-RGB mask uses the fixed channel convention:
                    #   R = dataset mask, G = dataset mask,
                    #   B = inpaint mask (1 means repair this pixel).
                    global_mask_3ch = torch.cat(
                        [
                            global_mask_pixel_values,
                            global_mask_pixel_values,
                            global_inpaint_masks,
                        ],
                        dim=1,
                    )
                    global_mask_3ch = global_mask_3ch * 2.0 - 1.0
                    global_mask_embeds = (
                        vae.encode(global_mask_3ch)
                        .latent_dist.mode()
                        * vae.config.scaling_factor
                    )
                    del global_mask_pixel_values
                    del global_mask_3ch

                    encoder_hidden_states_full = text_encoder(
                        input_ids
                    )[0]

                    # Online three-way InstructPix2Pix CFG always needs the
                    # empty-text embedding, even when training condition
                    # dropout itself is disabled.
                    null_conditioning = text_encoder(
                        tokenize_captions([""]).to(
                            accelerator.device
                        )
                    )[0]

                    thumbnail_seed = (
                        online_thumbnail_seed_base
                        + accelerator.process_index * 1_000_003
                        + epoch * max(len(train_dataloader), 1)
                        + step
                    ) % (2**63 - 1)
                    thumbnail_generator = torch.Generator(
                        device=accelerator.device
                    ).manual_seed(thumbnail_seed)

                    # Stage 1: the current live model makes one real 512px
                    # prediction per source image. No clean target tensor is
                    # passed into this call.
                    online_thumbnail_pixels = predict_online_thumbnail(
                        global_image_latents=global_original_image_embeds,
                        global_mask_latents=global_mask_embeds,
                        native_full_latents=original_latents_full,
                        prompt_embeds=encoder_hidden_states_full,
                        null_prompt_embeds=null_conditioning,
                        global_image_pixels=global_original_pixel_values,
                        global_inpaint_mask=global_inpaint_masks,
                        global_valid_mask=global_valid_masks,
                        thumbnail_generator=thumbnail_generator,
                    )

                    # Stage 2 conditioning is constructed in pixel space:
                    # the model prediction fills only the repair region and
                    # all known pixels come from the mean-filled input.
                    crop_condition_pixel_values = build_native_crop_condition(
                        thumbnail_pixels=online_thumbnail_pixels,
                        original_pixels=original_pixel_values,
                        native_inpaint_mask=inpaint_masks_full,
                    )
                    crop_condition_latents_full = (
                        vae.encode(crop_condition_pixel_values)
                        .latent_dist.mode()
                        * vae.config.scaling_factor
                    )

                    if crop_condition_latents_full.shape != original_latents_full.shape:
                        raise ValueError(
                            "Online crop-condition latent shape does not match "
                            "the native filled-image latent: "
                            f"{crop_condition_latents_full.shape} vs "
                            f"{original_latents_full.shape}."
                        )

                    del thumbnail_generator
                    del online_thumbnail_pixels
                    del crop_condition_pixel_values
                    del original_pixel_values
                    del global_original_pixel_values

                    # latents_full = (
                    #     latents_full
                    #     * vae.config.scaling_factor
                    # )

                    # original_latents_full = (
                    #     original_latents_full
                    #     * vae.config.scaling_factor
                    # )

                    # encoder_hidden_states_full = (
                    #     text_encoder(input_ids)[0]
                    # )

                    # if (
                    #     args.conditioning_dropout_prob
                    #     is not None
                    # ):
                    #     null_conditioning = text_encoder(
                    #         tokenize_captions([""]).to(
                    #             accelerator.device
                    #         )
                    #     )[0]
                    # else:
                    #     null_conditioning = None

            latents_full = latents_full.detach()
            original_latents_full = (
                original_latents_full.detach()
            )
            crop_condition_latents_full = (
                crop_condition_latents_full.detach()
            )
            global_target_latents = global_target_latents.detach()
            global_original_image_embeds = global_original_image_embeds.detach()
            global_mask_embeds = global_mask_embeds.detach()
            encoder_hidden_states_full = (
                encoder_hidden_states_full.detach()
            )

            null_conditioning = null_conditioning.detach()

            # =====================================================
            # 3. 每张完整图片扩展成 8 个训练样本
            #
            # B=2 时：
            # [2, 8, 2] -> [16, 2]
            # =====================================================
            centers = all_centers.reshape(
                full_bsz * num_crops_per_image,
                2,
            )

            # 完整 latent 从 B 扩展到 B*8
            latents_full_repeated = (
                latents_full.repeat_interleave(
                    num_crops_per_image,
                    dim=0,
                )
            )

            original_latents_full_repeated = (
                crop_condition_latents_full.repeat_interleave(
                    num_crops_per_image,
                    dim=0,
                )
            )
            filled_original_latents_full_repeated = (
                original_latents_full.repeat_interleave(
                    num_crops_per_image,
                    dim=0,
                )
            )

            encoder_hidden_states = (
                encoder_hidden_states_full
                .repeat_interleave(
                    num_crops_per_image,
                    dim=0,
                )
            )

            mask_full_repeated = (
                mask_full.repeat_interleave(
                    num_crops_per_image,
                    dim=0,
                )
            )

            inpaint_masks_repeated = (
                inpaint_masks_full.repeat_interleave(
                    num_crops_per_image,
                    dim=0,
                )
            )

            # Exactly one of each image's eight expanded samples is a
            # whole-image restoration sample. The selected sample gets a
            # letterboxed 512x512 VAE input and position (0, 0).
            global_sample_indices = torch.randint(
                num_crops_per_image,
                (full_bsz,),
                device=accelerator.device,
            )
            is_global_sample = torch.zeros(
                full_bsz,
                num_crops_per_image,
                device=accelerator.device,
                dtype=torch.bool,
            )
            is_global_sample.scatter_(
                1,
                global_sample_indices.unsqueeze(1),
                True,
            )
            is_global_sample = is_global_sample.reshape(-1)
            is_global_sample_4d = is_global_sample.view(-1, 1, 1, 1)

            # =====================================================
            # 4. 一次训练 7 个 crop + 1 个全图样本
            # =====================================================
            with accelerator.accumulate(unet):
                (
                    latents,
                    original_image_embeds,
                    centers,
                ) = crop_latents_by_centers(
                    latents_full_repeated,
                    original_latents_full_repeated,
                    centers,
                    crop_latent_size=crop_latent_size,
                )

                # B=1 时，bsz=8；其中一个样本用全图 latent 替换。
                bsz = latents.shape[0]

                global_target_latents_repeated = global_target_latents.repeat_interleave(
                    num_crops_per_image,
                    dim=0,
                )
                global_original_image_embeds_repeated = (
                    global_original_image_embeds.repeat_interleave(
                        num_crops_per_image,
                        dim=0,
                    )
                )
                global_mask_embeds_repeated = global_mask_embeds.repeat_interleave(
                    num_crops_per_image,
                    dim=0,
                )

                latents = torch.where(
                    is_global_sample_4d,
                    global_target_latents_repeated,
                    latents,
                )
                original_image_embeds = torch.where(
                    is_global_sample_4d,
                    global_original_image_embeds_repeated,
                    original_image_embeds,
                )
                # Crop rows retain the VAE latent of the pixel-composited
                # two-stage condition.  The single whole-image row retains
                # the original raw filled full-image context, exactly as in
                # the existing global-repair branch.
                original_latents_full_repeated = torch.where(
                    is_global_sample_4d,
                    filled_original_latents_full_repeated,
                    original_latents_full_repeated,
                )
                centers = torch.where(
                    is_global_sample.view(-1, 1),
                    torch.zeros_like(centers),
                    centers,
                )

                inpaint_mask_crop = (
                    crop_mask_by_centers(
                        inpaint_masks_repeated,
                        centers,
                        crop_pixel_size=crop_pixel_size,
                    )
                )

                global_inpaint_masks_repeated = global_inpaint_masks.repeat_interleave(
                    num_crops_per_image,
                    dim=0,
                )
                global_valid_masks_repeated = global_valid_masks.repeat_interleave(
                    num_crops_per_image,
                    dim=0,
                )
                valid_loss_mask = torch.ones_like(inpaint_mask_crop)
                inpaint_mask_crop = torch.where(
                    is_global_sample_4d,
                    global_inpaint_masks_repeated,
                    inpaint_mask_crop,
                )
                valid_loss_mask = torch.where(
                    is_global_sample_4d,
                    global_valid_masks_repeated,
                    valid_loss_mask,
                )

                with torch.no_grad():
                    with accelerator.autocast():
                        mask_crop = crop_mask_by_centers(
                            mask_full_repeated,
                            centers,
                            crop_pixel_size=crop_pixel_size,
                        )
                        # Preserve the existing three-channel VAE input and
                        # four-channel mask latent, but use the blue channel
                        # to carry the repair-region mask.  This does not
                        # change the UNet or checkpoint parameter shapes.
                        mask_3ch = torch.cat(
                            [
                                mask_crop,
                                mask_crop,
                                inpaint_mask_crop,
                            ],
                            dim=1,
                        )
                        mask_3ch = mask_3ch * 2.0 - 1.0
                        # [B, 3, 512, 512] -> [B, 4, 64, 64]
                        mask_image_embeds = (
                            vae.encode(mask_3ch)
                            .latent_dist.mode()
                        )

                        mask_image_embeds = (
                            mask_image_embeds
                            * vae.config.scaling_factor
                        )
                        
                        del mask_3ch

                mask_image_embeds = torch.where(
                    is_global_sample_4d,
                    global_mask_embeds_repeated,
                    mask_image_embeds,
                )

                # 每个 crop 独立噪声
                noise = torch.randn_like(latents)

                timesteps = torch.randint(
                    0,
                    noise_scheduler.config
                    .num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()

                noisy_latents = (
                    noise_scheduler.add_noise(
                        latents,
                        noise,
                        timesteps,
                    )
                )

                center_maps = centers.to(
                    device=latents.device,
                    dtype=original_image_embeds.dtype,
                ).view(
                    bsz,
                    2,
                    1,
                    1,
                ).expand(
                    bsz,
                    2,
                    noisy_latents.shape[2],
                    noisy_latents.shape[3],
                )

                # =================================================
                # Conditioning dropout
                # =================================================
                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(
                        bsz,
                        device=latents.device,
                        generator=generator,
                    )

                    # -------------------------------------------------
                    # 1. 文本条件 dropout
                    # random_p < 2p 时，使用空文本条件
                    # -------------------------------------------------
                    prompt_mask = (
                        random_p
                        < 2 * args.conditioning_dropout_prob
                    ).view(
                        bsz,
                        1,
                        1,
                    )

                    encoder_hidden_states = torch.where(
                        prompt_mask,
                        null_conditioning.to(
                            device=encoder_hidden_states.device,
                            dtype=encoder_hidden_states.dtype,
                        ),
                        encoder_hidden_states,
                    )

                    # -------------------------------------------------
                    # 2. 图像条件 dropout
                    #
                    # 原来：
                    # condition_embeds = image_mask * condition_embeds
                    #
                    # 现在没有 condition_embeds，
                    # 因此对每一种图像条件分别乘同一个 image_mask。
                    # -------------------------------------------------
                    image_mask = 1 - (
                        (
                            random_p
                            >= args.conditioning_dropout_prob
                        ).to(original_image_embeds.dtype)
                        *
                        (
                            random_p
                            < 3 * args.conditioning_dropout_prob
                        ).to(original_image_embeds.dtype)
                    )

                    image_mask = image_mask.view(
                        bsz,
                        1,
                        1,
                        1,
                    )

                    original_image_embeds = (
                        original_image_embeds * image_mask
                    )

                    mask_image_embeds = (
                        mask_image_embeds
                        * image_mask.to(mask_image_embeds.dtype)
                    )

                    original_latents_full_repeated = (
                        original_latents_full_repeated
                        * image_mask.to(
                            original_latents_full_repeated.dtype
                        )
                    )

                    center_maps = (
                        center_maps
                        * image_mask.to(center_maps.dtype)
                    )


                if (
                    noise_scheduler.config.prediction_type
                    == "epsilon"
                ):
                    target = noise

                elif (
                    noise_scheduler.config.prediction_type
                    == "v_prediction"
                ):
                    target = (
                        noise_scheduler.get_velocity(
                            latents,
                            noise,
                            timesteps,
                        )
                    )

                else:
                    raise ValueError(
                        "Unknown prediction type "
                        f"{noise_scheduler.config.prediction_type}"
                    )

                with accelerator.autocast():
                    unwrapped_unet.conv_in.set_condition_context(
                        original_crop_latents=original_image_embeds,
                        mask_embeds=mask_image_embeds,
                        original_latents_full=original_latents_full_repeated,
                        center_maps=center_maps,
                    )
                try:
                    with accelerator.autocast():
                        model_pred = unet(
                            noisy_latents,
                            timesteps,
                            encoder_hidden_states,
                            return_dict=False,
                        )[0]

                finally:

                    unwrapped_unet.conv_in.clear_condition_context()

                # =================================================
                # mask weighted loss
                # =================================================
                inpaint_mask_64 = F.interpolate(
                    inpaint_mask_crop.float(),
                    size=model_pred.shape[-2:],
                    mode="nearest",
                )

                inpaint_mask_64 = (
                    inpaint_mask_64.expand_as(
                        model_pred
                    )
                )

                valid_mask_64 = F.interpolate(
                    valid_loss_mask.float(),
                    size=model_pred.shape[-2:],
                    mode="nearest",
                ).expand_as(model_pred)

                repair_weight_alpha = 8.0

                loss_weights = (
                    valid_mask_64.float()
                    * (
                        1.0
                        + repair_weight_alpha
                        * inpaint_mask_64.float()
                    )
                )

                loss_map = F.mse_loss(
                    model_pred.float(),
                    target.float(),
                    reduction="none",
                )

                weighted_loss = (
                    (loss_map * loss_weights)
                    .sum(dim=(1, 2, 3))
                    /
                    loss_weights
                    .sum(dim=(1, 2, 3))
                    .clamp_min(1e-8)
                )

                normal_loss = (
                    (loss_map * valid_mask_64)
                    .sum(dim=(1, 2, 3))
                    /
                    valid_mask_64
                    .sum(dim=(1, 2, 3))
                    .clamp_min(1e-8)
                )

                bsz = model_pred.shape[0]

                use_weight = torch.ones(
                    bsz,
                    device=model_pred.device,
                    dtype=torch.bool,
                )

                num_unweighted = max(
                    1,
                    bsz // 4,
                )

                unweighted_indices = torch.randperm(
                    bsz,
                    device=model_pred.device,
                )[:num_unweighted]

                use_weight[
                    unweighted_indices
                ] = False

                loss = torch.where(
                    use_weight,
                    weighted_loss,
                    normal_loss,
                ).mean()

                avg_loss = accelerator.gather(
                    loss.detach().reshape(1)
                ).mean()

                train_loss += (
                    avg_loss.item()
                    / args.gradient_accumulation_steps
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        unet.parameters(),
                        args.max_grad_norm,
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(
                    set_to_none=True
                )

            # =====================================================
            # 当前 8 个 crop 完成一次更新
            # =====================================================
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(
                        unet.parameters()
                    )

                progress_bar.update(1)
                global_step += 1

                accelerator.log(
                    {
                        "train_loss": train_loss,
                        "crop_batch_size": bsz,
                        "global_samples": int(is_global_sample.sum().item()),
                    },
                    step=global_step,
                )

                train_loss = 0.0

                if (
                    global_step
                    % args.checkpointing_steps
                    == 0
                ):
                    if accelerator.is_main_process:
                        if (
                            args.checkpoints_total_limit
                            is not None
                        ):
                            checkpoints = [
                                d
                                for d in os.listdir(
                                    args.output_dir
                                )
                                if d.startswith(
                                    "checkpoint"
                                )
                            ]

                            checkpoints = sorted(
                                checkpoints,
                                key=lambda x: int(
                                    x.split("-")[1]
                                ),
                            )

                            if (
                                len(checkpoints)
                                >= args.checkpoints_total_limit
                            ):
                                num_to_remove = (
                                    len(checkpoints)
                                    - args.checkpoints_total_limit
                                    + 1
                                )

                                removing_checkpoints = (
                                    checkpoints[
                                        :num_to_remove
                                    ]
                                )

                                for checkpoint_name in (
                                    removing_checkpoints
                                ):
                                    shutil.rmtree(
                                        os.path.join(
                                            args.output_dir,
                                            checkpoint_name,
                                        )
                                    )

                        save_path = os.path.join(
                            args.output_dir,
                            f"checkpoint-{global_step}",
                        )

                        accelerator.save_state(
                            save_path
                        )

                        logger.info(
                            f"Saved state to "
                            f"{save_path}"
                        )

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "crop_batch": bsz,
            }
            
            
            
            # =====================================================
            # 当前 step 的 GPU Tensor 已经用完，释放 Python 引用
            # =====================================================

            del latents_full
            del original_latents_full
            del crop_condition_latents_full
            del latents_full_repeated
            del original_latents_full_repeated
            del filled_original_latents_full_repeated
            del global_target_latents
            del global_original_image_embeds
            del global_mask_embeds
            del global_target_latents_repeated
            del global_original_image_embeds_repeated
            del global_mask_embeds_repeated

            del mask_full
            del mask_full_repeated
            del mask_crop
            del mask_image_embeds

            del inpaint_masks_full
            del inpaint_masks_repeated
            del inpaint_mask_crop
            del inpaint_mask_64
            del global_inpaint_masks
            del global_inpaint_masks_repeated
            del global_valid_masks
            del global_valid_masks_repeated
            del valid_loss_mask
            del valid_mask_64

            del encoder_hidden_states_full
            del encoder_hidden_states

            if null_conditioning is not None:
                del null_conditioning

            del centers
            del all_centers
            del center_maps
            del global_sample_indices
            del is_global_sample
            del is_global_sample_4d

            del latents
            del original_image_embeds
            del noise
            del noisy_latents
            del timesteps

            del target
            del model_pred

            del loss_weights
            del loss_map
            del loss
            

            progress_bar.set_postfix(
                **logs
            )

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

        # 后面继续放原来的 validation


    # for epoch in range(first_epoch, args.num_train_epochs):
    #     unet.train()
    #     train_loss = 0.0
    #     for step, batch in enumerate(train_dataloader):
    #         # Skip steps until we reach the resumed step
    #         if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
    #             if step % args.gradient_accumulation_steps == 0:
    #                 progress_bar.update(1)
    #             continue

    #         with accelerator.accumulate(unet):
    #             # We want to learn the denoising process w.r.t the edited images which
    #             # are conditioned on the original image (which was edited) and the edit instruction.
    #             # So, first, convert images to latent space.

    #             latents_full = vae.encode(batch["edited_pixel_values"].to(weight_dtype)).latent_dist.sample()
    #             latents_full = latents_full * vae.config.scaling_factor

    #             # Get the text embedding for conditioning.
    #             encoder_hidden_states = text_encoder(batch["input_ids"])[0]
    #             # Get the additional image embedding for conditioning.
    #             # Instead of getting a diagonal Gaussian here, we simply take the mode.
    #             original_latents_full = vae.encode(batch["original_pixel_values"].to(weight_dtype)).latent_dist.mode()
    #             original_latents_full = original_latents_full * vae.config.scaling_factor
    #             centers = batch["centers"].to(weight_dtype)
    #             latents, original_image_embeds, centers = crop_latents_by_centers(
    #                 latents_full,
    #                 original_latents_full,
    #                 centers,
    #                 crop_latent_size=64,
    #             )
    #             mask_full = batch["mask_pixel_values"].to(
    #                 device=latents.device,
    #                 dtype=weight_dtype,
    #             )
    #             mask_crop = crop_mask_by_centers(
    #                 mask_full,
    #                 centers,
    #                 crop_pixel_size=512,
    #             )
    #             with accelerator.autocast():
    #                 mask_image_embeds = unet.conv_in.project_mask_crop(mask_crop)

    #             inpaint_masks = batch["inpaint_masks"].to(weight_dtype)
    #             inpaint_mask_crop = crop_mask_by_centers(
    #                 inpaint_masks,
    #                 centers,
    #                 crop_pixel_size=512,
    #             )
    #             # Sample noise that we'll add to the latents
    #             noise = torch.randn_like(latents)
    #             bsz = latents.shape[0]
    #             # Sample a random timestep for each image
    #             timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
    #             timesteps = timesteps.long()
    #             # Add noise to the latents according to the noise magnitude at each timestep
    #             # (this is the forward diffusion process)
    #             noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    #             # centers: (B, 2) -> (B, 2, 1, 1) -> (B, 2, H_latent, W_latent)
    #             center_maps = centers.to(
    #                 device=latents.device,
    #                 dtype=original_image_embeds.dtype,
    #             ).view(bsz, 2, 1, 1).expand(
    #                 bsz,
    #                 2,
    #                 noisy_latents.shape[2],
    #                 noisy_latents.shape[3],
    #             )
    #             with accelerator.autocast():
    #                 projected_global_latents = unet.conv_in.project_global_latents(original_image_embeds, original_latents_full, center_maps)

    #                 condition_embeds = torch.cat(
    #                     [original_image_embeds, mask_image_embeds, projected_global_latents, center_maps],
    #                     dim=1,
    #                 )
                
    #             # Conditioning dropout to support classifier-free guidance during inference. For more details
    #             # check out the section 3.2.1 of the original paper https://huggingface.co/papers/2211.09800.
    #             if args.conditioning_dropout_prob is not None:
    #                 random_p = torch.rand(bsz, device=latents.device, generator=generator)
    #                 # Sample masks for the edit prompts.
    #                 prompt_mask = random_p < 2 * args.conditioning_dropout_prob
    #                 prompt_mask = prompt_mask.reshape(bsz, 1, 1)
    #                 # Final text conditioning.
    #                 null_conditioning = text_encoder(tokenize_captions([""]).to(accelerator.device))[0]
    #                 encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)

    #                 # Sample masks for the original images.
    #                 image_mask_dtype = condition_embeds.dtype
    #                 image_mask = 1 - (
    #                     (random_p >= args.conditioning_dropout_prob).to(image_mask_dtype)
    #                     * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
    #                 )
    #                 image_mask = image_mask.reshape(bsz, 1, 1, 1)
    #                 # Final image conditioning.
    #                 condition_embeds = image_mask * condition_embeds

    #             # Concatenate the `original_image_embeds` with the `noisy_latents`.
    #             with accelerator.autocast():
    #                 concatenated_noisy_latents = torch.cat([noisy_latents, condition_embeds], dim=1)

    #             # Get the target for loss depending on the prediction type
    #             if noise_scheduler.config.prediction_type == "epsilon":
    #                 target = noise
    #             elif noise_scheduler.config.prediction_type == "v_prediction":
    #                 target = noise_scheduler.get_velocity(latents, noise, timesteps)
    #             else:
    #                 raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    #             with accelerator.autocast():
    #             # Predict the noise residual and compute loss
    #                 model_pred = unet(concatenated_noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
    #             # 1. 把 512×512 的 mask 下采样到 latent 尺寸 64×64

    #             #    mask 是二值图，所以必须用 nearest，不能用 bilinear。
    #             inpaint_mask_64 = F.interpolate(
    #                 inpaint_mask_crop,
    #                 size=(64, 64),
    #                 mode="nearest",
    #             )
    #             # 2. 扩展到和 model_pred / target 一样的通道数
    #             #    model_pred: [B, 4, 64, 64]
    #             #    inpaint_mask_64: [B, 1, 64, 64] -> [B, 4, 64, 64]
    #             inpaint_mask_64 = inpaint_mask_64.expand_as(model_pred)
    #             # 3. 构造权重
    #             #    mask=0 的地方权重是 1
    #             #    mask=1 的地方权重是 1 + repair_weight_alpha
    #             repair_weight_alpha = 4.0
    #             loss_weights = 1.0 + repair_weight_alpha * inpaint_mask_64.float()
    #             loss_weights = loss_weights.float()
    #             # 4. 逐像素 MSE
    #             loss_map = F.mse_loss(
    #                 model_pred.float(),
    #                 target.float(),
    #                 reduction="none",
    #             )
    #             # 5. weighted mean
    #             if random.random() < 0.7:
    #                 # 70% 使用mask加权
    #                 loss = (loss_map * loss_weights).sum() / loss_weights.sum()
    #             else:
    #                 # 30% 普通MSE
    #                 loss = F.mse_loss(
    #                     model_pred.float(),
    #                     target.float(),
    #                     reduction="mean",
    #                 )
    #             # loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

    #             # Gather the losses across all processes for logging (if we use distributed training).
    #             avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
    #             train_loss += avg_loss.item() / args.gradient_accumulation_steps

    #             # Backpropagate
    #             accelerator.backward(loss)
    #             if accelerator.sync_gradients:
    #                 accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
    #             optimizer.step()
    #             lr_scheduler.step()
    #             optimizer.zero_grad()

    #         # Checks if the accelerator has performed an optimization step behind the scenes
    #         if accelerator.sync_gradients:
    #             if args.use_ema:
    #                 ema_unet.step(unet.parameters())
    #             progress_bar.update(1)
    #             global_step += 1
    #             accelerator.log({"train_loss": train_loss}, step=global_step)
    #             train_loss = 0.0

    #             if global_step % args.checkpointing_steps == 0:
    #                 if accelerator.is_main_process:
    #                     # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    #                     if args.checkpoints_total_limit is not None:
    #                         checkpoints = os.listdir(args.output_dir)
    #                         checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
    #                         checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

    #                         # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
    #                         if len(checkpoints) >= args.checkpoints_total_limit:
    #                             num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
    #                             removing_checkpoints = checkpoints[0:num_to_remove]

    #                             logger.info(
    #                                 f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
    #                             )
    #                             logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

    #                             for removing_checkpoint in removing_checkpoints:
    #                                 removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
    #                                 shutil.rmtree(removing_checkpoint)

    #                     save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    #                     accelerator.save_state(save_path)
    #                     logger.info(f"Saved state to {save_path}")

    #         logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
    #         progress_bar.set_postfix(**logs)

    #         if global_step >= args.max_train_steps:
    #             break

    #     if accelerator.is_main_process:
    #         if (
    #             (args.val_image_url is not None)
    #             and (args.validation_prompt is not None)
    #             and (epoch % args.validation_epochs == 0)
    #         ):
    #             if args.use_ema:
    #                 # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
    #                 ema_unet.store(unet.parameters())
    #                 ema_unet.copy_to(unet.parameters())
    #             # The models need unwrapping because for compatibility in distributed training mode.
    #             pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    #                 args.pretrained_model_name_or_path,
    #                 unet=unwrap_model(unet),
    #                 text_encoder=unwrap_model(text_encoder),
    #                 vae=unwrap_model(vae),
    #                 revision=args.revision,
    #                 variant=args.variant,
    #                 torch_dtype=weight_dtype,
    #             )

    #             log_validation(
    #                 pipeline,
    #                 args,
    #                 accelerator,
    #                 generator,
    #             )

    #             if args.use_ema:
    #                 # Switch back to the original UNet parameters.
    #                 ema_unet.restore(unet.parameters())

    #             del pipeline
    #             torch.cuda.empty_cache()

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=unwrap_model(text_encoder),
            vae=unwrap_model(vae),
            unet=unwrap_model(unet),
            revision=args.revision,
            variant=args.variant,
        )
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

        if (args.val_image_url is not None) and (args.validation_prompt is not None):
            log_validation(
                pipeline,
                args,
                accelerator,
                generator,
            )
    accelerator.end_training()


if __name__ == "__main__":
    main()
