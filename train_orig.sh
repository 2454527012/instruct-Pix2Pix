#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_instruct_pix2pix_orig.py"

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "Training script not found: ${TRAIN_SCRIPT}" >&2
    echo "Copy train_instruct_pix2pix_orig.py next to train_orig.sh." >&2
    exit 1
fi

cd "${SCRIPT_DIR}"

CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch --mixed_precision="fp16" "${TRAIN_SCRIPT}" \
    --pretrained_model_name_or_path="/root/autodl-tmp/instruct-Pix2Pix/base_models/SD1.5" \
    --train_data_dir="/root/autodl-tmp/instruct-Pix2Pix/datasets/UAV-PDD/dataset_train.jsonl" \
    --edited_image_column="edit_image" \
    --edit_prompt_column="prompt" \
    --box_column="box_xml" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1550000 \
    --checkpointing_steps=500 \
    --checkpoints_total_limit=4 \
    --learning_rate=5e-05 \
    --max_grad_norm=1 \
    --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --mixed_precision=fp16 \
    --seed=42 \
    --random_flip \
    --output_dir="/root/autodl-tmp/instruct-Pix2Pix/trained_model/latent_20_orig" \
    --dataloader_num_workers=4 \
    --enable_xformers_memory_efficient_attention \
    --resume_from_checkpoint="latest"
    # --use_ema \
    # --gradient_checkpointing
