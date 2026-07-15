#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 \
accelerate launch --mixed_precision="fp16" train_instruct_pix2pix.py \
    --pretrained_model_name_or_path="/root/autodl-tmp/instruct/instruct-pix2pix/models/stable-diffusion-v1-5" \
    --train_data_dir="/root/autodl-tmp/instruct/instruct-pix2pix/datasets/UAVPDD/dataset_train.jsonl" \
    --edited_image_column="edit_image" \
    --mask_column="mask_image" \
    --edit_prompt_column="prompt" \
    --box_column="box_xml" \
    --resolution=512 \
    --train_batch_size=2 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=20000 \
    --checkpointing_steps=2000 \
    --checkpoints_total_limit=3 \
    --learning_rate=5e-05 \
    --max_grad_norm=1 \
    --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --mixed_precision=fp16 \
    --seed=42 \
    --random_flip \
    --output_dir="/root/autodl-tmp/instruct/instruct-pix2pix/outputs/ip2p_use_latent" \
    --dataloader_num_workers=4 \
    --resume_from_checkpoint="latest" \
    --enable_xformers_memory_efficient_attention
    # --use_ema \
    # --gradient_checkpointing
