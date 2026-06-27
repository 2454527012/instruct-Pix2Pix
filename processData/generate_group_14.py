import os
import sys
from pathlib import Path
from collections import Counter

sys.path.append("/root/autodl-tmp/instruct/instruct-pix2pix")

import PIL.Image
import torch
from safetensors.torch import load_file
from diffusers import UNet2DConditionModel

from processData.pipline_14 import SlidingWindowInstructPix2PixPipeline
from new_conv_in import ContentAwarePositionGatedConvIn


# ==========================
# 路径配置
# ==========================
base_model_path = "/root/autodl-tmp/instruct/instruct-pix2pix/models/stable-diffusion-v1-5"
model_path = "/root/autodl-tmp/instruct/instruct-pix2pix/outputs/ip2p_gate_512_20000"

input_dir = "/root/autodl-tmp/instruct/instruct-pix2pix/datasets/UAVPDD/patched_images"
output_dir = "/root/autodl-tmp/instruct/instruct-pix2pix/results_20000_3.0_2.0_75"

val_txt_path = "/root/autodl-tmp/instruct/instruct-pix2pix/datasets/UAVPDD/ImageSets/Main/val.txt"

os.makedirs(output_dir, exist_ok=True)

failed_log_path = Path(output_dir) / "failed_images.txt"


def check_model_weights(model):
    for name, p in model.named_parameters():
        if torch.isnan(p).any() or torch.isinf(p).any():
            raise RuntimeError(f"UNet weight NaN/Inf: {name}")
    print("UNet weights OK")


def find_image_file(input_dir, image_name):
    input_dir = Path(input_dir)
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

    for ext in image_extensions:
        image_path = input_dir / f"{image_name}{ext}"
        if image_path.exists():
            return image_path

    return None


# ==========================
# 读取 val.txt
# ==========================
with open(val_txt_path, "r", encoding="utf-8") as f:
    val_names = [
        line.strip()
        for line in f.readlines()
        if line.strip()
    ]

print("=" * 60)
print(f"Val txt: {val_txt_path}")
print(f"Val images count: {len(val_names)}")
print("=" * 60)


# ==========================
# 加载 UNet
# ==========================
print("Loading UNet...")

unet = UNet2DConditionModel.from_pretrained(
    base_model_path,
    subfolder="unet",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=False,
)

ContentAwarePositionGatedConvIn.apply_to_unet(unet)

state_dict = load_file(
    f"{model_path}/unet/diffusion_pytorch_model.safetensors"
)

print("state_dict dtype:", Counter(v.dtype for v in state_dict.values()))

missing, unexpected = unet.load_state_dict(state_dict, strict=False)

print("missing keys:", missing)
print("unexpected keys:", unexpected)
print("conv_in type:", type(unet.conv_in))
print(unet.conv_in)

check_model_weights(unet)


# ==========================
# 加载 Pipeline
# ==========================
print("Loading Pipeline...")

pipe = SlidingWindowInstructPix2PixPipeline.from_pretrained(
    model_path,
    unet=unet,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=False,
    device_map=None,
)

pipe = pipe.to("cuda")
pipe.set_progress_bar_config(disable=False)

print("Pipeline Ready")


# ==========================
# 推理配置
# ==========================
prompt = "Restore to aerial view of road damage taken by drone"

success = 0
failed = 0
skipped = 0
not_found = 0
failed_files = []

total = len(val_names)


# ==========================
# 根据 val.txt 批量处理
# ==========================
for idx, image_name in enumerate(val_names, start=1):

    image_file = find_image_file(input_dir, image_name)

    if image_file is None:
        not_found += 1
        failed_files.append((image_name, "Image file not found"))
        print(f"[{idx}/{total}] Not found: {image_name}")
        continue

    output_file = Path(output_dir) / f"{image_name}.png"

    if output_file.exists():
        skipped += 1
        print(f"[{idx}/{total}] Skip existing: {image_name}")
        continue

    try:
        print(f"[{idx}/{total}] Processing: {image_file.name}")

        original_image = PIL.Image.open(image_file).convert("RGB")

        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                image=original_image,
                crop_size=512,
                crop_overlap=128,
                guidance_scale=3.0,
                image_guidance_scale=2.0,
                num_inference_steps=75,
                crop_process_batch_size=4,
                safety_checker=None,
            ).images[0]

        result.save(output_file)

        success += 1
        print(f"[{idx}/{total}] Saved: {output_file}")

        del original_image
        del result
        torch.cuda.empty_cache()

    except Exception as e:
        failed += 1
        failed_files.append((str(image_file), str(e)))

        print(f"[{idx}/{total}] Failed: {image_file.name}")
        print(f"Error: {e}")

        torch.cuda.empty_cache()


# ==========================
# 保存失败日志
# ==========================
if failed_files:
    with open(failed_log_path, "w", encoding="utf-8") as f:
        for file_path, error_msg in failed_files:
            f.write(f"{file_path}\n")
            f.write(f"{error_msg}\n")
            f.write("-" * 80 + "\n")

    print(f"Failed log saved to: {failed_log_path}")


# ==========================
# 最终统计
# ==========================
print("=" * 60)
print("Val batch inference finished")
print(f"Total in val.txt : {total}")
print(f"Success          : {success}")
print(f"Skipped          : {skipped}")
print(f"Failed           : {failed}")
print(f"Not found        : {not_found}")
print(f"Output dir       : {output_dir}")
print("=" * 60)