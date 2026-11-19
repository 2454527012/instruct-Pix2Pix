"""使用 8 通道 baseline 批量生成图片。

从仓库根目录运行 python -m processData.generate_same_input_slide_8_random_window_average。
"""

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.multiprocessing as mp


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 模型、旧实验输入和新生成结果的默认路径。
MODEL_PATH = PROJECT_ROOT / "trained_model" / "latent_08_orig"
INPUT_DIR = PROJECT_ROOT / "results_global_noisy_inpaint_step100000"
OUTPUT_DIR = PROJECT_ROOT / "results_slide_8_random_window_average"
VAL_TXT = PROJECT_ROOT / "datasets" / "UAV-PDD" / "ImageSets" / "Main" / "val.txt"

# 与 generate_same_input.py 的原实验设置对齐。
PROMPT = "Road damage images acquired by unmanned aerial vehicles"
DTYPE = torch.float32
CROP_SIZE = 512
CROP_OVERLAP = 128
GUIDANCE_SCALE = 3.0
IMAGE_GUIDANCE_SCALE = 2.0
STEPS = 50
CROP_BATCH = 4
SEED = 42
MAX_IMAGES = 10000
MAX_WINDOW_JITTER_PIXELS = 256
MASK_AWARE_MIN_KNOWN_FRACTION = 0.50
MASK_AWARE_JITTER_ATTEMPTS = 16

# 同时支持直接运行 python processData/generate_same_input_slide_8_random_window_average.py。
if not __package__:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-model", type=Path, default=MODEL_PATH, help="8 通道模型的完整导出目录")
    parser.add_argument("--inputs", type=Path, default=INPUT_DIR, help="旧实验保存的输入图片目录")
    parser.add_argument("--val-list", type=Path, default=VAL_TXT, help="验证集样本列表")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="生成结果目录")
    parser.add_argument("--max-images", type=int, default=MAX_IMAGES, help="0 表示全部样本")
    parser.add_argument("--gpu-ids", type=int, nargs="+", help="CUDA 可见卡编号，默认使用所有可见卡")
    parser.add_argument("--overwrite", action="store_true", help="重新生成已有结果")
    args = parser.parse_args()
    if args.max_images < 0:
        parser.error("--max-images 必须大于等于 0")
    if args.inputs.resolve() == args.output.resolve():
        parser.error("--output 必须与 --inputs 使用不同目录")
    return args


def load_pipeline(args, device):
    from processData.pipline_slide_8_random_window_average import (
        RandomWindowAverageInstructPix2PixPipeline,
    )

    pipe = RandomWindowAverageInstructPix2PixPipeline.from_pretrained(
        str(args.baseline_model), torch_dtype=DTYPE, safety_checker=None
    )
    if pipe.unet.config.in_channels != 8:
        raise ValueError("--baseline-model 必须指向 8 通道模型的完整导出目录。")
    pipe.to(device)
    for module in (pipe.unet, pipe.vae, pipe.text_encoder):
        module.eval()
    return pipe


def load_inputs(directory, name):
    with Image.open(directory / f"{name}_original.png") as image:
        original = image.convert("RGB")
    with Image.open(directory / f"{name}_input.png") as image:
        edited = image.convert("RGB")
    if original.size != edited.size:
        raise ValueError(f"{name} 的原图和输入图尺寸不一致。")

    # 像旧实验一样，用原图和均值填充图的 RGB 差异重建修复区域。
    changed = np.any(np.asarray(original) != np.asarray(edited), axis=-1)
    inpaint_mask = Image.fromarray(changed.astype(np.uint8) * 255)
    return original, edited, inpaint_mask


def select_gpu_ids(requested_ids):
    if not torch.cuda.is_available():
        raise RuntimeError("图片生成需要可用的 CUDA GPU。")
    visible_count = torch.cuda.device_count()
    gpu_ids = list(range(visible_count)) if requested_ids is None else requested_ids
    if any(gpu_id < 0 or gpu_id >= visible_count for gpu_id in gpu_ids):
        raise ValueError(f"GPU 编号必须在 0 到 {visible_count - 1} 之间。")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("GPU 编号不能重复。")
    return gpu_ids


def worker(rank, gpu_ids, samples, args):
    gpu_id = gpu_ids[rank]
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    assigned = samples[rank::len(gpu_ids)]
    failures = 0
    print(f"[GPU {gpu_id}] 分配到 {len(assigned)} 张图，正在加载 8 通道模型", flush=True)
    pipe = load_pipeline(args, device)
    for index, name in assigned:
        output = args.output / f"{name}_generate.png"
        if output.exists() and not args.overwrite:
            continue
        try:
            original, edited, inpaint_mask = load_inputs(args.inputs, name)
            result = pipe(
                prompt=PROMPT,
                image=edited,
                inpaint_mask=inpaint_mask,
                crop_size=CROP_SIZE,
                crop_overlap=CROP_OVERLAP,
                guidance_scale=GUIDANCE_SCALE,
                image_guidance_scale=IMAGE_GUIDANCE_SCALE,
                num_inference_steps=STEPS,
                crop_process_batch_size=CROP_BATCH,
                generator=torch.Generator(device="cpu").manual_seed(SEED + index),
                max_window_jitter_pixels=MAX_WINDOW_JITTER_PIXELS,
                mask_aware_window_routing=True,
                mask_aware_min_known_fraction=MASK_AWARE_MIN_KNOWN_FRACTION,
                mask_aware_jitter_attempts=MASK_AWARE_JITTER_ATTEMPTS,
            ).images[0]
            for suffix, image in (
                ("original", original), ("input", edited),
                ("inpaint_mask", inpaint_mask),
            ):
                image.save(args.output / f"{name}_{suffix}.png")
            temporary = output.with_suffix(".tmp")
            result.save(temporary, format="PNG")
            temporary.replace(output)
            print(f"[GPU {gpu_id}] {name}", flush=True)
        except (OSError, ValueError, RuntimeError) as error:
            print(f"[GPU {gpu_id}] {name}: {error}", flush=True)
            failures += 1

    if failures:
        raise RuntimeError(f"GPU {gpu_id} 有 {failures} 张生成失败，详见上方日志。")


def main():
    args = parse_args()
    names = [line.strip() for line in args.val_list.read_text().splitlines() if line.strip()]
    names = list(enumerate(names))
    if args.max_images:
        names = names[:args.max_images]
    if not names:
        raise ValueError("验证列表没有样本。")
    if len({name for _, name in names}) != len(names):
        raise ValueError("验证列表有重复样本名，会写入同一输出文件。")
    gpu_ids = select_gpu_ids(args.gpu_ids)[:len(names)]
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"共 {len(names)} 张图，使用 CUDA 设备 {gpu_ids}", flush=True)
    mp.spawn(worker, args=(gpu_ids, names, args), nprocs=len(gpu_ids), join=True)
    print(f"生成完成，结果目录：{args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
