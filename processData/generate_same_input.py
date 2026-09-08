import sys
import gc
import os
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import PIL.Image

from diffusers import UNet2DConditionModel
from safetensors.torch import load_file


# ============================================================
# path
# ============================================================

PROJECT_ROOT = Path(
    "/root/autodl-tmp/instruct-Pix2Pix"
)

sys.path.append(
    str(PROJECT_ROOT)
)

from processData.pipline import (
    SlidingWindowInstructPix2PixPipeline
)

from new_conv_in import (
    ContentAwarePositionGatedConvIn
)


# ============================================================
# config
# ============================================================

BASE_MODEL_PATH = (
    PROJECT_ROOT
    / "base_models"
    / "SD1.5"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "trained_model"
    / "latent_20"
)


# ------------------------------------------------------------
# 旧版本结果目录
#
# 这里面的：
#
# xxx_original.png
# xxx_input.png
# xxx_mask.png
# xxx_generate.png
#
# 直接作为新实验的输入来源
# ------------------------------------------------------------

OLD_RESULT_DIR = (
    PROJECT_ROOT
    / "results_global_noisy_inpaint_step100000"
)


# ------------------------------------------------------------
# 新版本输出目录
# ------------------------------------------------------------

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results_global_mask_aware_routing_step100000"
)

# ------------------------------------------------------------
# 原来的 val.txt
#
# 继续使用它的顺序非常重要，
# 因为原来 generator 用的是 SEED + index
# ------------------------------------------------------------

VAL_TXT = (
    PROJECT_ROOT
    / "datasets"
    / "UAV-PDD"
    / "ImageSets"
    / "Main"
    / "val.txt"
)


PROMPT = (
    "Road damage images acquired by unmanned aerial vehicles"
)

# None 表示使用所有 CUDA 可见卡；例如 CUDA_VISIBLE_DEVICES=2,3 时，
# 此处可见卡编号仍是 0,1。若只想用部分卡，改为例如 (0, 2)。
GPU_IDS: tuple[int, ...] | None = None

DTYPE = torch.float32

CROP_SIZE = 512

CROP_OVERLAP = 128

GUIDANCE_SCALE = 3.0

IMAGE_GUIDIDANCE_SCALE = 2.0

STEPS = 50

CROP_BATCH = 4

SEED = 42


# ------------------------------------------------------------
# mask-aware 动态窗口路由实验
#
# 不再对整图铺设滑窗。每一步只运行与 inpaint mask 相交的 crop；边缘
# crop 必须同时看到已知区域，深中心则由相互重叠的桥接 crop 覆盖。
# 每个有效 crop 在合法范围内独立抖动，最终仅融合/更新修复 mask 内的
# latent，已知区域仅作为条件上下文。
#
# 本次关闭 confidence，隔离检验“窗口路由”本身是否优于 MultiDiffusion
# 的整图均匀滑窗与平均融合。
# ------------------------------------------------------------

CONFIDENCE_MIN_WEIGHT = 0.05
CONFIDENCE_CONTEXT_RADIUS = 128
CONFIDENCE_AGREEMENT_STRENGTH = 0.0
CONFIDENCE_TEMPORAL_STRENGTH = 0.0
CONFIDENCE_PROPAGATION_GAMMA = 3.0
WINDOW_JITTER_RATIO = 0.40
GLOBAL_CONTEXT_STRENGTH = 0.0
MASK_AWARE_WINDOW_ROUTING = True
MASK_AWARE_MIN_KNOWN_FRACTION = 0.50
MASK_AWARE_JITTER_ATTEMPTS = 16

MAX_IMAGES = 1000

SKIP_EXISTING = True


# ============================================================
# utils
# ============================================================

def clear_memory():

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# load model
# ============================================================

def load_pipeline(device: torch.device):

    print(f"[{device}] Loading UNet...", flush=True)

    unet = (
        UNet2DConditionModel
        .from_pretrained(
            str(BASE_MODEL_PATH),
            subfolder="unet",
            torch_dtype=DTYPE,
        )
    )

    # --------------------------------------------------------
    # replace conv_in
    # --------------------------------------------------------

    ContentAwarePositionGatedConvIn.apply_to_unet(
        unet
    )

    # --------------------------------------------------------
    # load checkpoint
    # --------------------------------------------------------

    weight_file = (
        MODEL_PATH
        / "unet"
        / "diffusion_pytorch_model.safetensors"
    )

    state_dict = load_file(
        str(weight_file)
    )

    missing_keys, unexpected_keys = (
        unet.load_state_dict(
            state_dict,
            strict=False
        )
    )

    print(f"[{device}] missing keys: {len(missing_keys)}", flush=True)

    print(f"[{device}] unexpected keys: {len(unexpected_keys)}", flush=True)

    # --------------------------------------------------------
    # pipeline
    # --------------------------------------------------------

    print(f"[{device}] Loading pipeline...", flush=True)

    pipe = (
        SlidingWindowInstructPix2PixPipeline
        .from_pretrained(
            str(MODEL_PATH),
            unet=unet,
            torch_dtype=DTYPE,
            safety_checker=None,
        )
    )

    pipe.to(device)

    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()

    return pipe


# ============================================================
# reconstruct inpaint mask
# ============================================================

def build_inpaint_mask(
    original,
    edited,
    threshold=0
):
    """
    根据旧实验保存下来的：

        original
        input

    恢复原来的 inpaint 区域。

    原理：

        original != input

    的位置就是被 fill_box_with_mean_color
    修改过的位置。


    Parameters
    ----------
    original:
        PIL RGB image

    edited:
        PIL RGB image

    threshold:
        RGB 最大差值超过多少认为发生修改。

        因为这里输入是 PNG，
        理论上 threshold=0 就可以。
    """

    original_np = np.asarray(
        original,
        dtype=np.int16
    )

    edited_np = np.asarray(
        edited,
        dtype=np.int16
    )

    # --------------------------------------------------------
    # RGB absolute difference
    # --------------------------------------------------------

    diff = np.abs(
        original_np - edited_np
    )

    # --------------------------------------------------------
    # 任意 RGB channel 不一样
    # 就认为该像素属于 inpaint area
    # --------------------------------------------------------

    changed = (
        np.max(
            diff,
            axis=2
        )
        > threshold
    )

    # --------------------------------------------------------
    # bool -> uint8
    #
    # known   = 0
    # inpaint = 255
    # --------------------------------------------------------

    mask_np = (
        changed.astype(
            np.uint8
        )
        * 255
    )

    inpaint_mask = PIL.Image.fromarray(
        mask_np,
        mode="L"
    )

    return inpaint_mask


# ============================================================
# prepare input
# ============================================================

def prepare_input(
    name
):
    """
    直接从旧实验目录中加载。

    不再：
        - 读取 XML
        - fill_box_with_mean_color
        - 随机产生区域

    从而确保新旧实验输入完全相同。
    """

    original_file = (
        OLD_RESULT_DIR
        / f"{name}_original.png"
    )

    input_file = (
        OLD_RESULT_DIR
        / f"{name}_input.png"
    )

    mask_file = (
        OLD_RESULT_DIR
        / f"{name}_mask.png"
    )

    # --------------------------------------------------------
    # existence check
    # --------------------------------------------------------

    if not original_file.exists():

        raise FileNotFoundError(
            f"Missing original: {original_file}"
        )

    if not input_file.exists():

        raise FileNotFoundError(
            f"Missing input: {input_file}"
        )

    if not mask_file.exists():

        raise FileNotFoundError(
            f"Missing mask: {mask_file}"
        )

    # --------------------------------------------------------
    # read
    # --------------------------------------------------------

    original = (
        PIL.Image
        .open(original_file)
        .convert("RGB")
    )

    edited = (
        PIL.Image
        .open(input_file)
        .convert("RGB")
    )

    mask = (
        PIL.Image
        .open(mask_file)
        .convert("L")
    )

    # --------------------------------------------------------
    # size check
    # --------------------------------------------------------

    if original.size != edited.size:

        raise ValueError(
            f"{name}: original and input size mismatch: "
            f"{original.size} vs {edited.size}"
        )

    if mask.size != edited.size:

        raise ValueError(
            f"{name}: mask and input size mismatch: "
            f"{mask.size} vs {edited.size}"
        )

    # --------------------------------------------------------
    # reconstruct the old inpaint mask
    # --------------------------------------------------------

    inpaint_mask = build_inpaint_mask(
        original,
        edited
    )

    return (
        original,
        edited,
        mask,
        inpaint_mask
    )


# ============================================================
# inference one image
# ============================================================

@torch.inference_mode()
def inference_one(
    pipe,
    name,
    index
):

    save_dir = OUTPUT_DIR

    output_file = (
        save_dir
        / f"{name}_generate.png"
    )

    # --------------------------------------------------------
    # skip
    # --------------------------------------------------------

    if (
        SKIP_EXISTING
        and output_file.exists()
    ):

        print(
            f"{name} has existed. skip"
        )

        return

    # --------------------------------------------------------
    # load EXACT same old inputs
    # --------------------------------------------------------

    (
        original,
        edited,
        mask,
        inpaint_mask
    ) = prepare_input(
        name
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # 使用和原实验完全一样的 seed
    #
    # 原来：
    #
    # generator.manual_seed(SEED + index)
    #
    # 现在仍然保持一致
    # --------------------------------------------------------

    generator = (
        torch.Generator(
            device="cpu"
        )
        .manual_seed(
            SEED + index
        )
    )

    # --------------------------------------------------------
    # inference
    # --------------------------------------------------------

    result = pipe(

        prompt=PROMPT,

        # ====================================================
        # EXACT SAME OLD INPUT
        # ====================================================

        image=edited,

        mask=mask,

        inpaint_mask=inpaint_mask,

        # ====================================================

        crop_size=CROP_SIZE,

        crop_overlap=CROP_OVERLAP,

        guidance_scale=GUIDANCE_SCALE,

        image_guidance_scale=
            IMAGE_GUIDIDANCE_SCALE,

        num_inference_steps=STEPS,

        crop_process_batch_size=
            CROP_BATCH,

        generator=generator,

        output_type="pil",

        # 关闭 confidence，隔离评估 mask-aware 窗口路由。
        confidence_min_weight=CONFIDENCE_MIN_WEIGHT,

        confidence_context_radius=CONFIDENCE_CONTEXT_RADIUS,

        confidence_agreement_strength=
            CONFIDENCE_AGREEMENT_STRENGTH,

        confidence_temporal_strength=
            CONFIDENCE_TEMPORAL_STRENGTH,

        confidence_propagation_gamma=
            CONFIDENCE_PROPAGATION_GAMMA,

        window_jitter_ratio=WINDOW_JITTER_RATIO,

        mask_aware_window_routing=MASK_AWARE_WINDOW_ROUTING,

        mask_aware_min_known_fraction=
            MASK_AWARE_MIN_KNOWN_FRACTION,

        mask_aware_jitter_attempts=
            MASK_AWARE_JITTER_ATTEMPTS,

        # 模型仅在 crop 上训练；全图分支暂不参与本次窗口路由消融。
        global_context_strength=GLOBAL_CONTEXT_STRENGTH,

    ).images[0]

    # --------------------------------------------------------
    # save
    #
    # 同样把 input 保存一次，
    # 后面可以直接 md5 / sha256 检查新旧是否一样
    # --------------------------------------------------------

    original.save(
        save_dir
        / f"{name}_original.png"
    )

    edited.save(
        save_dir
        / f"{name}_input.png"
    )

    mask.save(
        save_dir
        / f"{name}_mask.png"
    )

    # 建议这次顺便保存下来
    # 下一次就不需要再重建了
    inpaint_mask.save(
        save_dir
        / f"{name}_inpaint_mask.png"
    )

    result.save(
        output_file
    )

    print(
        "done:",
        name
    )


# ============================================================
# main
# ============================================================

def load_sample_names() -> list[str]:
    """按 val.txt 原始顺序读取样本，索引必须稳定以保持原有 seed。"""
    with open(VAL_TXT, "r") as file:
        names = [line.strip() for line in file if line.strip()]
    return names[:MAX_IMAGES] if MAX_IMAGES else names


def select_gpu_ids() -> list[int]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; multi-GPU generation requires CUDA.")

    visible_count = torch.cuda.device_count()
    gpu_ids = list(range(visible_count)) if GPU_IDS is None else list(GPU_IDS)
    if not gpu_ids:
        raise ValueError("GPU_IDS 不能为空。")
    invalid_ids = [gpu_id for gpu_id in gpu_ids if not 0 <= gpu_id < visible_count]
    if invalid_ids:
        raise ValueError(
            f"GPU_IDS={invalid_ids} 不在可见 CUDA 卡范围 0..{visible_count - 1} 内。"
        )
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("GPU_IDS 不能包含重复 GPU。")
    return gpu_ids


def worker(
    rank: int,
    gpu_ids: list[int],
    indexed_names: list[tuple[int, str]],
) -> None:
    """一个独立进程绑定一张 GPU，并处理自己按轮询分配到的样本。"""
    gpu_id = gpu_ids[rank]
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    assigned_samples = indexed_names[rank::len(gpu_ids)]
    print(
        f"[GPU {gpu_id}] assigned {len(assigned_samples)} images; loading model...",
        flush=True,
    )
    pipe = load_pipeline(device)

    success = 0
    skipped_or_missing = 0
    failed = 0
    for local_index, (original_index, name) in enumerate(assigned_samples, start=1):
        input_file = OLD_RESULT_DIR / f"{name}_input.png"
        original_file = OLD_RESULT_DIR / f"{name}_original.png"
        mask_file = OLD_RESULT_DIR / f"{name}_mask.png"
        if not input_file.exists() or not original_file.exists() or not mask_file.exists():
            print(f"[GPU {gpu_id}] missing old input: {name}", flush=True)
            skipped_or_missing += 1
            continue

        try:
            # original_index 来自完整 val.txt 的顺序，多卡分片不会改变 seed。
            inference_one(pipe, name, original_index)
            success += 1
            print(
                f"[GPU {gpu_id}] {local_index}/{len(assigned_samples)} done: {name}",
                flush=True,
            )
        except Exception as error:
            print(f"[GPU {gpu_id}] failed {name}: {error!r}", flush=True)
            failed += 1

        if local_index % 10 == 0:
            clear_memory()

    del pipe
    clear_memory()
    print(
        f"[GPU {gpu_id}] finished; success={success}, "
        f"missing={skipped_or_missing}, failed={failed}",
        flush=True,
    )


def main():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    names = load_sample_names()
    if not names:
        print("val.txt 中没有可处理的样本。")
        return

    available = sum((OLD_RESULT_DIR / f"{name}_input.png").exists() for name in names)
    gpu_ids = select_gpu_ids()
    worker_count = min(len(gpu_ids), len(names))
    gpu_ids = gpu_ids[:worker_count]
    indexed_names = list(enumerate(names))

    print(f"VAL total: {len(names)}", flush=True)
    print(f"Old inputs available: {available}", flush=True)
    print(f"Using {worker_count} GPUs: {gpu_ids}", flush=True)

    # spawn 为每张卡创建独立 CUDA 上下文；不能改成 fork。
    mp.spawn(
        worker,
        args=(gpu_ids, indexed_names),
        nprocs=worker_count,
        join=True,
    )
    print(f"All workers finished. Results in: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
