"""Test the trained 512x512 whole-image thumbnail inpainting branch.

Load saved inputs and repair masks using the same naming convention as
``generate_same_input.py``, then run only ``ThumbnailInstructPix2PixPipeline``.
Missing repair masks are reconstructed from original/input differences;
repair regions are never sampled randomly.
"""

import gc
import sys
from pathlib import Path

import cv2
import numpy as np
import PIL.Image
import torch
from diffusers import UNet2DConditionModel
from safetensors.torch import load_file


# Derive the repository root from this file rather than relying on the old
# hard-coded server path.  This works for both ``instruct-Pix2Pix`` and any
# copied checkout location.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from new_conv_in import ContentAwarePositionGatedConvIn
from processData.pipline_thumbnail import ThumbnailInstructPix2PixPipeline


# ============================================================
# config: edit only this section on the server
# ============================================================

BASE_MODEL_PATH = PROJECT_ROOT / "base_models" / "SD1.5"

# Set this to either the training output directory containing ``unet/``, or to
# a specific ``checkpoint-xxxxx`` directory containing ``unet/``.
MODEL_PATH = PROJECT_ROOT / "trained_model" / "latent_20"

# 与 generate_same_input.py 相同：读取已有实验的 input，而不随机生成区域。
OLD_RESULT_DIR = PROJECT_ROOT / "results_global_noisy_inpaint_step100000"
VAL_TXT = (
    PROJECT_ROOT
    / "datasets"
    / "UAV-PDD"
    / "ImageSets"
    / "Main"
    / "val.txt"
)
OUTPUT_DIR = PROJECT_ROOT / "results_thumbnail_test"

PROMPT = "Road damage images acquired by unmanned aerial vehicles"
DEVICE = "cuda"
# Keep fp32 for this initial thumbnail test.  The custom conv_in creates new
# condition branches after the base UNet is loaded; fp32 guarantees their
# input, weights, and bias use the same dtype.  A 512x512 thumbnail is small
# enough that this is the safest default.
DTYPE = torch.float32
STEPS = 50
GUIDANCE_SCALE = 3.0
IMAGE_GUIDANCE_SCALE = 2.0
SEED = 42

# Keep this list empty to use the first MAX_IMAGES names in VAL_TXT.  To test
# specific images, write e.g. TEST_NAMES = ["lr_00001_bottom_left"].
TEST_NAMES = []
MAX_IMAGES = 500
SKIP_EXISTING = False

# Only used when the saved inpaint mask is unavailable.
INPAINT_DIFF_THRESHOLD = 3
INPAINT_MIN_COMPONENT_AREA = 9


def crop_to_vae_multiple(image: PIL.Image.Image, scale_factor: int = 8):
    width, height = image.size
    width -= width % scale_factor
    height -= height % scale_factor
    if width <= 0 or height <= 0:
        raise ValueError(f"Image is too small after VAE alignment: {image.size}")
    return image.crop((0, 0, width, height))


def letterbox_preview(image: PIL.Image.Image, size: int, is_mask: bool):
    try:
        image_resample = PIL.Image.Resampling.BICUBIC
        mask_resample = PIL.Image.Resampling.NEAREST
    except AttributeError:
        image_resample = PIL.Image.BICUBIC
        mask_resample = PIL.Image.NEAREST

    width, height = image.size
    scale = min(size / width, size / height)
    resized_width = max(1, min(size, round(width * scale)))
    resized_height = max(1, min(size, round(height * scale)))
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2

    canvas = PIL.Image.new(image.mode, (size, size), color=0)
    canvas.paste(
        image.resize(
            (resized_width, resized_height),
            resample=mask_resample if is_mask else image_resample,
        ),
        (left, top),
    )
    return canvas


def resolve_weight_file(model_path: Path):
    candidates = (
        model_path / "unet" / "diffusion_pytorch_model.safetensors",
        model_path / "unet" / "diffusion_pytorch_model.bin",
        model_path / "diffusion_pytorch_model.safetensors",
        model_path / "diffusion_pytorch_model.bin",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    checked = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find trained UNet weights. Checked:\n  {checked}")


def load_pipeline():
    print("Loading base UNet...")
    unet = UNet2DConditionModel.from_pretrained(
        str(BASE_MODEL_PATH),
        subfolder="unet",
        torch_dtype=DTYPE,
    )
    ContentAwarePositionGatedConvIn.apply_to_unet(unet)

    weight_file = resolve_weight_file(MODEL_PATH)
    print(f"Loading trained UNet: {weight_file}")
    if weight_file.suffix == ".safetensors":
        state_dict = load_file(str(weight_file), device="cpu")
    else:
        state_dict = torch.load(weight_file, map_location="cpu")
    unet.load_state_dict(state_dict, strict=True)
    unet.to(dtype=DTYPE)

    pipe = ThumbnailInstructPix2PixPipeline.from_pretrained(
        str(BASE_MODEL_PATH),
        unet=unet,
        torch_dtype=DTYPE,
        safety_checker=None,
    )
    pipe.to(DEVICE)
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    pipe.set_progress_bar_config(disable=False)
    return pipe


def build_inpaint_mask(
    original,
    edited,
    threshold=INPAINT_DIFF_THRESHOLD,
    min_component_area=INPAINT_MIN_COMPONENT_AREA,
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

        旧 original/input 的源 JPEG 可能分别由 PIL/OpenCV 解码，即使
        最后都保存为 PNG，已知区也可能已经固化了很小的像素差。因此
        默认保留一个小容差。
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

    # 去掉超过容差后仍残留的孤立解码噪声。fill_box_with_mean_color 产生
    # 的修复区域是连续矩形，因此不会依赖小于该面积的离散连通域。
    changed_u8 = changed.astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        changed_u8,
        connectivity=8,
    )
    cleaned = np.zeros_like(changed_u8)
    for component_index in range(1, component_count):
        component_area = stats[
            component_index,
            cv2.CC_STAT_AREA,
        ]
        if component_area >= min_component_area:
            cleaned[labels == component_index] = 1

    # 填补均值色恰好与原像素相同造成的细小孔洞。
    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
    )

    # known=0, inpaint=255
    mask_np = cleaned * 255

    inpaint_mask = PIL.Image.fromarray(
        mask_np,
        mode="L"
    )

    return inpaint_mask


def prepare_input(name: str):
    paths = {
        "original": OLD_RESULT_DIR / f"{name}_original.png",
        "input": OLD_RESULT_DIR / f"{name}_input.png",
        "mask": OLD_RESULT_DIR / f"{name}_mask.png",
    }
    for kind, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {kind}: {path}")

    def read_image(path, mode):
        with PIL.Image.open(path) as image:
            return image.convert(mode)

    target = read_image(paths["original"], "RGB")
    filled = read_image(paths["input"], "RGB")
    mask = read_image(paths["mask"], "L")
    if target.size != filled.size or mask.size != filled.size:
        raise ValueError(f"{name}: original, input, and mask sizes must match")

    inpaint_file = OLD_RESULT_DIR / f"{name}_inpaint_mask.png"
    if inpaint_file.is_file():
        inpaint_mask = read_image(inpaint_file, "L").point(
            lambda value: 255 if value >= 128 else 0
        )
        mask_source = "saved"
    else:
        inpaint_mask = build_inpaint_mask(target, filled)
        mask_source = f"reconstructed(threshold={INPAINT_DIFF_THRESHOLD})"
    if inpaint_mask.size != filled.size:
        raise ValueError(f"{name}: inpaint mask and input sizes must match")

    # Apply identical VAE alignment to all four views.
    target, filled, mask, inpaint_mask = (
        crop_to_vae_multiple(image)
        for image in (target, filled, mask, inpaint_mask)
    )
    if inpaint_mask.getbbox() is None:
        raise ValueError(f"{name}: inpaint mask is empty after VAE alignment")
    print(f"{name}: inpaint_mask={mask_source}, bbox={inpaint_mask.getbbox()}")
    return target, filled, mask, inpaint_mask


def select_names():
    if TEST_NAMES:
        return TEST_NAMES

    with VAL_TXT.open("r", encoding="utf-8") as file:
        names = [line.strip() for line in file if line.strip()]
    return names[:MAX_IMAGES] if MAX_IMAGES > 0 else names


@torch.inference_mode()
def inference_one(pipe, name: str, index: int):
    output_file = OUTPUT_DIR / f"{name}_thumbnail_generate.png"
    if SKIP_EXISTING and output_file.exists():
        print(f"skip existing: {name}")
        return

    target, filled, mask, inpaint_mask = prepare_input(name)
    generator = torch.Generator(device="cpu").manual_seed(SEED + index)
    result = pipe(
        prompt=PROMPT,
        image=filled,
        mask=mask,
        inpaint_mask=inpaint_mask,
        thumbnail_size=512,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE_SCALE,
        image_guidance_scale=IMAGE_GUIDANCE_SCALE,
        generator=generator,
        output_type="pil",
    ).images[0]

    # Save all five views needed to check whether the thumbnail branch works.
    letterbox_preview(target, 512, is_mask=False).save(
        OUTPUT_DIR / f"{name}_thumbnail_target.png"
    )
    letterbox_preview(filled, 512, is_mask=False).save(
        OUTPUT_DIR / f"{name}_thumbnail_input.png"
    )
    letterbox_preview(mask, 512, is_mask=True).save(
        OUTPUT_DIR / f"{name}_thumbnail_mask.png"
    )
    letterbox_preview(inpaint_mask, 512, is_mask=True).save(
        OUTPUT_DIR / f"{name}_thumbnail_inpaint_mask.png"
    )
    result.save(output_file)
    print(f"saved: {output_file}")


def main():
    if not OLD_RESULT_DIR.is_dir():
        raise FileNotFoundError(f"Missing old result directory: {OLD_RESULT_DIR}")
    if STEPS <= 0:
        raise ValueError("STEPS must be positive")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    names = select_names()
    if not names:
        raise ValueError("No image names selected")

    print(f"Testing {len(names)} thumbnail image(s)")
    print(f"Output directory: {OUTPUT_DIR}")
    pipe = load_pipeline()
    failures = []

    for index, name in enumerate(names):
        try:
            inference_one(pipe, name, index)
        except Exception as error:
            failures.append((name, str(error)))
            print(f"failed: {name}\n{error}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if failures:
        failure_file = OUTPUT_DIR / "thumbnail_failed_images.txt"
        failure_file.write_text(
            "\n".join(f"{name}\n{reason}\n{'-' * 80}" for name, reason in failures),
            encoding="utf-8",
        )
        print(f"Failures: {len(failures)}; details: {failure_file}")
    print(f"Finished thumbnail test: {len(names) - len(failures)}/{len(names)} succeeded")


if __name__ == "__main__":
    main()
