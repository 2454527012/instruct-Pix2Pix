import gc
import sys
import traceback
from collections import Counter
from pathlib import Path

import PIL.Image
import torch
from diffusers import UNet2DConditionModel
from safetensors.torch import load_file


# ============================================================
# 项目路径
# ============================================================
PROJECT_ROOT = Path(
    "/root/autodl-tmp/instruct/instruct-pix2pix"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from processData.pipline import SlidingWindowInstructPix2PixPipeline
from new_conv_in import ContentAwarePositionGatedConvIn
from process_image import fill_box_with_mean_color


# ============================================================
# 路径配置
# ============================================================
BASE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "stable-diffusion-v1-5"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "ip2p_use_latent"
)

ORIGINAL_IMAGES_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "UAVPDD"
    / "JPEGImages"
)

XML_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "UAVPDD"
    / "Annotations"
)

MASK_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "UAVPDD"
    / "masks"
)

VAL_TXT_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "UAVPDD"
    / "ImageSets"
    / "Main"
    / "val.txt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results_global_latent_test"
)


# ============================================================
# 推理配置
# ============================================================
PROMPT = (
    "Road damage images acquired by unmanned aerial "
    "vehicles (UAVs)"
)

DEVICE = "cuda"
TORCH_DTYPE = torch.float32

CROP_SIZE = 512
CROP_OVERLAP = 128

GUIDANCE_SCALE = 3.0
IMAGE_GUIDANCE_SCALE = 2.0
NUM_INFERENCE_STEPS = 50

CROP_PROCESS_BATCH_SIZE = 4

SEED = 42

# None：测试整个验证集
# 例如 10：只测试前 10 张
MAX_TEST_IMAGES = 10

# 生成图已存在时是否跳过
SKIP_EXISTING = True

# 是否检查整个 UNet 中的 NaN/Inf
CHECK_ALL_UNET_WEIGHTS = True

# 判断“接近零”的阈值
NEAR_ZERO_THRESHOLD = 1e-8


# ============================================================
# 基础工具
# ============================================================
def clear_cuda_cache():
    """回收 Python 对象并释放 CUDA 缓存。"""

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def check_model_weights(model: torch.nn.Module):
    """检查整个模型参数是否包含 NaN 或 Inf。"""

    print("=" * 80)
    print("Checking complete UNet weights for NaN/Inf...")
    print("=" * 80)

    for name, parameter in model.named_parameters():
        data = parameter.detach()

        if torch.isnan(data).any():
            raise RuntimeError(
                f"UNet weight contains NaN: {name}"
            )

        if torch.isinf(data).any():
            raise RuntimeError(
                f"UNet weight contains Inf: {name}"
            )

    print("Complete UNet weights: OK")
    print("=" * 80)


def check_conv_in_weights(
    conv_in: torch.nn.Module,
    report_path: Path | None = None,
    near_zero_threshold: float = 1e-8,
):
    """
    检查 conv_in 内各参数是否：

        1. 全零
        2. 接近全零
        3. 包含 NaN/Inf
        4. 权重是否有正常方差

    输出统计：
        shape
        numel
        zero_ratio
        near_zero_ratio
        mean
        std
        abs_mean
        abs_max
    """

    separator = "=" * 150

    report_lines = [
        separator,
        "conv_in weight inspection",
        separator,
    ]

    all_zero_parameters = []
    almost_zero_parameters = []
    nan_inf_parameters = []

    for name, parameter in conv_in.named_parameters():
        data = parameter.detach().float().cpu()

        numel = data.numel()

        if numel == 0:
            continue

        has_nan = bool(torch.isnan(data).any().item())
        has_inf = bool(torch.isinf(data).any().item())

        finite_data = torch.nan_to_num(
            data,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        zero_count = int(
            torch.count_nonzero(
                finite_data == 0
            ).item()
        )

        near_zero_count = int(
            torch.count_nonzero(
                finite_data.abs() <= near_zero_threshold
            ).item()
        )

        zero_ratio = zero_count / numel
        near_zero_ratio = near_zero_count / numel

        is_all_zero = zero_count == numel

        # 99.99% 以上都接近 0，认为可能几乎没训练
        is_almost_zero = (
            not is_all_zero
            and near_zero_ratio >= 0.9999
        )

        mean_value = finite_data.mean().item()
        std_value = finite_data.std(
            unbiased=False
        ).item()
        abs_mean = finite_data.abs().mean().item()
        abs_max = finite_data.abs().max().item()

        if is_all_zero:
            status = "ALL_ZERO"
            all_zero_parameters.append(name)

        elif is_almost_zero:
            status = "ALMOST_ZERO"
            almost_zero_parameters.append(name)

        else:
            status = "OK"

        if has_nan or has_inf:
            status += "|NAN_INF"
            nan_inf_parameters.append(name)

        line = (
            f"{name:65s} "
            f"shape={str(tuple(data.shape)):25s} "
            f"numel={numel:10d} "
            f"zero={zero_ratio:9.4%} "
            f"near_zero={near_zero_ratio:9.4%} "
            f"mean={mean_value:+.6e} "
            f"std={std_value:.6e} "
            f"abs_mean={abs_mean:.6e} "
            f"abs_max={abs_max:.6e} "
            f"status={status}"
        )

        report_lines.append(line)

    report_lines.append(separator)
    report_lines.append("Summary")
    report_lines.append(separator)

    if all_zero_parameters:
        report_lines.append(
            "ALL-ZERO parameters:"
        )

        for name in all_zero_parameters:
            report_lines.append(
                f"  - {name}"
            )
    else:
        report_lines.append(
            "No all-zero parameters found."
        )

    if almost_zero_parameters:
        report_lines.append(
            "Nearly all-zero parameters:"
        )

        for name in almost_zero_parameters:
            report_lines.append(
                f"  - {name}"
            )
    else:
        report_lines.append(
            "No nearly all-zero parameters found."
        )

    if nan_inf_parameters:
        report_lines.append(
            "NaN/Inf parameters:"
        )

        for name in nan_inf_parameters:
            report_lines.append(
                f"  - {name}"
            )
    else:
        report_lines.append(
            "No NaN/Inf parameters found."
        )

    report_lines.append(separator)

    report_text = "\n".join(report_lines)

    print(report_text)

    if report_path is not None:
        report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        report_path.write_text(
            report_text,
            encoding="utf-8",
        )

        print(
            f"conv_in weight report saved to: "
            f"{report_path}"
        )

    if nan_inf_parameters:
        raise RuntimeError(
            "conv_in 中存在 NaN 或 Inf 参数："
            + ", ".join(nan_inf_parameters)
        )

    return {
        "all_zero": all_zero_parameters,
        "almost_zero": almost_zero_parameters,
        "nan_inf": nan_inf_parameters,
    }


def find_image_file(
    folder: str | Path,
    image_name: str,
):
    """在目录中按名字查找常见格式图片。"""

    folder = Path(folder)

    image_extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".JPG",
        ".JPEG",
        ".PNG",
        ".BMP",
        ".TIF",
        ".TIFF",
    ]

    for extension in image_extensions:
        image_path = (
            folder
            / f"{image_name}{extension}"
        )

        if image_path.exists():
            return image_path

    return None


def find_xml_file(
    folder: str | Path,
    image_name: str,
):
    """查找对应 XML 文件。"""

    xml_path = (
        Path(folder)
        / f"{image_name}.xml"
    )

    if xml_path.exists():
        return xml_path

    return None


def crop_pil_to_multiple(
    image: PIL.Image.Image,
    multiple: int = 8,
):
    """
    从右侧和底部裁剪图片，让宽高都是 multiple 的倍数。

    不缩放，避免引入额外模糊。
    """

    width, height = image.size

    new_width = (
        width
        - width % multiple
    )

    new_height = (
        height
        - height % multiple
    )

    if new_width <= 0 or new_height <= 0:
        raise ValueError(
            f"图片尺寸过小：{image.size}，"
            f"无法裁剪为 {multiple} 的倍数。"
        )

    if (
        new_width == width
        and new_height == height
    ):
        return image

    return image.crop(
        (
            0,
            0,
            new_width,
            new_height,
        )
    )


def save_tensor_mask(
    mask_tensor: torch.Tensor,
    save_path: Path,
):
    """将 Tensor mask 保存成单通道图片。"""

    mask_tensor = (
        mask_tensor
        .detach()
        .float()
        .cpu()
    )

    while mask_tensor.ndim > 2:
        mask_tensor = mask_tensor[0]

    if mask_tensor.max().item() <= 1.0:
        mask_tensor = (
            mask_tensor
            * 255.0
        )

    mask_array = (
        mask_tensor
        .clamp(0, 255)
        .byte()
        .numpy()
    )

    PIL.Image.fromarray(
        mask_array,
        mode="L",
    ).save(save_path)


# ============================================================
# 加载 UNet
# ============================================================
def load_custom_unet():
    print("=" * 80)
    print("Loading UNet...")
    print(f"Base model : {BASE_MODEL_PATH}")
    print(f"Checkpoint : {MODEL_PATH}")
    print("=" * 80)

    unet = (
        UNet2DConditionModel
        .from_pretrained(
            str(BASE_MODEL_PATH),
            subfolder="unet",
            torch_dtype=TORCH_DTYPE,
            low_cpu_mem_usage=False,
        )
    )

    # 先替换成训练时的自定义 conv_in
    ContentAwarePositionGatedConvIn.apply_to_unet(
        unet
    )

    state_dict_path = (
        MODEL_PATH
        / "unet"
        / "diffusion_pytorch_model.safetensors"
    )

    if not state_dict_path.exists():
        raise FileNotFoundError(
            f"找不到 UNet 权重："
            f"{state_dict_path}"
        )

    state_dict = load_file(
        str(state_dict_path)
    )

    print(
        "Checkpoint parameter dtypes:",
        Counter(
            value.dtype
            for value in state_dict.values()
        ),
    )

    load_result = unet.load_state_dict(
        state_dict,
        strict=False,
    )

    print("=" * 80)
    print("Missing keys:")
    print("=" * 80)

    if load_result.missing_keys:
        for key in load_result.missing_keys:
            print(f"  {key}")
    else:
        print("  None")

    print("=" * 80)
    print("Unexpected keys:")
    print("=" * 80)

    if load_result.unexpected_keys:
        for key in load_result.unexpected_keys:
            print(f"  {key}")
    else:
        print("  None")

    print("=" * 80)
    print("conv_in type:")
    print(type(unet.conv_in))
    print(unet.conv_in)
    print("=" * 80)

    if not hasattr(
        unet.conv_in,
        "project_mask_crop",
    ):
        raise AttributeError(
            "unet.conv_in 中不存在 "
            "project_mask_crop，请检查 "
            "ContentAwarePositionGatedConvIn。"
        )

    if not hasattr(
        unet.conv_in,
        "project_global_latents",
    ):
        raise AttributeError(
            "unet.conv_in 中不存在 "
            "project_global_latents，请检查 "
            "ContentAwarePositionGatedConvIn。"
        )

    # 检查自定义 conv_in 参数
    conv_report_path = (
        OUTPUT_DIR
        / "conv_in_weight_report.txt"
    )

    check_result = check_conv_in_weights(
        conv_in=unet.conv_in,
        report_path=conv_report_path,
        near_zero_threshold=NEAR_ZERO_THRESHOLD,
    )

    if check_result["all_zero"]:
        print()
        print("WARNING: 以下 conv_in 参数仍然全零：")

        for name in check_result["all_zero"]:
            print(f"  {name}")

        print(
            "如果这些层训练前进行了零初始化，"
            "训练完成后仍然全零，通常说明它们没有更新。"
        )

    if check_result["almost_zero"]:
        print()
        print("WARNING: 以下 conv_in 参数几乎全零：")

        for name in check_result["almost_zero"]:
            print(f"  {name}")

    if CHECK_ALL_UNET_WEIGHTS:
        check_model_weights(unet)

    return unet


# ============================================================
# 加载 Pipeline
# ============================================================
def load_pipeline():
    unet = load_custom_unet()

    print("=" * 80)
    print("Loading Pipeline...")
    print("=" * 80)

    pipe = (
        SlidingWindowInstructPix2PixPipeline
        .from_pretrained(
            str(MODEL_PATH),
            unet=unet,
            torch_dtype=TORCH_DTYPE,
            low_cpu_mem_usage=False,
            device_map=None,
            safety_checker=None,
        )
    )

    pipe = pipe.to(DEVICE)

    pipe.set_progress_bar_config(
        disable=False
    )

    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()

    print("Pipeline Ready")
    print(
        "Pipeline device:",
        pipe._execution_device,
    )
    print(
        "UNet input channels:",
        pipe.unet.config.in_channels,
    )
    print(
        "VAE scale factor:",
        pipe.vae_scale_factor,
    )
    print("=" * 80)

    return pipe


# ============================================================
# 准备单张测试数据
# ============================================================
def prepare_test_inputs(
    image_file: Path,
    xml_file: Path,
    mask_file: Path,
):
    """
    返回：
        original_image
        input_image
        mask_image

    对应训练逻辑：
        original_image:
            原始带裂缝图像

        input_image:
            fill_box_with_mean_color 生成的填充图

        mask_image:
            MASK_DIR 中读取的 mask_pixel_values 对应图片

    注意：
        fill_box_with_mean_color 返回的 inpaint_mask
        在这里不作为模型条件使用。
    """

    original_image = (
        PIL.Image
        .open(image_file)
        .convert("RGB")
    )

    # 第二个返回值 inpaint_mask 只对应训练 loss 加权区域，
    # 不是推理时输入到 mask_projector 的 mask。
    input_image, _ = fill_box_with_mean_color(
        str(xml_file),
        str(image_file),
        large_region_prob=1024,
    )

    input_image = input_image.convert(
        "RGB"
    )

    if mask_file is None:
        raise FileNotFoundError(
            f"没有找到 mask 图片："
            f"{image_file.stem}"
        )

    # 对应训练中的 batch["mask_pixel_values"]
    mask_image = (
        PIL.Image
        .open(mask_file)
        .convert("L")
    )

    if original_image.size != input_image.size:
        raise ValueError(
            "Original/input size mismatch: "
            f"original={original_image.size}, "
            f"input={input_image.size}"
        )

    if original_image.size != mask_image.size:
        raise ValueError(
            "Original/mask size mismatch: "
            f"original={original_image.size}, "
            f"mask={mask_image.size}"
        )

    if input_image.size != mask_image.size:
        raise ValueError(
            "Input/mask size mismatch: "
            f"input={input_image.size}, "
            f"mask={mask_image.size}"
        )

    original_image = crop_pil_to_multiple(
        original_image,
        multiple=8,
    )

    input_image = crop_pil_to_multiple(
        input_image,
        multiple=8,
    )

    mask_image = crop_pil_to_multiple(
        mask_image,
        multiple=8,
    )

    if not (
        original_image.size
        == input_image.size
        == mask_image.size
    ):
        raise ValueError(
            "裁剪后尺寸仍不一致："
            f"original={original_image.size}, "
            f"input={input_image.size}, "
            f"mask={mask_image.size}"
        )

    if (
        input_image.width < CROP_SIZE
        or input_image.height < CROP_SIZE
    ):
        raise ValueError(
            f"输入图像尺寸 {input_image.size} "
            f"小于 crop_size={CROP_SIZE}"
        )

    return (
        original_image,
        input_image,
        mask_image,
    )


# ============================================================
# 单张图片推理
# ============================================================
@torch.inference_mode()
def infer_one_image(
    pipe,
    image_name: str,
    image_file: Path,
    xml_file: Path,
    mask_file: Path,
    output_dir: Path,
    image_index: int = 0,
):
    """
    测试一张图片并保存：

        xxx_original.png
        xxx_input.png
        xxx_mask.png
        xxx_generate.png
    """

    output_original = (
        output_dir
        / f"{image_name}_original.png"
    )

    output_input = (
        output_dir
        / f"{image_name}_input.png"
    )

    output_mask = (
        output_dir
        / f"{image_name}_mask.png"
    )

    output_generate = (
        output_dir
        / f"{image_name}_generate.png"
    )

    if (
        SKIP_EXISTING
        and output_generate.exists()
    ):
        print(
            f"Skip existing: "
            f"{output_generate.name}"
        )
        return "skipped"

    original_image = None
    input_image = None
    mask_image = None
    result = None

    try:
        (
            original_image,
            input_image,
            mask_image,
        ) = prepare_test_inputs(
            image_file=image_file,
            xml_file=xml_file,
            mask_file=mask_file,
        )

        print(
            f"Image size: {input_image.size}"
        )
        print(
            f"Mask mode: {mask_image.mode}"
        )
        print(
            f"Crop size: {CROP_SIZE}"
        )
        print(
            f"Crop overlap: {CROP_OVERLAP}"
        )

        current_seed = (
            SEED
            + image_index
        )

        generator = torch.Generator(
            device=DEVICE
        ).manual_seed(
            current_seed
        )

        pipeline_output = pipe(
            prompt=PROMPT,
            image=input_image,
            mask=mask_image,
            crop_size=CROP_SIZE,
            crop_overlap=CROP_OVERLAP,
            guidance_scale=GUIDANCE_SCALE,
            image_guidance_scale=IMAGE_GUIDANCE_SCALE,
            num_inference_steps=NUM_INFERENCE_STEPS,
            crop_process_batch_size=CROP_PROCESS_BATCH_SIZE,
            generator=generator,
            output_type="pil",
            return_dict=True,
        )

        if (
            pipeline_output.images is None
            or len(pipeline_output.images) == 0
        ):
            raise RuntimeError(
                "Pipeline 没有返回生成图片。"
            )

        result = pipeline_output.images[0]

        original_image.save(
            output_original
        )

        input_image.save(
            output_input
        )

        if isinstance(
            mask_image,
            PIL.Image.Image,
        ):
            mask_image.save(
                output_mask
            )
        else:
            save_tensor_mask(
                mask_tensor=mask_image,
                save_path=output_mask,
            )

        result.save(
            output_generate
        )

        print(
            f"Saved original : "
            f"{output_original}"
        )
        print(
            f"Saved input    : "
            f"{output_input}"
        )
        print(
            f"Saved mask     : "
            f"{output_mask}"
        )
        print(
            f"Saved generated: "
            f"{output_generate}"
        )

        return "success"

    finally:
        del original_image
        del input_image
        del mask_image
        del result

        clear_cuda_cache()


# ============================================================
# 批量测试
# ============================================================
def run_batch_test():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    failed_log_path = (
        OUTPUT_DIR
        / "failed_images.txt"
    )

    if not VAL_TXT_PATH.exists():
        raise FileNotFoundError(
            f"找不到 val.txt："
            f"{VAL_TXT_PATH}"
        )

    with VAL_TXT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        val_names = [
            line.strip()
            for line in file
            if line.strip()
        ]

    if MAX_TEST_IMAGES is not None:
        val_names = val_names[
            :MAX_TEST_IMAGES
        ]

    print("=" * 80)
    print(
        f"Val txt          : "
        f"{VAL_TXT_PATH}"
    )
    print(
        f"Test image count : "
        f"{len(val_names)}"
    )
    print(
        f"Output directory : "
        f"{OUTPUT_DIR}"
    )
    print("=" * 80)

    pipe = load_pipeline()

    success = 0
    failed = 0
    skipped = 0

    image_not_found = 0
    xml_not_found = 0
    mask_not_found = 0

    failed_files = []

    total = len(val_names)

    for index, image_name in enumerate(
        val_names,
        start=1,
    ):
        print()
        print("=" * 80)
        print(
            f"[{index}/{total}] "
            f"Processing: {image_name}"
        )
        print("=" * 80)

        image_file = find_image_file(
            ORIGINAL_IMAGES_DIR,
            image_name,
        )

        xml_file = find_xml_file(
            XML_DIR,
            image_name,
        )

        mask_file = find_image_file(
            MASK_DIR,
            image_name,
        )

        if image_file is None:
            image_not_found += 1

            error_message = (
                "Image file not found"
            )

            failed_files.append(
                (
                    image_name,
                    error_message,
                )
            )

            print(
                f"[{index}/{total}] "
                f"{error_message}: "
                f"{image_name}"
            )
            continue

        if xml_file is None:
            xml_not_found += 1

            error_message = (
                "XML file not found"
            )

            failed_files.append(
                (
                    image_name,
                    error_message,
                )
            )

            print(
                f"[{index}/{total}] "
                f"{error_message}: "
                f"{image_name}"
            )
            continue

        # mask 始终是必需的
        if mask_file is None:
            mask_not_found += 1

            error_message = (
                "Mask file not found"
            )

            failed_files.append(
                (
                    image_name,
                    error_message,
                )
            )

            print(
                f"[{index}/{total}] "
                f"{error_message}: "
                f"{image_name}"
            )
            continue

        try:
            status = infer_one_image(
                pipe=pipe,
                image_name=image_name,
                image_file=image_file,
                xml_file=xml_file,
                mask_file=mask_file,
                output_dir=OUTPUT_DIR,
                image_index=index - 1,
            )

            if status == "success":
                success += 1

                print(
                    f"[{index}/{total}] "
                    f"Success: {image_name}"
                )

            elif status == "skipped":
                skipped += 1

                print(
                    f"[{index}/{total}] "
                    f"Skipped: {image_name}"
                )

        except Exception as exception:
            failed += 1

            error_text = (
                f"{type(exception).__name__}: "
                f"{exception}\n\n"
                f"{traceback.format_exc()}"
            )

            failed_files.append(
                (
                    str(image_file),
                    error_text,
                )
            )

            print(
                f"[{index}/{total}] "
                f"Failed: {image_file.name}"
            )

            print(error_text)

            clear_cuda_cache()

    if failed_files:
        with failed_log_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            for (
                file_path,
                error_message,
            ) in failed_files:
                file.write(
                    f"File: {file_path}\n"
                )
                file.write(
                    f"{error_message}\n"
                )
                file.write(
                    "-" * 100
                    + "\n"
                )

        print(
            f"Failed log saved to: "
            f"{failed_log_path}"
        )

    print()
    print("=" * 80)
    print(
        "Validation batch inference finished"
    )
    print(
        f"Total tested     : {total}"
    )
    print(
        f"Success          : {success}"
    )
    print(
        f"Skipped          : {skipped}"
    )
    print(
        f"Failed inference : {failed}"
    )
    print(
        f"Image not found  : "
        f"{image_not_found}"
    )
    print(
        f"XML not found    : "
        f"{xml_not_found}"
    )
    print(
        f"Mask not found   : "
        f"{mask_not_found}"
    )
    print(
        f"Output directory : "
        f"{OUTPUT_DIR}"
    )
    print("=" * 80)


# ============================================================
# 程序入口
# ============================================================
if __name__ == "__main__":
    run_batch_test()