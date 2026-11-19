import argparse
import os
from pathlib import Path

# V100 (sm_70) can fail to select a cuDNN v8 frontend execution plan with some
# PyTorch/cuDNN combinations. This variable must be set before importing torch.
os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image

from deepcrack.my_test import load_model


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use one worker per GPU to generate DeepCrack masks."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("/root/autodl-tmp/instruct-Pix2Pix/datasets/UAV-PDD/JPEGImages"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/autodl-tmp/instruct-Pix2Pix/datasets/UAV-PDD/masks"),
    )
    parser.add_argument(
        "--weight-path",
        type=Path,
        default=Path("deepcrack/checkpoints/DeepCrack_CT260_FT1.pth"),
    )
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated physical GPU IDs, e.g. 0,1,2,3. Default: all visible GPUs.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def predict(model, image_tensor: torch.Tensor) -> np.ndarray:
    with torch.inference_mode():
        output = model(image_tensor)
        # DeepCrack commonly returns a sequence of side outputs. Keep compatibility
        # with models that return a tensor directly as well.
        logits = output[0] if isinstance(output, (tuple, list)) else output
        return torch.sigmoid(logits).squeeze().float().cpu().numpy()


def save_mask(output_path: Path, mask) -> None:
    """Normalize model output and save it as an 8-bit grayscale PNG."""
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    elif not isinstance(mask, np.ndarray):
        mask = np.asarray(mask)

    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(
            f"Mask must be two-dimensional before saving, got {mask.shape}"
        )

    if mask.dtype == np.bool_:
        mask = mask.astype(np.uint8) * 255
    elif np.issubdtype(mask.dtype, np.floating):
        # Support either probabilities in [0, 1] or mask values in [0, 255].
        if mask.size and float(mask.max()) <= 1.0:
            mask = mask * 255.0
        mask = np.clip(mask, 0, 255).astype(np.uint8)
    elif mask.dtype != np.uint8:
        mask = np.clip(mask, 0, 255).astype(np.uint8)

    mask = np.ascontiguousarray(mask, dtype=np.uint8)
    # Some OpenCV 4.8 builds reject NumPy arrays after the environment's NumPy
    # package has been replaced. Pillow avoids that OpenCV/NumPy ABI boundary.
    Image.fromarray(mask, mode="L").save(output_path, format="PNG")


def generate_mask_for_full_image(
    image: np.ndarray,
    window_size: int,
    model,
    device: torch.device,
    threshold: float = 0.5,
) -> np.ndarray:
    if window_size <= 0:
        raise ValueError("window_size must be greater than zero")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    height, width = image.shape[:2]
    padded_height = ((height + window_size - 1) // window_size) * window_size
    padded_width = ((width + window_size - 1) // window_size) * window_size
    padded_image = cv2.copyMakeBorder(
        image,
        0,
        padded_height - height,
        0,
        padded_width - width,
        cv2.BORDER_REFLECT,
    )

    mask = np.zeros((padded_height, padded_width), dtype=np.uint8)
    for y in range(0, padded_height, window_size):
        for x in range(0, padded_width, window_size):
            roi = padded_image[y : y + window_size, x : x + window_size]
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(
                np.ascontiguousarray(rgb.transpose(2, 0, 1))
            ).unsqueeze(0)
            tensor = tensor.to(device=device, dtype=torch.float32).div_(255.0)

            probability = predict(model, tensor)
            if probability.ndim != 2:
                raise RuntimeError(
                    f"Expected a 2D prediction, got shape {probability.shape}"
                )
            mask[y : y + window_size, x : x + window_size] = (
                probability > threshold
            ).astype(np.uint8) * 255

    return mask[:height, :width]


def load_model_on_device(weight_path: Path, device: torch.device):
    # Some DeepCrack load_model implementations use torch.load without map_location,
    # so select the worker GPU before loading the checkpoint.
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    model = load_model(str(weight_path))
    model = model.to(device)
    model.eval()
    return model


def worker(
    rank: int,
    gpu_ids: list[int],
    image_files: list[Path],
    output_dir: Path,
    weight_path: Path,
    window_size: int,
    threshold: float,
    skip_existing: bool,
):
    physical_gpu_id = gpu_ids[rank]
    device = torch.device(f"cuda:{physical_gpu_id}")
    model = load_model_on_device(weight_path, device)
    assigned_files = image_files[rank::len(gpu_ids)]

    print(
        f"[GPU {physical_gpu_id}] loaded model; assigned {len(assigned_files)} images; "
        f"cuDNN={torch.backends.cudnn.version()}, "
        f"cuDNN enabled={torch.backends.cudnn.enabled}",
        flush=True,
    )
    for index, image_path in enumerate(assigned_files, start=1):
        output_path = output_dir / f"{image_path.stem}.png"
        if skip_existing and output_path.exists():
            print(f"[GPU {physical_gpu_id}] skip existing: {output_path.name}", flush=True)
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[GPU {physical_gpu_id}] failed to load: {image_path}", flush=True)
            continue

        try:
            mask = generate_mask_for_full_image(
                image=image,
                window_size=window_size,
                model=model,
                device=device,
                threshold=threshold,
            )
            save_mask(output_path, mask)
            print(
                f"[GPU {physical_gpu_id}] {index}/{len(assigned_files)} saved: "
                f"{output_path.name}",
                flush=True,
            )
        except RuntimeError as error:
            # On some V100 environments even the legacy cuDNN API cannot find an
            # engine for a DeepCrack convolution. Fall back to native CUDA
            # convolution once, then keep cuDNN disabled for this worker.
            error_message = str(error).lower()
            is_cudnn_algorithm_error = any(
                message in error_message
                for message in (
                    "unable to find an engine to execute this computation",
                    "unable to find a valid cudnn algorithm to run convolution",
                )
            )
            if not is_cudnn_algorithm_error or not torch.backends.cudnn.enabled:
                print(
                    f"[GPU {physical_gpu_id}] failed {image_path.name}: {error}",
                    flush=True,
                )
                continue

            print(
                f"[GPU {physical_gpu_id}] cuDNN engine unavailable; "
                f"disabling cuDNN and retrying {image_path.name}",
                flush=True,
            )
            torch.backends.cudnn.enabled = False
            torch.cuda.empty_cache()

            try:
                mask = generate_mask_for_full_image(
                    image=image,
                    window_size=window_size,
                    model=model,
                    device=device,
                    threshold=threshold,
                )
                save_mask(output_path, mask)
                print(
                    f"[GPU {physical_gpu_id}] {index}/{len(assigned_files)} saved after "
                    f"cuDNN fallback: {output_path.name}",
                    flush=True,
                )
            except Exception as retry_error:
                print(
                    f"[GPU {physical_gpu_id}] failed after cuDNN fallback "
                    f"{image_path.name}: {retry_error}",
                    flush=True,
                )
        except Exception as error:
            print(f"[GPU {physical_gpu_id}] failed {image_path.name}: {error}", flush=True)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this script requires at least one GPU")

    visible_gpu_count = torch.cuda.device_count()
    if args.gpus is None:
        gpu_ids = list(range(visible_gpu_count))
    else:
        gpu_ids = [int(item.strip()) for item in args.gpus.split(",") if item.strip()]

    if not gpu_ids:
        raise ValueError("No GPU was selected")
    invalid_gpu_ids = [gpu_id for gpu_id in gpu_ids if not 0 <= gpu_id < visible_gpu_count]
    if invalid_gpu_ids:
        raise ValueError(
            f"Invalid GPU IDs {invalid_gpu_ids}; visible CUDA IDs are "
            f"0..{visible_gpu_count - 1}"
        )
    if not args.image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {args.image_dir}")
    if not args.weight_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.weight_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_files = sorted(
        path
        for path in args.image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        print(f"No supported images found in {args.image_dir}")
        return

    worker_count = min(len(gpu_ids), len(image_files))
    gpu_ids = gpu_ids[:worker_count]
    print(
        f"Processing {len(image_files)} images with {worker_count} GPUs: {gpu_ids}",
        flush=True,
    )
    mp.spawn(
        worker,
        args=(
            gpu_ids,
            image_files,
            args.output_dir,
            args.weight_path,
            args.window_size,
            args.threshold,
            args.skip_existing,
        ),
        nprocs=worker_count,
        join=True,
    )
    print(f"All done. Results in: {args.output_dir}")


if __name__ == "__main__":
    # spawn is safer than fork for independent CUDA contexts.
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
