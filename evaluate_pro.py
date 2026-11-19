from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


# ============================================================
# Constants
# ============================================================

GENERATED_SUFFIX = "_generate.png"

REGIONS = (
    "full",
    "roi",
    "context",
    "boundary",
)

BASIC_METRICS = (
    "mae",
    "mse",
    "rmse",
    "nrmse",
    "psnr",
    "ssim",
    "grad_mae",
)

METRICS = tuple(
    f"{region}_{metric}"
    for region in REGIONS
    for metric in BASIC_METRICS
) + (
    "full_lpips",
    "roi_lpips",
    "context_lpips",
    "boundary_lpips",
)

PER_IMAGE_COLUMNS = (
    "method",
    "sample_id",
    "generated_file",
    "mask_source",
    "image_height",
    "image_width",
    "roi_pixels",
    "roi_fraction",
    "boundary_pixels",
    "boundary_fraction",
    *METRICS,
)

TEXT_COLUMNS = {
    "method",
    "sample_id",
    "generated_file",
    "mask_source",
}

INTEGER_COLUMNS = {
    "image_height",
    "image_width",
    "roi_pixels",
    "boundary_pixels",
}


# ============================================================
# Arguments
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare paired *_generate.png inpainting outputs."
    )

    parser.add_argument(
        "--result-dirs",
        type=Path,
        nargs="+",
        help=(
            "Result directories. "
            "Default: all --glob directories under --root."
        ),
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
    )

    parser.add_argument(
        "--glob",
        default="results*",
        help=(
            "Directories to evaluate under --root "
            "(default: results*)."
        ),
    )

    parser.add_argument(
        "--reference-dir",
        type=Path,
        help=(
            "Shared directory for original/input/inpaint-mask files "
            "when absent in a method directory."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_pro"),
        help=(
            "Report root. Each result directory gets its own "
            "subdirectory here (default: evaluation_pro)."
        ),
    )

    parser.add_argument(
        "--all-available",
        action="store_true",
        help=(
            "Do not restrict all methods to their common "
            "generated-image IDs."
        ),
    )

    parser.add_argument(
        "--max-images",
        type=int,
        help="Limit images per method for a quick test.",
    )

    parser.add_argument(
        "--changed-tolerance",
        type=int,
        default=0,
        help=(
            "RGB threshold used when reconstructing an absent "
            "inpaint mask (default: 0)."
        ),
    )

    parser.add_argument(
        "--mask-threshold",
        type=int,
        default=127,
        help=(
            "Values above this in *_inpaint_mask.png "
            "are part of the repair region."
        ),
    )

    parser.add_argument(
        "--boundary-width",
        type=int,
        default=8,
        help=(
            "Width in pixels of the repair-boundary band "
            "used for boundary metrics (default: 8)."
        ),
    )

    parser.add_argument(
        "--psnr-cap",
        type=float,
        default=100.0,
        help=(
            "Finite PSNR reported for a perfect reconstruction "
            "(default: 100)."
        ),
    )

    parser.add_argument(
        "--lpips",
        action="store_true",
        help=(
            "Calculate optional LPIPS perceptual metrics "
            "(requires package lpips)."
        ),
    )

    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "LPIPS device: auto, cpu, cuda, or cuda:N."
        ),
    )

    parser.add_argument(
        "--lpips-size",
        type=int,
        default=512,
        help=(
            "Resize LPIPS input so its longest side is this many "
            "pixels; 0 keeps source size."
        ),
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Stop on a bad sample instead of recording it "
            "in failures.csv."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "CPU processes used for image metrics. 0 chooses "
            "up to 8 automatically; 1 disables parallelism. "
            "LPIPS and --strict run serially."
        ),
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "Ignore existing per-image reports and recompute all "
            "selected samples."
        ),
    )

    args = parser.parse_args()

    if (
        args.changed_tolerance < 0
        or not 0 <= args.mask_threshold <= 255
        or args.lpips_size < 0
        or args.boundary_width < 0
        or args.psnr_cap <= 0
        or args.workers < 0
        or (
            args.max_images is not None
            and args.max_images < 1
        )
    ):
        parser.error(
            "changed tolerance/lpips size/boundary width/workers "
            "must be non-negative, PSNR cap must be positive, and "
            "mask threshold must be in [0, 255]"
        )

    return args


# ============================================================
# Image loading
# ============================================================

def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(
            image.convert("RGB"),
            dtype=np.uint8,
        )


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(
            image.convert("L"),
            dtype=np.uint8,
        )


# ============================================================
# Result directory discovery
# ============================================================

def discover_dirs(
    args: argparse.Namespace,
) -> list[Path]:

    if args.result_dirs:
        directories = [
            path.resolve()
            for path in args.result_dirs
        ]
    else:
        directories = sorted(
            path.resolve()
            for path in args.root.resolve().glob(args.glob)
            if path.is_dir()
        )

    valid = []

    for directory in directories:

        if not directory.is_dir():
            raise FileNotFoundError(
                f"Result directory does not exist: "
                f"{directory}"
            )

        if any(
            directory.glob(
                f"*{GENERATED_SUFFIX}"
            )
        ):
            valid.append(directory)

        else:
            print(
                f"Warning: {directory} has no "
                f"*{GENERATED_SUFFIX}; skipped.",
                file=sys.stderr,
            )

    if not valid:
        raise FileNotFoundError(
            "No generated result images found. "
            "Check --result-dirs or --root/--glob."
        )

    if len(
        {path.name for path in valid}
    ) != len(valid):
        raise ValueError(
            "Result directory names must be unique "
            "because they label the report."
        )

    return valid


def generated_by_id(
    directory: Path,
) -> dict[str, Path]:

    paths: dict[str, Path] = {}

    for path in sorted(
        directory.glob(
            f"*{GENERATED_SUFFIX}"
        )
    ):

        sample_id = path.name[
            :-len(GENERATED_SUFFIX)
        ]

        if sample_id in paths:
            raise ValueError(
                f"Duplicate generated sample "
                f"{sample_id!r} in {directory}"
            )

        paths[sample_id] = path

    return paths


# ============================================================
# Reference files
# ============================================================

def reference_file(
    filename: str,
    method_dir: Path,
    reference_dir: Path | None,
) -> Path:

    """
    Method-local metadata wins.

    reference_dir is used only when the file does
    not exist inside method_dir.
    """

    path = method_dir / filename

    if path.is_file():
        return path

    if reference_dir is not None:
        reference_path = (
            reference_dir / filename
        )

        if reference_path.is_file():
            return reference_path

    if reference_dir is None:
        checked = str(path)
    else:
        checked = (
            f"{path} or "
            f"{reference_dir / filename}"
        )

    raise FileNotFoundError(
        f"Missing reference file: {checked}"
    )


# ============================================================
# Repair mask
# ============================================================

def repair_mask(
    sample_id: str,
    method_dir: Path,
    reference_dir: Path | None,
    original: np.ndarray,
    input_image: np.ndarray,
    threshold: int,
    changed_tolerance: int,
) -> tuple[np.ndarray, str]:

    """
    Find actual inpainting / repair pixels.

    Prefer *_inpaint_mask.png.

    If absent, reconstruct the repair mask from
    original-input difference.
    """

    try:

        mask_path = reference_file(
            f"{sample_id}_inpaint_mask.png",
            method_dir,
            reference_dir,
        )

        saved_mask = load_gray(mask_path)

        if (
            saved_mask.shape
            != original.shape[:2]
        ):
            raise ValueError(
                "inpaint-mask size "
                f"{saved_mask.shape} differs from "
                f"image size {original.shape[:2]}"
            )

        return (
            saved_mask > threshold,
            "saved_inpaint_mask",
        )

    except FileNotFoundError:

        delta = np.abs(
            original.astype(np.int16)
            - input_image.astype(np.int16)
        )

        mask = np.any(
            delta > changed_tolerance,
            axis=2,
        )

        return (
            mask,
            "derived_from_original_input",
        )


# ============================================================
# Boundary mask
# ============================================================

def make_boundary_mask(
    mask: np.ndarray,
    width: int,
) -> np.ndarray:

    """
    Construct a band around the inpainting-mask boundary.

    It contains pixels both inside and outside ROI.

    Conceptually:

        Boundary =
            Dilate(mask)
            -
            Erode(mask)

    This measures transition/seam quality.
    """

    if (
        width <= 0
        or not np.any(mask)
    ):
        return np.zeros_like(
            mask,
            dtype=bool,
        )

    image = Image.fromarray(
        mask.astype(np.uint8) * 255
    )

    kernel_size = (
        2 * width + 1
    )

    dilated = np.asarray(
        image.filter(
            ImageFilter.MaxFilter(
                kernel_size
            )
        ),
        dtype=np.uint8,
    ) > 0

    eroded = np.asarray(
        image.filter(
            ImageFilter.MinFilter(
                kernel_size
            )
        ),
        dtype=np.uint8,
    ) > 0

    boundary = (
        dilated
        & ~eroded
    )

    return boundary


# ============================================================
# SSIM
# ============================================================

def compute_ssim_map(
    original: np.ndarray,
    generated: np.ndarray,
) -> np.ndarray:

    """
    RGB SSIM map using:

        11 x 11 Gaussian window
        sigma = 1.5

    Standard SSIM constants for 8-bit RGB.
    """

    try:
        import cv2
    except ImportError:
        cv2 = None

    if min(
        original.shape[:2]
    ) < 11:
        raise ValueError(
            "Images smaller than 11 pixels "
            "cannot be scored with SSIM"
        )

    x = original.astype(
        np.float64
    )

    y = generated.astype(
        np.float64
    )

    # --------------------------------------------------------
    # Gaussian blur
    # --------------------------------------------------------

    if cv2 is not None:

        def gaussian_blur(
            image: np.ndarray,
        ) -> np.ndarray:

            return cv2.GaussianBlur(
                image,
                (11, 11),
                1.5,
                borderType=cv2.BORDER_REFLECT,
            )

    else:

        coordinates = np.arange(
            -5,
            6,
        )

        weights = np.exp(
            -0.5
            * (
                coordinates / 1.5
            ) ** 2
        )

        weights /= weights.sum()

        def gaussian_blur(
            image: np.ndarray,
        ) -> np.ndarray:

            padded = np.pad(
                image,
                (
                    (0, 0),
                    (5, 5),
                    (0, 0),
                ),
                mode="reflect",
            )

            horizontal = sum(
                weight
                * padded[
                    :,
                    offset:
                    offset
                    + image.shape[1],
                    :,
                ]
                for offset, weight
                in enumerate(weights)
            )

            padded = np.pad(
                horizontal,
                (
                    (5, 5),
                    (0, 0),
                    (0, 0),
                ),
                mode="reflect",
            )

            vertical = sum(
                weight
                * padded[
                    offset:
                    offset
                    + image.shape[0],
                    :,
                    :,
                ]
                for offset, weight
                in enumerate(weights)
            )

            return vertical

    # --------------------------------------------------------
    # SSIM statistics
    # --------------------------------------------------------

    mu_x = gaussian_blur(x)

    mu_y = gaussian_blur(y)

    sigma_x = (
        gaussian_blur(x * x)
        - mu_x * mu_x
    )

    sigma_y = (
        gaussian_blur(y * y)
        - mu_y * mu_y
    )

    sigma_xy = (
        gaussian_blur(x * y)
        - mu_x * mu_y
    )

    c1 = (
        0.01 * 255
    ) ** 2

    c2 = (
        0.03 * 255
    ) ** 2

    numerator = (
        (
            2 * mu_x * mu_y
            + c1
        )
        *
        (
            2 * sigma_xy
            + c2
        )
    )

    denominator = (
        (
            mu_x * mu_x
            + mu_y * mu_y
            + c1
        )
        *
        (
            sigma_x
            + sigma_y
            + c2
        )
    )

    values = (
        numerator
        / denominator
    )

    # RGB -> scalar SSIM map
    return values.mean(
        axis=2
    )


# ============================================================
# Gradient / edge metrics
# ============================================================

def rgb_to_gray(
    image: np.ndarray,
) -> np.ndarray:

    """
    RGB -> luminance.
    """

    image = image.astype(
        np.float64
    )

    return (
        0.299
        * image[:, :, 0]
        +
        0.587
        * image[:, :, 1]
        +
        0.114
        * image[:, :, 2]
    )


def gradient_magnitude(
    image: np.ndarray,
) -> np.ndarray:

    """
    Compute gradient magnitude.

    Uses Sobel if OpenCV exists.
    Otherwise uses numpy.gradient.
    """

    gray = rgb_to_gray(
        image
    )

    try:
        import cv2

        gx = cv2.Sobel(
            gray,
            cv2.CV_64F,
            1,
            0,
            ksize=3,
            borderType=cv2.BORDER_REFLECT,
        )

        gy = cv2.Sobel(
            gray,
            cv2.CV_64F,
            0,
            1,
            ksize=3,
            borderType=cv2.BORDER_REFLECT,
        )

    except ImportError:

        gy, gx = np.gradient(
            gray
        )

    magnitude = np.sqrt(
        gx * gx
        +
        gy * gy
    )

    return magnitude


def gradient_error_map(
    original: np.ndarray,
    generated: np.ndarray,
) -> np.ndarray:

    """
    Per-pixel difference between gradient magnitudes.

    Lower = generated edges / texture are closer to GT.
    """

    original_gradient = (
        gradient_magnitude(
            original
        )
    )

    generated_gradient = (
        gradient_magnitude(
            generated
        )
    )

    return np.abs(
        original_gradient
        -
        generated_gradient
    )


# ============================================================
# Generic metric helpers
# ============================================================

def region_mean(
    values: np.ndarray,
    region: np.ndarray | None,
) -> float:

    if region is None:
        return float(
            values.mean()
        )

    if not np.any(region):
        return math.nan

    return float(
        values[region].mean()
    )


def psnr(
    mse: float,
    cap: float,
) -> float:

    if math.isnan(mse):
        return math.nan

    if mse == 0:
        return cap

    value = (
        20
        * math.log10(
            255
            /
            math.sqrt(mse)
        )
    )

    return min(
        value,
        cap,
    )


# ============================================================
# MAE / MSE / RMSE / NRMSE / PSNR / SSIM / Gradient
# ============================================================

def calculate_metrics(
    original: np.ndarray,
    generated: np.ndarray,
    mask: np.ndarray,
    boundary_mask: np.ndarray,
    psnr_cap: float,
) -> dict[str, float]:

    difference = (
        original.astype(np.float64)
        -
        generated.astype(np.float64)
    )

    # --------------------------------------------------------
    # Pixel error maps
    # --------------------------------------------------------

    per_pixel_abs = (
        np.abs(difference)
        .mean(axis=2)
    )

    per_pixel_sq = (
        difference
        * difference
    ).mean(axis=2)

    # --------------------------------------------------------
    # Structural similarity
    # --------------------------------------------------------

    ssim_map = compute_ssim_map(
        original,
        generated,
    )

    # --------------------------------------------------------
    # Gradient / edge similarity
    # --------------------------------------------------------

    grad_error_map = (
        gradient_error_map(
            original,
            generated,
        )
    )

    # --------------------------------------------------------
    # Regions
    # --------------------------------------------------------

    regions = {
        "full": None,
        "roi": mask,
        "context": ~mask,
        "boundary": boundary_mask,
    }

    metrics: dict[str, float] = {}

    for name, region in regions.items():

        mae = region_mean(
            per_pixel_abs,
            region,
        )

        mse = region_mean(
            per_pixel_sq,
            region,
        )

        # ----------------------------------------------------
        # RMSE
        # ----------------------------------------------------

        if math.isnan(mse):
            rmse = math.nan
        else:
            rmse = math.sqrt(
                mse
            )

        # ----------------------------------------------------
        # NRMSE
        #
        # Normalize by full 8-bit range.
        #
        # 0 = perfect
        # 1 = RMSE == 255
        # ----------------------------------------------------

        if math.isnan(rmse):
            nrmse = math.nan
        else:
            nrmse = (
                rmse / 255.0
            )

        metrics[
            f"{name}_mae"
        ] = mae

        metrics[
            f"{name}_mse"
        ] = mse

        metrics[
            f"{name}_rmse"
        ] = rmse

        metrics[
            f"{name}_nrmse"
        ] = nrmse

        metrics[
            f"{name}_psnr"
        ] = psnr(
            mse,
            psnr_cap,
        )

        metrics[
            f"{name}_ssim"
        ] = region_mean(
            ssim_map,
            region,
        )

        metrics[
            f"{name}_grad_mae"
        ] = region_mean(
            grad_error_map,
            region,
        )

    return metrics


# ============================================================
# LPIPS
# ============================================================

class LPIPS:

    """
    Optional LPIPS backend.

    Network:
        AlexNet

    Input:
        RGB [-1, 1]

    Lower = better.
    """

    def __init__(
        self,
        device_name: str,
        longest_edge: int,
    ) -> None:

        try:
            import lpips
            import torch

        except ImportError as error:

            raise RuntimeError(
                "--lpips requires the optional "
                "lpips package.\n"
                "Install with:\n"
                "    pip install lpips"
            ) from error

        if device_name == "auto":

            device_name = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.torch = torch

        self.device = torch.device(
            device_name
        )

        self.longest_edge = (
            longest_edge
        )

        self.model = (
            lpips.LPIPS(
                net="alex"
            )
            .to(self.device)
            .eval()
        )

        print(
            "LPIPS enabled on "
            f"{self.device} "
            "(AlexNet).",
            file=sys.stderr,
        )

    def tensor(
        self,
        image: np.ndarray,
    ) -> Any:

        pil = Image.fromarray(
            image
        )

        # ----------------------------------------------------
        # Optional downsampling
        # ----------------------------------------------------

        if (
            self.longest_edge
            and max(pil.size)
            > self.longest_edge
        ):

            scale = (
                self.longest_edge
                /
                max(pil.size)
            )

            new_width = max(
                1,
                round(
                    pil.width
                    * scale
                ),
            )

            new_height = max(
                1,
                round(
                    pil.height
                    * scale
                ),
            )

            pil = pil.resize(
                (
                    new_width,
                    new_height,
                ),
                Image.Resampling.BICUBIC,
            )

        # ----------------------------------------------------
        # [0,255] -> [-1,1]
        # ----------------------------------------------------

        array = (
            np.asarray(
                pil,
                dtype=np.float32,
            )
            / 127.5
            - 1.0
        )

        tensor = (
            self.torch
            .from_numpy(
                np.ascontiguousarray(
                    array.transpose(
                        2,
                        0,
                        1,
                    )
                )
            )
            .unsqueeze(0)
            .to(self.device)
        )

        return tensor

    def __call__(
        self,
        original: np.ndarray,
        generated: np.ndarray,
    ) -> float:

        with self.torch.inference_mode():

            value = self.model(
                self.tensor(original),
                self.tensor(generated),
            )

        return float(
            value.item()
        )


# ============================================================
# Region-specific LPIPS
# ============================================================

def add_lpips(
    metrics: dict[str, float],
    original: np.ndarray,
    generated: np.ndarray,
    mask: np.ndarray,
    boundary_mask: np.ndarray,
    evaluator: LPIPS | None,
) -> None:

    if evaluator is None:

        metrics[
            "full_lpips"
        ] = math.nan

        metrics[
            "roi_lpips"
        ] = math.nan

        metrics[
            "context_lpips"
        ] = math.nan

        metrics[
            "boundary_lpips"
        ] = math.nan

        return

    # --------------------------------------------------------
    # Full LPIPS
    # --------------------------------------------------------

    metrics[
        "full_lpips"
    ] = evaluator(
        original,
        generated,
    )

    # --------------------------------------------------------
    # ROI LPIPS
    #
    # Original image remains outside ROI.
    # Generated pixels only replace ROI.
    # --------------------------------------------------------

    if np.any(mask):

        roi_composite = (
            original.copy()
        )

        roi_composite[
            mask
        ] = generated[
            mask
        ]

        metrics[
            "roi_lpips"
        ] = evaluator(
            original,
            roi_composite,
        )

    else:

        metrics[
            "roi_lpips"
        ] = math.nan

    # --------------------------------------------------------
    # Context LPIPS
    #
    # Generated pixels only replace pixels outside ROI.
    # --------------------------------------------------------

    context_mask = ~mask

    if np.any(context_mask):

        context_composite = (
            original.copy()
        )

        context_composite[
            context_mask
        ] = generated[
            context_mask
        ]

        metrics[
            "context_lpips"
        ] = evaluator(
            original,
            context_composite,
        )

    else:

        metrics[
            "context_lpips"
        ] = math.nan

    # --------------------------------------------------------
    # Boundary LPIPS
    # --------------------------------------------------------

    if np.any(boundary_mask):

        boundary_composite = (
            original.copy()
        )

        boundary_composite[
            boundary_mask
        ] = generated[
            boundary_mask
        ]

        metrics[
            "boundary_lpips"
        ] = evaluator(
            original,
            boundary_composite,
        )

    else:

        metrics[
            "boundary_lpips"
        ] = math.nan


# ============================================================
# CSV helpers
# ============================================================

def number_or_none(
    value: Any,
) -> Any:

    if isinstance(
        value,
        (
            float,
            np.floating,
        ),
    ):

        if not math.isfinite(
            float(value)
        ):
            return None

    return value


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    columns: tuple[str, ...],
) -> None:

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )

        writer.writeheader()

        for row in rows:

            output_row = {}

            for key in columns:

                value = row.get(
                    key,
                    "",
                )

                if (
                    number_or_none(
                        value
                    )
                    is None
                ):
                    value = ""

                output_row[
                    key
                ] = value

            writer.writerow(
                output_row
            )


# ============================================================
# Evaluate one directory
# ============================================================

def normalized_path(
    value: str | Path | None,
) -> str | None:

    if value is None:
        return None

    return str(
        Path(value).resolve()
    )


def cache_settings_match(
    summary_path: Path,
    directory: Path,
    reference_dir: Path | None,
    args: argparse.Namespace,
) -> bool:

    """
    Check whether an existing report was produced with compatible
    metric settings. Older reports without a setting are accepted:
    their per-image rows are still useful and --no-cache can force
    a complete refresh when needed.
    """

    if not summary_path.is_file():
        return True

    try:

        with summary_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            report = json.load(file)

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return False

    settings = report.get(
        "settings",
        {},
    )

    if not isinstance(settings, dict):
        return False

    if (
        "result_dir" in settings
        and normalized_path(
            settings["result_dir"]
        )
        != normalized_path(directory)
    ):
        return False

    if (
        "reference_dir" in settings
        and normalized_path(
            settings["reference_dir"]
        )
        != normalized_path(reference_dir)
    ):
        return False

    expected = {
        "lpips_enabled": args.lpips,
        "boundary_width": args.boundary_width,
        "mask_threshold": args.mask_threshold,
        "changed_tolerance": args.changed_tolerance,
        "psnr_cap": args.psnr_cap,
    }

    if args.lpips:
        expected["lpips_network"] = "alex"
        expected[
            "lpips_resize_longest_edge"
        ] = args.lpips_size

    return all(
        key not in settings
        or settings[key] == value
        for key, value in expected.items()
    )


def deserialize_cached_row(
    row: dict[str, str],
) -> dict[str, Any]:

    output: dict[str, Any] = {}

    for column in PER_IMAGE_COLUMNS:

        value = row[column]

        if column in TEXT_COLUMNS:
            output[column] = value

        elif column in INTEGER_COLUMNS:
            output[column] = int(value)

        else:
            output[column] = (
                float(value)
                if value != ""
                else math.nan
            )

    return output


def load_cached_rows(
    cache_dir: Path,
    directory: Path,
    outputs: dict[str, Path],
    reference_dir: Path | None,
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:

    """
    Load completed samples from a previous report.

    A cache row is reusable only when it has all current columns,
    belongs to this method directory, and points to the same generated
    file. This avoids recalculating already evaluated result folders.
    """

    if args.no_cache:
        return {}

    csv_path = cache_dir / "metrics_cache.csv"

    # Reports produced before caching stored their rows only in the
    # user-facing per-image CSV. Seed the persistent cache from it.
    if not csv_path.is_file():
        csv_path = cache_dir / "per_image_metrics.csv"

    if not csv_path.is_file():
        return {}

    if not cache_settings_match(
        cache_dir / "summary.json",
        directory,
        reference_dir,
        args,
    ):
        return {}

    try:

        with csv_path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:

            reader = csv.DictReader(file)

            if (
                reader.fieldnames is None
                or not set(PER_IMAGE_COLUMNS).issubset(
                    reader.fieldnames
                )
            ):
                return {}

            cached: dict[str, dict[str, Any]] = {}

            for raw_row in reader:

                sample_id = raw_row["sample_id"]

                if (
                    raw_row["method"] != directory.name
                    or sample_id not in outputs
                ):
                    continue

                if normalized_path(
                    raw_row["generated_file"]
                ) != normalized_path(
                    outputs[sample_id]
                ):
                    continue

                cached[sample_id] = (
                    deserialize_cached_row(raw_row)
                )

            return cached

    except (
        OSError,
        TypeError,
        ValueError,
    ):
        # A partial or hand-edited CSV is not a valid cache. Rebuild it.
        return {}


def configure_cpu_worker() -> None:

    """Prevent OpenCV threads from oversubscribing worker processes."""

    try:
        import cv2

        cv2.setNumThreads(1)

    except ImportError:
        pass


def resolved_worker_count(
    args: argparse.Namespace,
    lpips: LPIPS | None,
) -> int:

    # The LPIPS model lives on one CPU/GPU device and strict mode must
    # preserve its original fail-fast behavior.
    if lpips is not None or args.strict:
        return 1

    if args.workers:
        return args.workers

    return min(
        8,
        max(
            1,
            os.cpu_count() or 1,
        ),
    )


def evaluate_sample(
    sample_id: str,
    generated_path: Path,
    directory: Path,
    reference_dir: Path | None,
    args: argparse.Namespace,
    lpips: LPIPS | None,
) -> dict[str, Any]:

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    generated = load_rgb(
        generated_path
    )

    original = load_rgb(
        reference_file(
            f"{sample_id}_original.png",
            directory,
            reference_dir,
        )
    )

    input_image = load_rgb(
        reference_file(
            f"{sample_id}_input.png",
            directory,
            reference_dir,
        )
    )

    # --------------------------------------------------------
    # Shape validation
    # --------------------------------------------------------

    if (
        original.shape != generated.shape
        or original.shape != input_image.shape
    ):
        raise ValueError(
            "generated/original/input shapes "
            "must match, got "
            f"{generated.shape}, "
            f"{original.shape}, "
            f"{input_image.shape}"
        )

    # --------------------------------------------------------
    # ROI mask and boundary
    # --------------------------------------------------------

    mask, source = repair_mask(
        sample_id,
        directory,
        reference_dir,
        original,
        input_image,
        args.mask_threshold,
        args.changed_tolerance,
    )

    boundary_mask = make_boundary_mask(
        mask,
        args.boundary_width,
    )

    # --------------------------------------------------------
    # Pixel, structural, edge, and perceptual metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        original,
        generated,
        mask,
        boundary_mask,
        args.psnr_cap,
    )

    add_lpips(
        metrics,
        original,
        generated,
        mask,
        boundary_mask,
        lpips,
    )

    return {
        "method": directory.name,
        "sample_id": sample_id,
        "generated_file": str(generated_path),
        "mask_source": source,
        "image_height": original.shape[0],
        "image_width": original.shape[1],
        "roi_pixels": int(mask.sum()),
        "roi_fraction": float(mask.mean()),
        "boundary_pixels": int(boundary_mask.sum()),
        "boundary_fraction": float(boundary_mask.mean()),
        **metrics,
    }


def evaluate_directory(
    directory: Path,
    sample_ids: list[str],
    reference_dir: Path | None,
    args: argparse.Namespace,
    lpips: LPIPS | None,
    cache_dir: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, str]],
    list[dict[str, Any]],
]:

    outputs = generated_by_id(
        directory
    )

    total = len(
        sample_ids
    )

    rows_by_id = load_cached_rows(
        cache_dir,
        directory,
        outputs,
        reference_dir,
        args,
    )

    pending_ids = [
        sample_id
        for sample_id in sample_ids
        if sample_id not in rows_by_id
    ]

    completed = sum(
        sample_id in rows_by_id
        for sample_id in sample_ids
    )

    if completed:
        print(
            f"{directory.name}: reused "
            f"{completed}/{total} cached samples.",
            file=sys.stderr,
        )

    failures_by_id: dict[str, dict[str, str]] = {}

    def record_result(
        sample_id: str,
        error: Exception | None,
        row: dict[str, Any] | None,
    ) -> None:

        nonlocal completed

        completed += 1

        if error is None:
            assert row is not None
            rows_by_id[sample_id] = row

        else:
            message = str(error)

            if args.strict:
                raise RuntimeError(
                    f"{directory.name}/"
                    f"{sample_id}: "
                    f"{message}"
                ) from error

            failures_by_id[sample_id] = {
                "method": directory.name,
                "sample_id": sample_id,
                "error": message,
            }

            print(
                "Warning: "
                f"{directory.name}/"
                f"{sample_id}: "
                f"{message}",
                file=sys.stderr,
            )

        if (
            completed % 25 == 0
            or completed == total
        ):
            print(
                f"{directory.name}: "
                f"{completed}/{total}",
                file=sys.stderr,
            )

    workers = resolved_worker_count(
        args,
        lpips,
    )

    if pending_ids and workers > 1:
        print(
            f"{directory.name}: calculating "
            f"{len(pending_ids)} samples with "
            f"{workers} CPU workers.",
            file=sys.stderr,
        )

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=configure_cpu_worker,
        ) as executor:

            futures = {
                sample_id: executor.submit(
                    evaluate_sample,
                    sample_id,
                    outputs[sample_id],
                    directory,
                    reference_dir,
                    args,
                    None,
                )
                for sample_id in pending_ids
            }

            # Retrieve in sample order so CSV and warning order stay stable.
            for sample_id in pending_ids:

                try:
                    row = futures[sample_id].result()

                except Exception as error:
                    record_result(
                        sample_id,
                        error,
                        None,
                    )

                else:
                    record_result(
                        sample_id,
                        None,
                        row,
                    )

    else:

        for sample_id in pending_ids:

            try:
                row = evaluate_sample(
                    sample_id,
                    outputs[sample_id],
                    directory,
                    reference_dir,
                    args,
                    lpips,
                )

            except Exception as error:
                record_result(
                    sample_id,
                    error,
                    None,
                )

            else:
                record_result(
                    sample_id,
                    None,
                    row,
                )

    return (
        [
            rows_by_id[sample_id]
            for sample_id in sample_ids
            if sample_id in rows_by_id
        ],
        [
            failures_by_id[sample_id]
            for sample_id in sample_ids
            if sample_id in failures_by_id
        ],
        [
            rows_by_id[sample_id]
            for sample_id in sorted(rows_by_id)
        ],
    )


# ============================================================
# Summaries
# ============================================================

def summarize(
    rows: list[dict[str, Any]],
    names: list[str],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:

    result: dict[
        str,
        Any,
    ] = {
        "methods": {}
    }

    csv_rows = []

    for name in names:

        method_rows = [
            row
            for row in rows
            if row[
                "method"
            ] == name
        ]

        report = {
            "num_images":
                len(method_rows),

            "num_images_with_repair_region":
                sum(
                    row[
                        "roi_pixels"
                    ] > 0
                    for row
                    in method_rows
                ),

            "metrics":
                {},
        }

        csv_row = {

            "method":
                name,

            "num_images":
                report[
                    "num_images"
                ],

            "num_images_with_repair_region":
                report[
                    "num_images_with_repair_region"
                ],
        }

        for metric in METRICS:

            values = np.asarray(
                [
                    row[
                        metric
                    ]
                    for row
                    in method_rows
                ],
                dtype=float,
            )

            values = values[
                np.isfinite(values)
            ]

            stats = {
                "count":
                    int(
                        len(values)
                    ),

                "mean":
                    None,

                "std":
                    None,

                "median":
                    None,

                "min":
                    None,

                "max":
                    None,
            }

            if len(values):

                stats.update(

                    mean=float(
                        values.mean()
                    ),

                    std=float(
                        values.std()
                    ),

                    median=float(
                        np.median(
                            values
                        )
                    ),

                    min=float(
                        values.min()
                    ),

                    max=float(
                        values.max()
                    ),
                )

            report[
                "metrics"
            ][
                metric
            ] = stats

            csv_row[
                f"{metric}_mean"
            ] = stats[
                "mean"
            ]

        result[
            "methods"
        ][
            name
        ] = report

        csv_rows.append(
            csv_row
        )

    return (
        result,
        csv_rows,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    # --------------------------------------------------------
    # Discover result folders
    # --------------------------------------------------------

    directories = discover_dirs(
        args
    )

    outputs = {
        directory:
            generated_by_id(
                directory
            )
        for directory
        in directories
    }

    # --------------------------------------------------------
    # Fair comparison:
    #
    # by default evaluate only images existing in all methods.
    # --------------------------------------------------------

    id_sets = [
        set(items.keys())
        for items
        in outputs.values()
    ]

    common_ids = (
        set.intersection(
            *id_sets
        )
    )

    if not common_ids:

        raise RuntimeError(
            "Selected result directories have no "
            "common *_generate.png sample IDs."
        )

    # --------------------------------------------------------
    # Sample selection
    # --------------------------------------------------------

    if args.all_available:

        sample_ids = {
            directory:
                sorted(
                    items.keys()
                )
            for directory, items
            in outputs.items()
        }

        print(
            "Using all available outputs; "
            "sample counts can differ between methods.",
            file=sys.stderr,
        )

    else:

        sample_ids = {
            directory:
                sorted(
                    common_ids
                )
            for directory
            in directories
        }

        counts = {
            directory.name:
                len(items)
            for directory, items
            in outputs.items()
        }

        if any(
            set(items.keys())
            != common_ids
            for items
            in outputs.values()
        ):

            print(
                "Fair comparison uses the common "
                f"{len(common_ids)} images "
                f"(available: {counts}).",
                file=sys.stderr,
            )

    # --------------------------------------------------------
    # Quick testing
    # --------------------------------------------------------

    if args.max_images is not None:

        sample_ids = {
            directory:
                ids[
                    :args.max_images
                ]
            for directory, ids
            in sample_ids.items()
        }

    # --------------------------------------------------------
    # Shared reference directory
    # --------------------------------------------------------

    reference_dir = (
        args.reference_dir.resolve()
        if args.reference_dir
        else None
    )

    if (
        reference_dir is not None
        and not reference_dir.is_dir()
    ):

        raise FileNotFoundError(
            "Reference directory does "
            f"not exist: {reference_dir}"
        )

    # --------------------------------------------------------
    # Optional LPIPS
    # --------------------------------------------------------

    lpips_evaluator = (

        LPIPS(
            args.device,
            args.lpips_size,
        )

        if args.lpips

        else None
    )

    # --------------------------------------------------------
    # Output root
    # --------------------------------------------------------

    output_dir = (
        args.output_dir.resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_columns = (

        "method",

        "num_images",

        "num_images_with_repair_region",

        *(
            f"{name}_mean"
            for name
            in METRICS
        ),
    )

    # --------------------------------------------------------
    # Evaluate every method
    # --------------------------------------------------------

    all_rows: list[
        dict[str, Any]
    ] = []

    all_failures: list[
        dict[str, str]
    ] = []

    for directory in directories:

        print(
            f"Evaluating "
            f"{directory.name} ...",
            file=sys.stderr,
        )

        method_output_dir = (
            output_dir
            / directory.name
        )

        method_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        rows, errors, cache_rows = evaluate_directory(

            directory,

            sample_ids[
                directory
            ],

            reference_dir,

            args,

            lpips_evaluator,

            method_output_dir,
        )

        all_rows.extend(
            rows
        )

        all_failures.extend(
            errors
        )

        # ----------------------------------------------------
        # Persistent cache
        # ----------------------------------------------------

        write_csv(
            method_output_dir
            / "metrics_cache.csv",

            cache_rows,

            PER_IMAGE_COLUMNS,
        )

        # ----------------------------------------------------
        # Per-image CSV
        #
        # Keep every known sample here. In a fair comparison, rows
        # contains only the common IDs, while cache_rows also preserves
        # previously evaluated IDs unique to this method.
        # ----------------------------------------------------

        write_csv(
            method_output_dir
            / "per_image_metrics.csv",

            cache_rows,

            PER_IMAGE_COLUMNS,
        )

        # This file is the exact subset used by summary.json and the
        # cross-method comparison of the current invocation.
        write_csv(
            method_output_dir
            / "current_comparison_metrics.csv",

            rows,

            PER_IMAGE_COLUMNS,
        )

        # ----------------------------------------------------
        # Failure CSV
        # ----------------------------------------------------

        write_csv(
            method_output_dir
            / "failures.csv",

            errors,

            (
                "method",
                "sample_id",
                "error",
            ),
        )

        # ----------------------------------------------------
        # Method summary
        # ----------------------------------------------------

        (
            method_summary,
            method_summary_rows,
        ) = summarize(

            rows,

            [
                directory.name
            ],
        )

        method_summary[
            "settings"
        ] = {

            "result_dir":
                str(
                    directory
                ),

            "reference_dir":
                (
                    str(
                        reference_dir
                    )
                    if reference_dir
                    else None
                ),

            "common_sample_intersection":
                not args.all_available,

            "lpips_enabled":
                args.lpips,

            "lpips_network":
                (
                    "alex"
                    if args.lpips
                    else None
                ),

            "lpips_resize_longest_edge":
                args.lpips_size,

            "boundary_width":
                args.boundary_width,

            "mask_threshold":
                args.mask_threshold,

            "changed_tolerance":
                args.changed_tolerance,

            "psnr_cap":
                args.psnr_cap,
        }

        # ----------------------------------------------------
        # Method JSON
        # ----------------------------------------------------

        with (
            method_output_dir
            / "summary.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                method_summary,
                file,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )

        # ----------------------------------------------------
        # Method summary CSV
        # ----------------------------------------------------

        write_csv(
            method_output_dir
            / "summary.csv",

            method_summary_rows,

            summary_columns,
        )

    # ========================================================
    # Global comparison
    # ========================================================

    summary, summary_rows = summarize(

        all_rows,

        [
            directory.name
            for directory
            in directories
        ],
    )

    summary[
        "settings"
    ] = {

        "result_dirs":
            [
                str(directory)
                for directory
                in directories
            ],

        "reference_dir":
            (
                str(
                    reference_dir
                )
                if reference_dir
                else None
            ),

        "common_sample_intersection":
            not args.all_available,

        "lpips_enabled":
            args.lpips,

        "lpips_network":
            (
                "alex"
                if args.lpips
                else None
            ),

        "lpips_resize_longest_edge":
            args.lpips_size,

        "boundary_width":
            args.boundary_width,

        "mask_threshold":
            args.mask_threshold,

        "changed_tolerance":
            args.changed_tolerance,

        "psnr_cap":
            args.psnr_cap,

        "metric_interpretation": {

            "mae":
                "lower is better; "
                "mean absolute RGB error",

            "mse":
                "lower is better; "
                "mean squared RGB error",

            "rmse":
                "lower is better; "
                "root mean squared RGB error",

            "nrmse":
                "lower is better; "
                "RMSE normalized by 255",

            "psnr":
                "higher is better; "
                "pixel-level fidelity",

            "ssim":
                "higher is better; "
                "structural similarity",

            "grad_mae":
                "lower is better; "
                "gradient/edge reconstruction error",

            "lpips":
                "lower is better; "
                "learned perceptual distance",

            "roi":
                "repair/inpainting region",

            "context":
                "pixels outside repair region",

            "boundary":
                (
                    "band surrounding the repair-mask "
                    "boundary; measures seam/transition quality"
                ),
        },
    }

    # --------------------------------------------------------
    # Comparison JSON
    # --------------------------------------------------------

    with (
        output_dir
        / "comparison_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    # --------------------------------------------------------
    # Comparison CSV
    # --------------------------------------------------------

    write_csv(
        output_dir
        / "comparison_summary.csv",

        summary_rows,

        summary_columns,
    )

    # --------------------------------------------------------
    # Comparison failures
    # --------------------------------------------------------

    write_csv(
        output_dir
        / "comparison_failures.csv",

        all_failures,

        (
            "method",
            "sample_id",
            "error",
        ),
    )

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print(
        "\nFinished."
    )

    print(
        "Reports saved to:",
        output_dir,
    )

    print(
        "\nRecommended primary metrics:"
    )

    print(
        "  ROI:"
        "       MAE↓ RMSE↓ PSNR↑ SSIM↑ "
        "Grad-MAE↓ LPIPS↓"
    )

    print(
        "  Boundary:"
        "  MAE↓ PSNR↑ SSIM↑ "
        "Grad-MAE↓ LPIPS↓"
    )

    print(
        "  Context:"
        "   MAE↓ LPIPS↓"
    )

    if not args.lpips:

        print(
            "\nLPIPS was NOT calculated."
        )

        print(
            "Enable with:"
        )

        print(
            "  python evaluate_pro.py "
            "--lpips --device cuda"
        )

    if all_failures:

        print(
            f"\nSkipped "
            f"{len(all_failures)} bad samples."
        )

        print(
            "See:",
            output_dir
            / "comparison_failures.csv",
        )


if __name__ == "__main__":
    main()
