"""Evaluate paired image-inpainting results.

Files are discovered from ``<sample>_generate.png``.  Each result directory
needs matching ``<sample>_original.png`` (ground truth) and
``<sample>_input.png`` (the damaged input).  ``<sample>_inpaint_mask.png`` is
used when it exists; otherwise the repair area is rebuilt from the difference
between original and input.

Examples
--------
Compare every ``results_*`` directory in the current directory::

    python evaluate.py

Compare selected methods, with one directory as the common reference source::

    python evaluate.py --result-dirs results_global_latent_inpaint_step100000 \
        results_global_noisy_inpaint_step100000 \
        results_global_noisy_inpaint_confidence_step100000 \
        --reference-dir results_global_latent_inpaint_step100000

Add the optional perceptual metric (requires ``pip install lpips``)::

    python evaluate.py --lpips --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


GENERATED_SUFFIX = "_generate.png"
METRICS = (
    "full_mae", "full_rmse", "full_psnr", "full_ssim",
    "roi_mae", "roi_rmse", "roi_psnr", "roi_ssim",
    "context_mae", "context_rmse", "context_psnr", "context_ssim",
    "full_lpips", "roi_lpips",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare paired *_generate.png inpainting outputs."
    )
    parser.add_argument(
        "--result-dirs", type=Path, nargs="+",
        help="Result directories. Default: all --glob directories under --root.",
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--glob", default="results_*")
    parser.add_argument(
        "--reference-dir", type=Path,
        help="Shared directory for original/input/inpaint-mask files when absent in a method directory.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("evaluation"),
        help="Report root. Each result directory gets its own subdirectory here (default: evaluation).",
    )
    parser.add_argument(
        "--all-available", action="store_true",
        help="Do not restrict all methods to their common generated-image IDs.",
    )
    parser.add_argument("--max-images", type=int, help="Limit images per method for a quick test.")
    parser.add_argument(
        "--changed-tolerance", type=int, default=0,
        help="RGB threshold used when reconstructing an absent inpaint mask (default: 0).",
    )
    parser.add_argument(
        "--mask-threshold", type=int, default=127,
        help="Values above this in *_inpaint_mask.png are part of the repair region.",
    )
    parser.add_argument(
        "--psnr-cap", type=float, default=100.0,
        help="Finite PSNR reported for a perfect reconstruction (default: 100).",
    )
    parser.add_argument(
        "--lpips", action="store_true",
        help="Calculate optional LPIPS perceptual metrics (requires package lpips).",
    )
    parser.add_argument("--device", default="auto", help="LPIPS device: auto, cpu, cuda, or cuda:N.")
    parser.add_argument(
        "--lpips-size", type=int, default=512,
        help="Resize LPIPS input so its longest side is this many pixels; 0 keeps source size.",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Stop on a bad sample instead of recording it in failures.csv.",
    )
    args = parser.parse_args()
    if (
        args.changed_tolerance < 0
        or not 0 <= args.mask_threshold <= 255
        or args.lpips_size < 0
        or args.psnr_cap <= 0
        or args.max_images is not None and args.max_images < 1
    ):
        parser.error("changed tolerance/lpips size must be non-negative and mask threshold must be in [0, 255]")
    return args


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def discover_dirs(args: argparse.Namespace) -> list[Path]:
    directories = [p.resolve() for p in args.result_dirs] if args.result_dirs else sorted(
        p.resolve() for p in args.root.resolve().glob(args.glob) if p.is_dir()
    )
    valid = []
    for directory in directories:
        if not directory.is_dir():
            raise FileNotFoundError(f"Result directory does not exist: {directory}")
        if any(directory.glob(f"*{GENERATED_SUFFIX}")):
            valid.append(directory)
        else:
            print(f"Warning: {directory} has no *{GENERATED_SUFFIX}; skipped.", file=sys.stderr)
    if not valid:
        raise FileNotFoundError("No generated result images found. Check --result-dirs or --root/--glob.")
    if len({p.name for p in valid}) != len(valid):
        raise ValueError("Result directory names must be unique because they label the report.")
    return valid


def generated_by_id(directory: Path) -> dict[str, Path]:
    paths = {}
    for path in sorted(directory.glob(f"*{GENERATED_SUFFIX}")):
        sample_id = path.name[:-len(GENERATED_SUFFIX)]
        if sample_id in paths:
            raise ValueError(f"Duplicate generated sample {sample_id!r} in {directory}")
        paths[sample_id] = path
    return paths


def reference_file(filename: str, method_dir: Path, reference_dir: Path | None) -> Path:
    """Method-local metadata wins; a reference directory fills in missing files."""
    path = method_dir / filename
    if path.is_file():
        return path
    if reference_dir is not None and (reference_dir / filename).is_file():
        return reference_dir / filename
    checked = f"{path}" if reference_dir is None else f"{path} or {reference_dir / filename}"
    raise FileNotFoundError(f"Missing reference file: {checked}")


def repair_mask(
    sample_id: str,
    method_dir: Path,
    reference_dir: Path | None,
    original: np.ndarray,
    input_image: np.ndarray,
    threshold: int,
    changed_tolerance: int,
) -> tuple[np.ndarray, str]:
    """Find actual filled/inpainted pixels, never the model-conditioning *_mask.png."""
    try:
        saved_mask = load_gray(reference_file(f"{sample_id}_inpaint_mask.png", method_dir, reference_dir))
        if saved_mask.shape != original.shape[:2]:
            raise ValueError(f"inpaint-mask size {saved_mask.shape} differs from image size {original.shape[:2]}")
        return saved_mask > threshold, "saved_inpaint_mask"
    except FileNotFoundError:
        delta = np.abs(original.astype(np.int16) - input_image.astype(np.int16))
        return np.any(delta > changed_tolerance, axis=2), "derived_from_original_input"


def compute_ssim_map(original: np.ndarray, generated: np.ndarray) -> np.ndarray:
    """11x11 Gaussian-window RGB SSIM map (the standard SSIM constants)."""
    try:
        import cv2
    except ImportError:
        cv2 = None
    if min(original.shape[:2]) < 11:
        raise ValueError("Images smaller than 11 pixels cannot be scored with SSIM")
    x = original.astype(np.float64)
    y = generated.astype(np.float64)
    if cv2 is not None:
        def gaussian_blur(image: np.ndarray) -> np.ndarray:
            return cv2.GaussianBlur(image, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT)
    else:
        # This dependency-free fallback is slower than OpenCV but preserves a
        # runnable evaluator in lightweight analysis environments.
        weights = np.exp(-0.5 * (np.arange(-5, 6) / 1.5) ** 2)
        weights /= weights.sum()

        def gaussian_blur(image: np.ndarray) -> np.ndarray:
            padded = np.pad(image, ((0, 0), (5, 5), (0, 0)), mode="reflect")
            horizontal = sum(
                weight * padded[:, offset:offset + image.shape[1], :]
                for offset, weight in enumerate(weights)
            )
            padded = np.pad(horizontal, ((5, 5), (0, 0), (0, 0)), mode="reflect")
            return sum(
                weight * padded[offset:offset + image.shape[0], :, :]
                for offset, weight in enumerate(weights)
            )

    mu_x = gaussian_blur(x)
    mu_y = gaussian_blur(y)
    sigma_x = gaussian_blur(x * x) - mu_x * mu_x
    sigma_y = gaussian_blur(y * y) - mu_y * mu_y
    sigma_xy = gaussian_blur(x * y) - mu_x * mu_y
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    values = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    )
    return values.mean(axis=2)


def region_mean(values: np.ndarray, region: np.ndarray | None) -> float:
    if region is None:
        return float(values.mean())
    return float(values[region].mean()) if np.any(region) else math.nan


def psnr(mse: float, cap: float) -> float:
    if math.isnan(mse):
        return math.nan
    if mse == 0:
        return cap
    return min(20 * math.log10(255 / math.sqrt(mse)), cap)


def calculate_metrics(
    original: np.ndarray, generated: np.ndarray, mask: np.ndarray, psnr_cap: float
) -> dict[str, float]:
    difference = original.astype(np.float64) - generated.astype(np.float64)
    per_pixel_abs = np.abs(difference).mean(axis=2)
    per_pixel_sq = (difference * difference).mean(axis=2)
    ssim = compute_ssim_map(original, generated)
    metrics = {}
    for name, region in (("full", None), ("roi", mask), ("context", ~mask)):
        mae = region_mean(per_pixel_abs, region)
        mse = region_mean(per_pixel_sq, region)
        metrics[f"{name}_mae"] = mae
        metrics[f"{name}_rmse"] = math.sqrt(mse) if not math.isnan(mse) else math.nan
        metrics[f"{name}_psnr"] = psnr(mse, psnr_cap)
        metrics[f"{name}_ssim"] = region_mean(ssim, region)
    return metrics


class LPIPS:
    """Lazy optional LPIPS backend; keeping it optional avoids model downloads by default."""

    def __init__(self, device_name: str, longest_edge: int) -> None:
        try:
            import lpips
            import torch
        except ImportError as error:
            raise RuntimeError("--lpips needs the optional package. Install it with: pip install lpips") from error
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.device = torch.device(device_name)
        self.longest_edge = longest_edge
        self.model = lpips.LPIPS(net="alex").to(self.device).eval()
        print(f"LPIPS enabled on {self.device} (AlexNet).", file=sys.stderr)

    def tensor(self, image: np.ndarray) -> Any:
        pil = Image.fromarray(image)
        if self.longest_edge and max(pil.size) > self.longest_edge:
            scale = self.longest_edge / max(pil.size)
            pil = pil.resize(
                (max(1, round(pil.width * scale)), max(1, round(pil.height * scale))),
                Image.Resampling.BICUBIC,
            )
        array = np.asarray(pil, dtype=np.float32) / 127.5 - 1
        return self.torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).unsqueeze(0).to(self.device)

    def __call__(self, original: np.ndarray, generated: np.ndarray) -> float:
        with self.torch.inference_mode():
            return float(self.model(self.tensor(original), self.tensor(generated)).item())


def add_lpips(
    metrics: dict[str, float], original: np.ndarray, generated: np.ndarray, mask: np.ndarray, evaluator: LPIPS | None
) -> None:
    if evaluator is None:
        metrics["full_lpips"] = metrics["roi_lpips"] = math.nan
        return
    # Composite ground truth outside ROI to measure only repaired content while
    # retaining real surrounding context for LPIPS's receptive field.
    roi_composite = original.copy()
    roi_composite[mask] = generated[mask]
    metrics["full_lpips"] = evaluator(original, generated)
    metrics["roi_lpips"] = evaluator(original, roi_composite) if np.any(mask) else math.nan


def number_or_none(value: Any) -> Any:
    return None if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)) else value


def write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if number_or_none(row.get(key)) is None else row.get(key, "") for key in columns})


def evaluate_directory(
    directory: Path, sample_ids: list[str], reference_dir: Path | None,
    args: argparse.Namespace, lpips: LPIPS | None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    outputs = generated_by_id(directory)
    rows, failures = [], []
    for index, sample_id in enumerate(sample_ids, 1):
        try:
            generated = load_rgb(outputs[sample_id])
            original = load_rgb(reference_file(f"{sample_id}_original.png", directory, reference_dir))
            input_image = load_rgb(reference_file(f"{sample_id}_input.png", directory, reference_dir))
            if original.shape != generated.shape or original.shape != input_image.shape:
                raise ValueError(
                    f"generated/original/input shapes must match, got {generated.shape}, {original.shape}, {input_image.shape}"
                )
            mask, source = repair_mask(
                sample_id, directory, reference_dir, original, input_image,
                args.mask_threshold, args.changed_tolerance,
            )
            metrics = calculate_metrics(original, generated, mask, args.psnr_cap)
            add_lpips(metrics, original, generated, mask, lpips)
            rows.append({
                "method": directory.name, "sample_id": sample_id,
                "generated_file": str(outputs[sample_id]), "mask_source": source,
                "image_height": original.shape[0], "image_width": original.shape[1],
                "roi_pixels": int(mask.sum()), "roi_fraction": float(mask.mean()), **metrics,
            })
        except Exception as error:
            message = str(error)
            if args.strict:
                raise RuntimeError(f"{directory.name}/{sample_id}: {message}") from error
            failures.append({"method": directory.name, "sample_id": sample_id, "error": message})
            print(f"Warning: {directory.name}/{sample_id}: {message}", file=sys.stderr)
        if index % 25 == 0 or index == len(sample_ids):
            print(f"{directory.name}: {index}/{len(sample_ids)}", file=sys.stderr)
    return rows, failures


def summarize(rows: list[dict[str, Any]], names: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result, csv_rows = {"methods": {}}, []
    for name in names:
        method_rows = [row for row in rows if row["method"] == name]
        report = {
            "num_images": len(method_rows),
            "num_images_with_repair_region": sum(row["roi_pixels"] > 0 for row in method_rows),
            "metrics": {},
        }
        csv_row = {
            "method": name, "num_images": report["num_images"],
            "num_images_with_repair_region": report["num_images_with_repair_region"],
        }
        for metric in METRICS:
            values = np.asarray([row[metric] for row in method_rows], dtype=float)
            values = values[np.isfinite(values)]
            stats = {"count": int(len(values)), "mean": None, "std": None, "median": None}
            if len(values):
                stats.update(mean=float(values.mean()), std=float(values.std()), median=float(np.median(values)))
            report["metrics"][metric] = stats
            csv_row[f"{metric}_mean"] = stats["mean"]
        result["methods"][name] = report
        csv_rows.append(csv_row)
    return result, csv_rows


def main() -> None:
    args = parse_args()
    directories = discover_dirs(args)
    outputs = {directory: generated_by_id(directory) for directory in directories}
    common_ids = set.intersection(*(set(items) for items in outputs.values()))
    if not common_ids:
        raise RuntimeError("Selected result directories have no common *_generate.png sample IDs.")
    if args.all_available:
        sample_ids = {directory: sorted(items) for directory, items in outputs.items()}
        print("Using all available outputs; sample counts can differ between methods.", file=sys.stderr)
    else:
        sample_ids = {directory: sorted(common_ids) for directory in directories}
        counts = {directory.name: len(items) for directory, items in outputs.items()}
        if any(set(items) != common_ids for items in outputs.values()):
            print(f"Fair comparison uses the common {len(common_ids)} images (available: {counts}).", file=sys.stderr)
    if args.max_images is not None:
        sample_ids = {directory: ids[:args.max_images] for directory, ids in sample_ids.items()}

    reference_dir = args.reference_dir.resolve() if args.reference_dir else None
    if reference_dir is not None and not reference_dir.is_dir():
        raise FileNotFoundError(f"Reference directory does not exist: {reference_dir}")
    lpips = LPIPS(args.device, args.lpips_size) if args.lpips else None
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_image_columns = (
        "method", "sample_id", "generated_file", "mask_source", "image_height", "image_width",
        "roi_pixels", "roi_fraction", *METRICS,
    )
    summary_columns = (
        "method", "num_images", "num_images_with_repair_region", *(f"{name}_mean" for name in METRICS),
    )
    all_rows, all_failures = [], []
    for directory in directories:
        print(f"Evaluating {directory.name} ...", file=sys.stderr)
        rows, errors = evaluate_directory(directory, sample_ids[directory], reference_dir, args, lpips)
        all_rows.extend(rows)
        all_failures.extend(errors)

        # Keep the complete per-image and per-method reports beside each other,
        # one directory per generation strategy.
        method_output_dir = output_dir / directory.name
        method_output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(method_output_dir / "per_image_metrics.csv", rows, per_image_columns)
        write_csv(method_output_dir / "failures.csv", errors, ("method", "sample_id", "error"))
        method_summary, method_summary_rows = summarize(rows, [directory.name])
        method_summary["settings"] = {
            "result_dir": str(directory),
            "reference_dir": str(reference_dir) if reference_dir else None,
            "common_sample_intersection": not args.all_available,
            "lpips_enabled": args.lpips,
        }
        with (method_output_dir / "summary.json").open("w", encoding="utf-8") as file:
            json.dump(method_summary, file, ensure_ascii=False, indent=2, allow_nan=False)
        write_csv(method_output_dir / "summary.csv", method_summary_rows, summary_columns)

    summary, summary_rows = summarize(all_rows, [directory.name for directory in directories])
    summary["settings"] = {
        "result_dirs": [str(directory) for directory in directories],
        "reference_dir": str(reference_dir) if reference_dir else None,
        "common_sample_intersection": not args.all_available,
        "lpips_enabled": args.lpips,
        "metric_interpretation": {
            "roi_lpips": "lower is better; optional perceptual distance in repair region",
            "roi_ssim": "higher is better; structural similarity in repair region",
            "roi_psnr": "higher is better; pixel-level fidelity in repair region",
            "context_mae": "lower is better; unintended changes outside repair region",
        },
    }
    with (output_dir / "comparison_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
    write_csv(output_dir / "comparison_summary.csv", summary_rows, summary_columns)
    write_csv(output_dir / "comparison_failures.csv", all_failures, ("method", "sample_id", "error"))
    print(f"Finished. Per-strategy reports and comparison summary: {output_dir}")
    print("Use ROI metrics as primary: lower LPIPS/MAE and higher SSIM/PSNR; context MAE should be low.")
    if all_failures:
        print(f"Skipped {len(all_failures)} bad samples; see {output_dir / 'comparison_failures.csv'}.")


if __name__ == "__main__":
    main()
