"""不改动正式 pipeline 的逐步 latent 调试版本。

这个类复用 ``pipline.py`` 的完整推理逻辑，只在全图 scheduler 更新前
记录：当前 noisy latent x_t、融合后的噪声预测 epsilon，以及由两者计算的
全图 x0 预测。正式 pipeline 不会被改动。
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .pipline import SlidingWindowInstructPix2PixPipeline


class SlidingWindowInstructPix2PixDebugPipeline(
    SlidingWindowInstructPix2PixPipeline
):
    """保存每个 denoising step 的全图 noise 和 x0 预测的调试 pipeline。

    参数：
        debug_dir:
            调试结果目录。为 None 时，自动创建到本文件同级的
            ``debug_outputs/`` 目录。
        debug_save_every:
            每隔多少个 denoising step 保存一次；1 表示每一步都保存。
        debug_save_latents:
            是否保存可复算的 .pt latent（x_t / noise_pred / x0_pred）。
        debug_save_images:
            是否保存 PNG。noise 保存四通道可视化图；x_t 和 x0 都保存
            VAE decode 预览图。

    ``last_debug_dir`` 会在一次推理开始后指向本次调试结果目录。
    """

    @torch.no_grad()
    def __call__(
        self,
        *args: Any,
        debug_dir: str | Path | None = None,
        debug_save_every: int = 1,
        debug_save_latents: bool = True,
        debug_save_images: bool = True,
        **kwargs: Any,
    ):
        if debug_save_every < 1:
            raise ValueError("debug_save_every 必须大于等于 1。")

        if debug_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_dir = (
                Path(__file__).resolve().parent
                / "debug_outputs"
                / f"instruct_pix2pix_debug_{timestamp}"
            )
            output_dir.mkdir(parents=True, exist_ok=False)
        else:
            output_dir = Path(debug_dir).expanduser().resolve()
            output_dir.mkdir(parents=True, exist_ok=True)

        (output_dir / "latents").mkdir(exist_ok=True)
        (output_dir / "noise_preview").mkdir(exist_ok=True)
        (output_dir / "xt_preview").mkdir(exist_ok=True)
        (output_dir / "x0_preview").mkdir(exist_ok=True)

        self.last_debug_dir = output_dir
        print(f"[debug] 保存本次去噪过程到: {output_dir}", flush=True)
        records: list[dict[str, Any]] = []
        step_index = 0
        original_scheduler_step = self.scheduler.step

        def traced_scheduler_step(model_output, timestep, sample, *step_args, **step_kwargs):
            """记录主 scheduler 的输入，再交还给原 scheduler 更新。"""
            nonlocal step_index

            if step_index % debug_save_every == 0:
                x0_prediction = self.predict_original_sample(
                    sample=sample,
                    model_output=model_output,
                    timestep=timestep,
                )
                name = self._debug_step_name(step_index, timestep)

                if debug_save_latents:
                    # 保存在 CPU float32，避免 GPU 显存被调试数据长期占用，且便于离线分析。
                    torch.save(
                        sample.detach().float().cpu(),
                        output_dir / "latents" / f"{name}_x_t.pt",
                    )
                    torch.save(
                        model_output.detach().float().cpu(),
                        output_dir / "latents" / f"{name}_noise_pred.pt",
                    )
                    if x0_prediction is not None:
                        torch.save(
                            x0_prediction.detach().float().cpu(),
                            output_dir / "latents" / f"{name}_x0_pred.pt",
                        )

                if debug_save_images:
                    # noise 不能直接 VAE decode；这里保存带正负号的四通道 latent 可视化。
                    self._save_noise_preview(
                        noise=model_output,
                        output_path=output_dir / "noise_preview" / f"{name}.png",
                    )
                    # x_t 的早期预览会有很强噪声，这是正常现象；它用于观察
                    # 全图状态在每次 scheduler 更新前如何逐步收敛。
                    self._save_latent_preview(
                        latent=sample,
                        output_path=output_dir / "xt_preview" / f"{name}.png",
                    )
                    if x0_prediction is not None:
                        self._save_latent_preview(
                            latent=x0_prediction,
                            output_path=output_dir / "x0_preview" / f"{name}.png",
                        )

                records.append(
                    {
                        "step": step_index,
                        "timestep": self._debug_timestep_value(timestep),
                        "file_stem": name,
                        "has_x0_prediction": x0_prediction is not None,
                    }
                )

            step_index += 1
            return original_scheduler_step(
                model_output,
                timestep,
                sample,
                *step_args,
                **step_kwargs,
            )

        # 正式 __call__ 中对 self.scheduler.step 的调用只有一次 / timestep。
        # 临时替换它即可拿到“融合完成、整图更新之前”的准确全图量，且不触碰正式 pipeline。
        self.scheduler.step = traced_scheduler_step
        try:
            result = super().__call__(*args, **kwargs)
        finally:
            self.scheduler.step = original_scheduler_step
            metadata = {
                "description": (
                    "x_t 是 scheduler 更新前的全图带噪 latent；"
                    "noise_pred 是所有 crop 融合后的全图噪声预测；"
                    "x0_pred 是根据 scheduler 的 prediction_type 推导的全图 x0。"
                ),
                "debug_save_every": debug_save_every,
                "steps": records,
            }
            (output_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return result

    @staticmethod
    def _debug_timestep_value(timestep: Any) -> int | float | str:
        if torch.is_tensor(timestep):
            if timestep.numel() == 1:
                value = timestep.detach().cpu().item()
                return int(value) if float(value).is_integer() else float(value)
            return str(timestep.detach().cpu().tolist())
        if isinstance(timestep, (int, float)):
            return timestep
        return str(timestep)

    def _debug_step_name(self, step_index: int, timestep: Any) -> str:
        value = self._debug_timestep_value(timestep)
        if isinstance(value, float):
            value_text = f"{value:.4f}".replace(".", "p")
        else:
            value_text = str(value).replace("/", "_").replace(" ", "")
        return f"step_{step_index:03d}_t_{value_text}"

    @staticmethod
    def _save_noise_preview(noise: torch.Tensor, output_path: Path) -> None:
        """把 4 个 latent 噪声通道绘制为 2x2 灰度图；中灰表示 0。"""
        channels = noise.detach().float().cpu()[0].numpy()
        max_abs = float(np.quantile(np.abs(channels), 0.99))
        max_abs = max(max_abs, 1e-6)
        channels = np.clip(channels / max_abs, -1.0, 1.0)
        channels = np.rint((channels * 0.5 + 0.5) * 255.0).astype(np.uint8)

        channel_count, height, width = channels.shape
        columns = 2
        rows = math.ceil(channel_count / columns)
        padding = 4
        canvas = np.full(
            (rows * height + (rows - 1) * padding,
             columns * width + (columns - 1) * padding),
            127,
            dtype=np.uint8,
        )
        for channel_index in range(channel_count):
            row = channel_index // columns
            column = channel_index % columns
            top = row * (height + padding)
            left = column * (width + padding)
            canvas[top : top + height, left : left + width] = channels[channel_index]

        Image.fromarray(canvas, mode="L").save(output_path)

    def _save_latent_preview(
        self,
        latent: torch.Tensor,
        output_path: Path,
    ) -> None:
        """把 x_t 或 x0 latent 解码为便于肉眼观察的 RGB 图。"""
        decoded = self.vae.decode(
            latent / self.vae.config.scaling_factor,
            return_dict=False,
        )[0]
        image = self.image_processor.postprocess(
            decoded,
            output_type="pil",
            do_denormalize=[True] * decoded.shape[0],
        )[0]
        image.save(output_path)
