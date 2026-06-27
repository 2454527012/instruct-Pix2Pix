from typing import Any, Callable

import PIL.Image
import torch
import torch.nn.functional as F

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_instruct_pix2pix import (
    StableDiffusionInstructPix2PixPipeline as BaseInstructPix2PixPipeline,
    retrieve_latents,
)
from diffusers.utils.torch_utils import randn_tensor


class SlidingWindowInstructPix2PixPipeline(BaseInstructPix2PixPipeline):
    """
    完全按你的逻辑写的版本：

    1. 先在像素空间把原图切成多个重叠 crop
    2. 每个 crop 单独 VAE encode，得到 image_crop_latents，作为条件
    3. 整图补黑边压缩到 crop_size，再 VAE encode，得到 square_image_latents，作为全局条件
    4. 每个 crop 单独初始化自己的 noisy latent
    5. 每一步 denoising：
        - 每个 crop 自己 UNet 预测 noise
        - 每个 crop 自己 scheduler.step，得到 next_crop_latent
        - 所有 crop latent 的重叠区域平均
        - 平均结果写回每个 crop latent
    6. 最后每个 crop latent 单独 VAE decode
    7. 在像素空间把所有 crop 拼回整图
    8. 像素空间重叠区域再次平均
    """

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str] = None,
        image: PipelineImageInput = None,
        crop_size: int = 512,
        crop_overlap: int = 128,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        negative_prompt: str | list[str] | None = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Callable[[int, int], None] | PipelineCallback | MultiPipelineCallbacks | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["crop_latents"],
        cross_attention_kwargs: dict[str, Any] | None = None,
        crop_process_batch_size: int = 8,
        **kwargs,
    ):
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if image is None:
            raise ValueError("`image` cannot be None.")

        if crop_size % self.vae_scale_factor != 0:
            raise ValueError(f"crop_size={crop_size} 必须能被 {self.vae_scale_factor} 整除。")

        if crop_overlap % self.vae_scale_factor != 0:
            raise ValueError(f"crop_overlap={crop_overlap} 必须能被 {self.vae_scale_factor} 整除。")

        if crop_overlap >= crop_size:
            raise ValueError("crop_overlap 必须小于 crop_size。")

        self._guidance_scale = guidance_scale
        self._image_guidance_scale = image_guidance_scale

        device = self._execution_device

        # ------------------------------------------------------------
        # 1. batch size
        # ------------------------------------------------------------
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError("这里建议直接传 prompt 字符串或 prompt list。")

        batch_size = batch_size * num_images_per_prompt

        if batch_size != 1:
            raise ValueError("这版先按单张图推理写，batch_size 目前建议保持 1。")

        # ------------------------------------------------------------
        # 2. 文本编码
        #
        # InstructPix2Pix CFG 会得到三份：
        #   text condition
        #   image condition
        #   uncondition
        # ------------------------------------------------------------
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
        )

        # ------------------------------------------------------------
        # 3. 原图预处理
        #
        # image:
        #   [1, 3, H, W]
        #   range [-1, 1]
        # ------------------------------------------------------------
        image = self.image_processor.preprocess(image)
        image = image.to(device=device, dtype=prompt_embeds.dtype)

        _, _, pixel_h, pixel_w = image.shape

        if pixel_h < crop_size or pixel_w < crop_size:
            raise ValueError(
                f"输入图像尺寸 {(pixel_h, pixel_w)} 小于 crop_size={crop_size}，"
                f"请减小 crop_size 或放大输入图。"
            )

        if pixel_h % self.vae_scale_factor != 0 or pixel_w % self.vae_scale_factor != 0:
            raise ValueError(
                f"预处理后的图像尺寸 {(pixel_h, pixel_w)} 必须能被 {self.vae_scale_factor} 整除。"
            )

        latent_crop_size = crop_size // self.vae_scale_factor
        latent_h = pixel_h // self.vae_scale_factor
        latent_w = pixel_w // self.vae_scale_factor

        # ------------------------------------------------------------
        # 4. 像素空间生成 crop 窗口
        #
        # 注意：
        #   这里的窗口是像素坐标，不是 latent 坐标。
        # ------------------------------------------------------------
        windows = self.get_adaptive_pixel_windows(
            image_h=pixel_h,
            image_w=pixel_w,
            crop_size=crop_size,
            target_overlap=crop_overlap,
        )

        num_crops = len(windows)

        # ------------------------------------------------------------
        # 5. 构造整图压缩条件 square image
        #
        # 这个对应你训练代码里的 square_orig。
        # 它是全局条件，不随 timestep 变化。
        # ------------------------------------------------------------
        square_image = self.make_square_image_tensor(
            image=image,
            crop_size=crop_size,
        )

        square_image_latents = self.encode_image_to_latents(
            image=square_image,
            dtype=prompt_embeds.dtype,
            device=device,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )

        # square_image_latents: [3, 4, 64, 64]
        # 每个 crop 都用同一个 square 条件，所以复制 num_crops 份
        square_image_latents = self.repeat_cfg_tensor_for_crops(
            square_image_latents,
            num_crops=num_crops,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )

        # ------------------------------------------------------------
        # 6. 先切像素 crop，再分别 VAE encode
        #
        # 这一步和你的训练代码一致：
        #   orig_cropped = Ft.crop(...)
        #   vae.encode(orig_cropped)
        # ------------------------------------------------------------
        image_crops = []

        for win in windows:
            top = win["pixel_top"]
            left = win["pixel_left"]

            crop = image[
                :,
                :,
                top : top + crop_size,
                left : left + crop_size,
            ]

            image_crops.append(crop)

        # [num_crops, 3, crop_size, crop_size]
        image_crops = torch.cat(image_crops, dim=0)

        # image_crop_latents:
        #   no CFG: [num_crops, 4, 64, 64]
        #   CFG:    [3*num_crops, 4, 64, 64]
        image_crop_latents = self.encode_image_to_latents(
            image=image_crops,
            dtype=prompt_embeds.dtype,
            device=device,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )

        # ------------------------------------------------------------
        # 7. 为每个 crop 准备 center map
        #
        # center map 也是条件，不随 timestep 变化。
        # ------------------------------------------------------------
        center_maps = []

        for win in windows:
            cm = self.prepare_center_maps_for_pixel_window(
                pixel_top=win["pixel_top"],
                pixel_left=win["pixel_left"],
                crop_size=crop_size,
                image_h=pixel_h,
                image_w=pixel_w,
                latent_crop_size=latent_crop_size,
                dtype=prompt_embeds.dtype,
                device=device,
                do_classifier_free_guidance=False,
            )
            center_maps.append(cm)

        # [num_crops, 2, 64, 64]
        center_maps = torch.cat(center_maps, dim=0)

        if self.do_classifier_free_guidance:
            uncond_center_maps = torch.zeros_like(center_maps)
            center_maps = torch.cat(
                [
                    center_maps,
                    center_maps,
                    uncond_center_maps,
                ],
                dim=0,
            )

        # ------------------------------------------------------------
        # 8. prompt_embeds 也要复制到每个 crop
        #
        # 原本 prompt_embeds:
        #   CFG: [3, seq, dim]
        #
        # 现在每个 crop 都要一份：
        #   CFG: [3*num_crops, seq, dim]
        # ------------------------------------------------------------
        prompt_embeds = self.repeat_cfg_tensor_for_crops(
            prompt_embeds,
            num_crops=num_crops,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )

        # ------------------------------------------------------------
        # 9. 初始化每个 crop 自己的 noisy latent
        #
        # crop_latents:
        #   [num_crops, 4, 64, 64]
        #
        # 注意：
        #   这里没有整图 latent。
        #   每个 crop 都有自己的 latent。
        # ------------------------------------------------------------
        crop_latents = self.prepare_crop_latents(
            windows=windows,
            latent_h=latent_h,
            latent_w=latent_w,
            num_channels_latents=self.vae.config.latent_channels,
            latent_crop_size=latent_crop_size,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
        )

        # ------------------------------------------------------------
        # 10. 通道检查
        # ------------------------------------------------------------
        total_channels = (
            self.vae.config.latent_channels
            + 4
            + 4
            + 2
        )

        if total_channels != self.unet.config.in_channels:
            raise ValueError(
                f"UNet 输入通道不匹配：当前 pipeline 构造 {total_channels} 通道，"
                f"但 unet.config.in_channels={self.unet.config.in_channels}。"
            )

        # ------------------------------------------------------------
        # 11. scheduler
        # ------------------------------------------------------------
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        # ------------------------------------------------------------
        # 12. denoising loop
        # ------------------------------------------------------------
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for step_index, t in enumerate(timesteps):
                # ----------------------------------------------------
                # 12.1 对所有 crop 做 CFG 扩展
                #
                # crop_latents:
                #   [N, 4, 64, 64]
                #
                # latent_model_input:
                #   [3N, 4, 64, 64]
                # ----------------------------------------------------
                if self.do_classifier_free_guidance:
                    latent_model_input = torch.cat(
                        [
                            crop_latents,
                            crop_latents,
                            crop_latents,
                        ],
                        dim=0,
                    )
                else:
                    latent_model_input = crop_latents

                check_tensor("crop_latents before scale", crop_latents)
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input,
                    t,
                )
                check_tensor("latent_model_input after scale", latent_model_input)
                next_crop_latents_chunks = []

                # ----------------------------------------------------
                # 12.2 分批跑 UNet，避免 crop 太多时爆显存
                # ----------------------------------------------------
                total_model_batch = latent_model_input.shape[0]
                
                for start in range(0, total_model_batch, crop_process_batch_size):
                    end = min(start + crop_process_batch_size, total_model_batch)

                    latent_chunk = latent_model_input[start:end]
                    image_cond_chunk = image_crop_latents[start:end]
                    square_cond_chunk = square_image_latents[start:end]
                    center_chunk = center_maps[start:end]
                    prompt_chunk = prompt_embeds[start:end]

                    check_tensor("latent_chunk", latent_chunk)
                    check_tensor("image_cond_chunk", image_cond_chunk)
                    check_tensor("square_cond_chunk", square_cond_chunk)
                    check_tensor("center_chunk", center_chunk)
                    check_tensor("prompt_chunk", prompt_chunk)

                    model_input = torch.cat(
                        [
                            latent_chunk,
                            image_cond_chunk,
                            square_cond_chunk,
                            center_chunk,
                        ],
                        dim=1,
                    )
                    check_tensor("model_input", model_input)

                    noise_pred_chunk = self.unet(
                        model_input,
                        t,
                        encoder_hidden_states=prompt_chunk,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                    )[0]
                    check_tensor("noise_pred_chunk", noise_pred_chunk)

                    next_crop_latents_chunks.append(noise_pred_chunk)

                crop_noise_pred = torch.cat(next_crop_latents_chunks, dim=0)

                # ----------------------------------------------------
                # 12.3 InstructPix2Pix CFG
                #
                # crop_noise_pred:
                #   [3N, 4, 64, 64]
                #
                # 得到：
                #   [N, 4, 64, 64]
                # ----------------------------------------------------
                if self.do_classifier_free_guidance:
                    noise_pred_text, noise_pred_image, noise_pred_uncond = crop_noise_pred.chunk(3)

                    crop_noise_pred = (
                        noise_pred_uncond
                        + self.guidance_scale * (noise_pred_text - noise_pred_image)
                        + self.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                    )

                # ----------------------------------------------------
                # 12.4 每个 crop 单独 scheduler.step
                #
                # 这里输入的是每个 crop 自己当前的 latent。
                # 输出是每个 crop 自己的 next latent。
                # ----------------------------------------------------
                next_crop_latents = self.scheduler.step(
                    crop_noise_pred,
                    t,
                    crop_latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )[0]
                check_tensor("next_crop_latents", next_crop_latents)
                # ----------------------------------------------------
                # 12.5 关键：同步 crop latent 的重叠区域
                #
                # 做法：
                #   临时建立一个整图 latent 累加画布
                #   把每个 crop latent 放到对应位置
                #   重叠处求平均
                #   再把平均结果切回每个 crop
                #
                # 注意：
                #   这不是生成整图 latent。
                #   这只是为了让 crop 之间的重叠区域变成一致。
                # ----------------------------------------------------
                # crop_latents = self.sync_crop_latent_overlaps(
                #     crop_latents=next_crop_latents,
                #     windows=windows,
                #     latent_h=latent_h,
                #     latent_w=latent_w,
                #     latent_crop_size=latent_crop_size,
                # )
                crop_latents = next_crop_latents
                check_tensor("crop_latents after sync", crop_latents)

                # callback
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]

                    callback_outputs = callback_on_step_end(
                        self,
                        step_index,
                        t,
                        callback_kwargs,
                    )

                    crop_latents = callback_outputs.pop("crop_latents", crop_latents)

                if step_index == len(timesteps) - 1 or (
                    (step_index + 1) > num_warmup_steps
                    and (step_index + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                    if callback is not None and callback_steps is not None and step_index % callback_steps == 0:
                        callback(step_index, t, crop_latents)

        # ------------------------------------------------------------
        # 13. 每个 crop latent 单独 VAE decode
        #
        # crop_images:
        #   [num_crops, 3, crop_size, crop_size]
        #   range 大致 [-1, 1]
        # ------------------------------------------------------------
        crop_images_list = []
        decode_batch_size = 2  # 显存不够就改成 1

        for start in range(0, crop_latents.shape[0], decode_batch_size):
            end = min(start + decode_batch_size, crop_latents.shape[0])
            crop_images_chunk = self.vae.decode(
                crop_latents[start:end] / self.vae.config.scaling_factor,
                return_dict=False,
            )[0]

            crop_images_list.append(crop_images_chunk)

        crop_images = torch.cat(crop_images_list, dim=0)

        # ------------------------------------------------------------
        # 14. 像素空间拼接，并对重叠区域简单平均
        #
        # 这一步得到和原图尺寸一致的图片 tensor：
        #   [1, 3, pixel_h, pixel_w]
        # ------------------------------------------------------------
        output_image_tensor = self.stitch_crop_images(
            crop_images=crop_images,
            windows=windows,
            image_h=pixel_h,
            image_w=pixel_w,
            crop_size=crop_size,
        )

        # ------------------------------------------------------------
        # 15. safety checker + postprocess
        # ------------------------------------------------------------
        if output_type != "latent":
            output_image_tensor, has_nsfw_concept = self.run_safety_checker(
                output_image_tensor,
                device,
                prompt_embeds.dtype,
            )
            if has_nsfw_concept is None:
                do_denormalize = [True] * output_image_tensor.shape[0]
            else:
                do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
            images = self.image_processor.postprocess(
                output_image_tensor,
                output_type=output_type,
                do_denormalize=do_denormalize,
            )
        else:
            images = crop_latents
            has_nsfw_concept = None

        self.maybe_free_model_hooks()

        if not return_dict:
            return (images, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=images,
            nsfw_content_detected=has_nsfw_concept,
        )

    # ================================================================
    # 工具函数 1：原图补黑边成正方形，再 resize
    # ================================================================
    def make_square_image_tensor(self, image: torch.Tensor, crop_size: int):
        """
        对应训练代码中的 square_orig。

        输入：
            image: [1, 3, H, W], range [-1, 1]

        输出：
            square: [1, 3, crop_size, crop_size]
        """

        b, c, h, w = image.shape
        square_size = max(h, w)

        square = image.new_full(
            (b, c, square_size, square_size),
            -1.0,
        )

        top = (square_size - h) // 2
        left = (square_size - w) // 2

        square[:, :, top : top + h, left : left + w] = image

        square = F.interpolate(
            square,
            size=(crop_size, crop_size),
            mode="bilinear",
            align_corners=False,
        )

        return square

    # ================================================================
    # 工具函数 2：像素空间自适应生成 crop 窗口
    # ================================================================
    def get_adaptive_pixel_windows(
        self,
        image_h: int,
        image_w: int,
        crop_size: int,
        target_overlap: int,
    ):
        """
        在像素空间生成 crop 坐标。

        目标：
            1. crop_size 固定，比如 512
            2. 重叠尽量接近 target_overlap，比如 128
            3. 如果不能刚好 128，可以稍微大一点
            4. 第一块贴左/上边
            5. 最后一块贴右/下边
            6. 完整覆盖原图
        """

        tops = self.get_adaptive_starts(
            length=image_h,
            crop=crop_size,
            target_overlap=target_overlap,
        )

        lefts = self.get_adaptive_starts(
            length=image_w,
            crop=crop_size,
            target_overlap=target_overlap,
        )

        windows = []

        for top in tops:
            for left in lefts:
                windows.append(
                    {
                        "pixel_top": top,
                        "pixel_left": left,
                    }
                )

        return windows

    def get_adaptive_starts(
        self,
        length: int,
        crop: int,
        target_overlap: int,
    ):
        """
        生成一个方向上的 crop 起点。

        例如：
            length = 2592
            crop = 512
            target_overlap = 128

        理想 stride:
            512 - 128 = 384

        如果不能整除，就自动调整 stride，
        保证最后一个 crop 刚好贴到右边/下边。
        """

        if length < crop:
            raise ValueError(f"length={length} 小于 crop={crop}。")

        if length == crop:
            return [0]

        target_stride = crop - target_overlap

        if target_stride <= 0:
            raise ValueError("target_overlap 太大。")

        max_start = length - crop

        # 用 ceil，保证实际 stride 不超过 target_stride
        # 也就是实际 overlap 通常 >= target_overlap
        num_intervals = (max_start + target_stride - 1) // target_stride
        num_intervals = max(num_intervals, 1)

        starts = []

        for i in range(num_intervals + 1):
            start = round(i * max_start / num_intervals)
            starts.append(int(start))

        starts = sorted(set(starts))

        if starts[0] != 0:
            starts.insert(0, 0)

        if starts[-1] != max_start:
            starts.append(max_start)

        return starts

    # ================================================================
    # 工具函数 3：image tensor -> VAE latent
    # ================================================================
    def encode_image_to_latents(
        self,
        image: torch.Tensor,
        dtype,
        device,
        do_classifier_free_guidance: bool,
    ):
        """
        输入：
            image:
                [N, 3, H, W] 或 [N, 4, H/8, W/8]

        输出：
            no CFG:
                [N, 4, H/8, W/8]

            CFG:
                [3N, 4, H/8, W/8]
                顺序是：
                    image_latents
                    image_latents
                    zero_latents
        """

        image = image.to(device=device, dtype=dtype)

        if image.shape[1] == 4:
            image_latents = image
        else:
            image_latents = retrieve_latents(
                self.vae.encode(image),
                sample_mode="argmax",
            )

        if do_classifier_free_guidance:
            zero_latents = torch.zeros_like(image_latents)

            image_latents = torch.cat(
                [
                    image_latents,
                    image_latents,
                    zero_latents,
                ],
                dim=0,
            )

        return image_latents

    # ================================================================
    # 工具函数 4：复制 CFG tensor 到每个 crop
    # ================================================================
    def repeat_cfg_tensor_for_crops(
        self,
        tensor: torch.Tensor,
        num_crops: int,
        do_classifier_free_guidance: bool,
    ):
        """
        原始 CFG tensor 是：
            [3, ...]

        现在有 num_crops 个 crop，需要变成：
            [3*num_crops, ...]

        并且顺序必须保持：
            text crops
            image crops
            uncond crops
        """

        if not do_classifier_free_guidance:
            return tensor.repeat_interleave(num_crops, dim=0)

        text_part, image_part, uncond_part = tensor.chunk(3, dim=0)

        text_part = text_part.repeat_interleave(num_crops, dim=0)
        image_part = image_part.repeat_interleave(num_crops, dim=0)
        uncond_part = uncond_part.repeat_interleave(num_crops, dim=0)

        return torch.cat(
            [
                text_part,
                image_part,
                uncond_part,
            ],
            dim=0,
        )

    # ================================================================
    # 工具函数 5：生成 center maps
    # ================================================================
    def prepare_center_maps_for_pixel_window(
        self,
        pixel_top: int,
        pixel_left: int,
        crop_size: int,
        image_h: int,
        image_w: int,
        latent_crop_size: int,
        dtype,
        device,
        do_classifier_free_guidance: bool,
    ):
        """
        和训练代码对齐。

        训练里：
            cx = (left + resolution / 2) / w_img
            cy = (top + resolution / 2) / h_img

        这里一样，只是 top/left 来自推理时的 crop 窗口。
        """

        cx = (pixel_left + crop_size / 2.0) / image_w
        cy = (pixel_top + crop_size / 2.0) / image_h

        center_position = torch.tensor(
            [[cx, cy]],
            dtype=dtype,
            device=device,
        )

        center_maps = center_position.view(
            1,
            2,
            1,
            1,
        ).expand(
            1,
            2,
            latent_crop_size,
            latent_crop_size,
        )

        if do_classifier_free_guidance:
            zero_maps = torch.zeros_like(center_maps)

            center_maps = torch.cat(
                [
                    center_maps,
                    center_maps,
                    zero_maps,
                ],
                dim=0,
            )

        return center_maps

    # ================================================================
    # 工具函数 6：初始化每个 crop 的 noisy latent
    # ================================================================
    def prepare_crop_latents(
        self,
        windows,
        latent_h: int,
        latent_w: int,
        num_channels_latents: int,
        latent_crop_size: int,
        dtype,
        device,
        generator,
    ):
        """
        先生成整图 latent 噪声，再切出每个 crop latent。

        这样一开始所有 crop 的重叠区域就是一致的。
        """

        global_shape = (
            1,
            num_channels_latents,
            latent_h,
            latent_w,
        )

        global_latents = randn_tensor(
            global_shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )

        global_latents = global_latents * self.scheduler.init_noise_sigma

        crop_latents = []

        for win in windows:
            latent_top = win["pixel_top"] // self.vae_scale_factor
            latent_left = win["pixel_left"] // self.vae_scale_factor

            crop = global_latents[
                :,
                :,
                latent_top : latent_top + latent_crop_size,
                latent_left : latent_left + latent_crop_size,
            ]

            crop_latents.append(crop)

        crop_latents = torch.cat(crop_latents, dim=0)

        return crop_latents
    # ================================================================
    # 工具函数 7：同步 crop latent 的重叠区域
    # ================================================================
    def sync_crop_latent_overlaps(
        self,
        crop_latents: torch.Tensor,
        windows: list[dict],
        latent_h: int,
        latent_w: int,
        latent_crop_size: int,
    ):
        """
        让 crop latent 的重叠区域变成一样。

        做法：
            1. 临时创建一个整图 latent canvas
            2. 把每个 crop latent 放到对应位置
            3. 重叠处累加
            4. 除以 count 得到平均
            5. 再把平均后的区域切回每个 crop

        结果：
            如果两个 crop 有重叠区域，
            那么同步后这两个 crop 的对应 latent 区域数值完全一样。
        """

        num_crops, channels, _, _ = crop_latents.shape
        device = crop_latents.device
        dtype = crop_latents.dtype

        global_sum = torch.zeros(
            1,
            channels,
            latent_h,
            latent_w,
            device=device,
            dtype=dtype,
        )

        global_count = torch.zeros(
            1,
            1,
            latent_h,
            latent_w,
            device=device,
            dtype=dtype,
        )

        # 先把所有 crop 放到全局 latent 坐标里
        for idx, win in enumerate(windows):
            latent_top = win["pixel_top"] // self.vae_scale_factor
            latent_left = win["pixel_left"] // self.vae_scale_factor

            global_sum[
                :,
                :,
                latent_top : latent_top + latent_crop_size,
                latent_left : latent_left + latent_crop_size,
            ] += crop_latents[idx : idx + 1]

            global_count[
                :,
                :,
                latent_top : latent_top + latent_crop_size,
                latent_left : latent_left + latent_crop_size,
            ] += 1.0

        global_avg = global_sum / global_count.clamp(min=1e-6)

        # 再把平均后的结果切回每个 crop
        synced = torch.empty_like(crop_latents)

        for idx, win in enumerate(windows):
            latent_top = win["pixel_top"] // self.vae_scale_factor
            latent_left = win["pixel_left"] // self.vae_scale_factor

            synced[idx : idx + 1] = global_avg[
                :,
                :,
                latent_top : latent_top + latent_crop_size,
                latent_left : latent_left + latent_crop_size,
            ]

        return synced

    # ================================================================
    # 工具函数 8：最终像素空间拼接
    # ================================================================
    def stitch_crop_images(
        self,
        crop_images: torch.Tensor,
        windows: list[dict],
        image_h: int,
        image_w: int,
        crop_size: int,
    ):
        """
        把 decode 出来的 crop image 拼回原图尺寸。

        重叠区域简单平均。

        crop_images:
            [num_crops, 3, crop_size, crop_size]

        输出：
            [1, 3, image_h, image_w]
        """

        device = crop_images.device
        dtype = crop_images.dtype

        output_sum = torch.zeros(
            1,
            3,
            image_h,
            image_w,
            device=device,
            dtype=dtype,
        )

        output_count = torch.zeros(
            1,
            1,
            image_h,
            image_w,
            device=device,
            dtype=dtype,
        )

        for idx, win in enumerate(windows):
            top = win["pixel_top"]
            left = win["pixel_left"]

            output_sum[
                :,
                :,
                top : top + crop_size,
                left : left + crop_size,
            ] += crop_images[idx : idx + 1]

            output_count[
                :,
                :,
                top : top + crop_size,
                left : left + crop_size,
            ] += 1.0

        output = output_sum / output_count.clamp(min=1e-6)

        return output
    
def check_tensor(name, x):
    if torch.isnan(x).any() or torch.isinf(x).any():
        print(name, x.min(), x.max(), x.mean())
        raise RuntimeError(f"{name} contains NaN or Inf")