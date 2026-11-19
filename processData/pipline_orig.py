"""8 通道局部条件 baseline 的整图滑窗推理。"""

from typing import Any, Callable

import torch
import torch.nn.functional as F

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
import cv2
import numpy as np
from PIL import Image

from processData.pipline import SlidingWindowInstructPix2PixPipeline


class LocalConditionInstructPix2PixPipeline(SlidingWindowInstructPix2PixPipeline):
    """沿用原滑窗工具，修复掩码只控制窗口和整图 latent 的更新。"""

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str] = None,
        image: PipelineImageInput = None,
        inpaint_mask: PipelineImageInput = None,
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
        callback_on_step_end_tensor_inputs: list[str] = ["global_latents"],
        cross_attention_kwargs: dict[str, Any] | None = None,
        crop_process_batch_size: int = 8,
        confidence_min_weight: float = 0.05,
        confidence_context_radius: int = 128,
        confidence_agreement_strength: float = 1.0,
        confidence_temporal_strength: float = 0.5,
        confidence_propagation_gamma: float = 3.0,
        window_jitter_ratio: float = 0.4,
        mask_aware_window_routing: bool = False,
        mask_aware_min_known_fraction: float = 0.05,
        mask_aware_jitter_attempts: int = 16,
        window_seed: int | None = None,
        **kwargs,
    ):
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if image is None:
            raise ValueError("`image` cannot be None.")

        if inpaint_mask is None:
            raise ValueError("请提供 inpaint_mask 以确定滑窗的修复区域。")

        if self.vae.config.latent_channels != 4 or self.unet.config.in_channels != 8:
            raise ValueError("此流程需要 4 通道 VAE latent 和 8 通道 UNet 输入。")

        if crop_process_batch_size <= 0:
            raise ValueError("crop_process_batch_size 必须大于 0。")

        if crop_size <= 0 or crop_overlap < 0:
            raise ValueError("crop_size 必须大于 0，crop_overlap 必须大于等于 0。")

        if crop_size % self.vae_scale_factor != 0:
            raise ValueError(f"crop_size={crop_size} 必须能被 {self.vae_scale_factor} 整除。")

        if crop_overlap % self.vae_scale_factor != 0:
            raise ValueError(f"crop_overlap={crop_overlap} 必须能被 {self.vae_scale_factor} 整除。")

        if crop_overlap >= crop_size:
            raise ValueError("crop_overlap 必须小于 crop_size。")

        if not 0.0 < confidence_min_weight <= 1.0:
            raise ValueError("confidence_min_weight 必须在 (0, 1] 内。")

        if confidence_context_radius < 0:
            raise ValueError("confidence_context_radius 不能小于 0。")

        if confidence_agreement_strength < 0.0:
            raise ValueError("confidence_agreement_strength 不能小于 0。")

        if confidence_temporal_strength < 0.0:
            raise ValueError("confidence_temporal_strength 不能小于 0。")

        if confidence_propagation_gamma <= 0.0:
            raise ValueError("confidence_propagation_gamma 必须大于 0。")

        if not 0.0 <= window_jitter_ratio <= 0.5:
            raise ValueError("window_jitter_ratio 必须在 [0, 0.5] 内。")

        if not 0.0 <= mask_aware_min_known_fraction <= 1.0:
            raise ValueError("mask_aware_min_known_fraction 必须在 [0, 1] 内。")

        if mask_aware_jitter_attempts < 1:
            raise ValueError("mask_aware_jitter_attempts 必须大于等于 1。")

        self._guidance_scale = guidance_scale
        self._image_guidance_scale = image_guidance_scale

        device = self._execution_device

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError("这里建议直接传 prompt 字符串或 prompt list。")

        batch_size = batch_size * num_images_per_prompt

        if batch_size != 1:
            raise ValueError("这版先按单张图推理写，batch_size 目前保持 1。")

        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
        )

        dtype = prompt_embeds.dtype

        image = self.image_processor.preprocess(image)
        image = image.to(device=device, dtype=dtype)

        image_batch, _, pixel_h, pixel_w = image.shape
        if image_batch != 1:
            raise ValueError("滑窗推理一次处理一张图。")

        if pixel_h < crop_size or pixel_w < crop_size:
            raise ValueError(
                f"输入图像尺寸 {(pixel_h, pixel_w)} 小于 crop_size={crop_size}。"
            )

        if pixel_h % self.vae_scale_factor != 0 or pixel_w % self.vae_scale_factor != 0:
            raise ValueError(
                f"预处理后的图像尺寸 {(pixel_h, pixel_w)} 必须能被 {self.vae_scale_factor} 整除。"
            )

        latent_h = pixel_h // self.vae_scale_factor
        latent_w = pixel_w // self.vae_scale_factor
        latent_crop_size = crop_size // self.vae_scale_factor

        inpaint_mask_tensor = self.preprocess_mask_to_tensor(
            mask=inpaint_mask,
            height=pixel_h,
            width=pixel_w,
            device=device,
            dtype=dtype,
        )
        binary_mask = (
            inpaint_mask_tensor >= 0.5
        ).float()

        mask_np = (
            binary_mask[0, 0]
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        dist = cv2.distanceTransform(
            1 - mask_np,
            cv2.DIST_L2,
            5,
        )

        transition_width = 128

        weight = np.clip(
            1.0 - dist / transition_width,
            0.0,
            1.0,
        )

        weight[mask_np == 1] = 1.0

        weight = torch.from_numpy(weight)[None, None]

        inpaint_mask_latent = F.interpolate(
            weight,
            size=(latent_h, latent_w),
            mode="bilinear",
            align_corners=False,
        ).to(
            device=device,
            dtype=dtype,
        )

        minimum_random_overlap = max(
            crop_overlap // 4,
            self.vae_scale_factor,
        )
        base_tops, base_lefts = self.get_randomizable_window_starts(
            image_h=pixel_h,
            image_w=pixel_w,
            crop_size=crop_size,
            target_overlap=crop_overlap,
            minimum_overlap=minimum_random_overlap,
        )
        windows = self.make_pixel_windows(base_tops, base_lefts)

        mask_aware_base_windows = None
        if mask_aware_window_routing:
            mask_aware_base_windows = self.build_mask_aware_base_windows(
                repair_mask=mask_np,
                base_tops=base_tops,
                base_lefts=base_lefts,
                crop_size=crop_size,
                min_known_fraction=mask_aware_min_known_fraction,
            )
            if len(mask_aware_base_windows) == 0:
                print(
                    "Warning: mask-aware routing found no repair pixels; "
                    "falling back to regular windows.",
                    flush=True,
                )
                mask_aware_base_windows = None
            else:
                print(
                    "mask-aware routing: "
                    f"{len(mask_aware_base_windows)}/{len(windows)} "
                    "base windows intersect the repair mask.",
                    flush=True,
                )

        mask_aware_routing_enabled = mask_aware_base_windows is not None
        window_jitter_pixels = (
            int(
                crop_overlap
                * window_jitter_ratio
                / self.vae_scale_factor
            )
            * self.vae_scale_factor
        )
        window_generator = self.create_window_generator(
            generator=generator,
            window_seed=window_seed,
        )

        (
            repair_mask_latent,
            repair_distance_latent,
        ) = self.prepare_global_confidence_maps(
            inpaint_mask=inpaint_mask_tensor,
            latent_h=latent_h,
            latent_w=latent_w,
        )
        if mask_aware_routing_enabled:
            routing_update_mask_latent = (
                F.interpolate(
                    binary_mask,
                    size=(latent_h, latent_w),
                    mode="area",
                )
                > 0.0
            ).to(dtype=dtype)
        else:
            routing_update_mask_latent = torch.ones(
                1,
                1,
                latent_h,
                latent_w,
                device=device,
                dtype=dtype,
            )
        state_inpaint_mask_latent = (
            routing_update_mask_latent
            if mask_aware_routing_enabled
            else inpaint_mask_latent
        )
        original_latents_full = self.encode_image_full_to_latents(
            image=image,
            dtype=dtype,
            device=device,
        )

        base_prompt_embeds = prompt_embeds

        global_latents, global_noise = self.prepare_global_latents(
            latent_h=latent_h,
            latent_w=latent_w,
            num_channels_latents=self.vae.config.latent_channels,
            dtype=dtype,
            device=device,
            generator=generator,
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        num_warmup_steps = (
            len(timesteps) - num_inference_steps * self.scheduler.order
        )
        previous_global_pred_original_sample = None

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for step_index, t in enumerate(timesteps):

                if mask_aware_routing_enabled:
                    windows = self.sample_mask_aware_pixel_windows(
                        base_windows=mask_aware_base_windows,
                        repair_mask=mask_np,
                        image_h=pixel_h,
                        image_w=pixel_w,
                        crop_size=crop_size,
                        jitter_pixels=window_jitter_pixels,
                        min_known_fraction=mask_aware_min_known_fraction,
                        max_attempts=mask_aware_jitter_attempts,
                        generator=window_generator,
                    )
                else:
                    windows = self.sample_jittered_pixel_windows(
                        base_tops=base_tops,
                        base_lefts=base_lefts,
                        crop_size=crop_size,
                        minimum_overlap=minimum_random_overlap,
                        jitter_pixels=window_jitter_pixels,
                        generator=window_generator,
                    )

                num_crops = len(windows)
                crop_prompt_embeds = self.repeat_cfg_tensor_for_crops(
                    tensor=base_prompt_embeds,
                    num_crops=num_crops,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                )

                crop_latents = self.crop_latents_from_global(
                    global_latents=global_latents,
                    windows=windows,
                    latent_crop_size=latent_crop_size,
                )
                image_crop_latents = self.crop_latents_from_global(
                    global_latents=original_latents_full,
                    windows=windows,
                    latent_crop_size=latent_crop_size,
                )

                if self.do_classifier_free_guidance:
                    image_crop_latents = self.make_cfg_condition(
                        image_crop_latents
                    )

                if self.do_classifier_free_guidance:
                    latent_model_input = torch.cat(
                        [crop_latents, crop_latents, crop_latents],
                        dim=0,
                    )
                else:
                    latent_model_input = crop_latents

                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input,
                    t,
                )

                crop_noise_pred = self.run_conditioned_unet_in_batches(
                    latent_model_input=latent_model_input,
                    timestep=t,
                    image_condition=image_crop_latents,
                    prompt_embeds=crop_prompt_embeds,
                    process_batch_size=crop_process_batch_size,
                    cross_attention_kwargs=cross_attention_kwargs,
                )

                if self.do_classifier_free_guidance:
                    noise_pred_text, noise_pred_image, noise_pred_uncond = crop_noise_pred.chunk(3)

                    crop_noise_pred = (
                        noise_pred_uncond
                        + self.guidance_scale * (noise_pred_text - noise_pred_image)
                        + self.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                    )

                crop_update_masks = self.crop_latents_from_global(
                    global_latents=routing_update_mask_latent,
                    windows=windows,
                    latent_crop_size=latent_crop_size,
                )

                global_generation_confidence = self.build_generation_confidence_map(
                    repair_mask_latent=repair_mask_latent,
                    repair_distance_latent=repair_distance_latent,
                    step_index=step_index,
                    num_steps=len(timesteps),
                    propagation_gamma=confidence_propagation_gamma,
                )
                crop_generation_confidence = self.crop_latents_from_global(
                    global_latents=global_generation_confidence,
                    windows=windows,
                    latent_crop_size=latent_crop_size,
                )
                crop_context_support = self.prepare_crop_context_support(
                    repair_mask_latent=repair_mask_latent,
                    windows=windows,
                    latent_crop_size=latent_crop_size,
                    context_radius_pixels=confidence_context_radius,
                )
                _, global_crop_coverage = self.fuse_crop_tensors_to_global(
                    crop_tensors=crop_update_masks,
                    crop_weights=crop_update_masks,
                    windows=windows,
                    latent_h=latent_h,
                    latent_w=latent_w,
                    latent_crop_size=latent_crop_size,
                    return_weight_sum=True,
                )
                crop_overlap_counts = self.crop_latents_from_global(
                    global_latents=global_crop_coverage,
                    windows=windows,
                    latent_crop_size=latent_crop_size,
                )

                denoise_progress = (
                    1.0
                    if len(timesteps) == 1
                    else step_index / (len(timesteps) - 1)
                )
                context_weight = (
                    confidence_min_weight
                    + (1.0 - confidence_min_weight) * crop_context_support
                ).pow(1.0 - 0.5 * denoise_progress)
                base_crop_weights = context_weight * crop_update_masks

                preliminary_global_noise_pred = self.fuse_crop_tensors_to_global(
                    crop_tensors=crop_noise_pred,
                    crop_weights=base_crop_weights,
                    windows=windows,
                    latent_h=latent_h,
                    latent_w=latent_w,
                    latent_crop_size=latent_crop_size,
                )

                preliminary_noise_crops = self.crop_latents_from_global(
                    global_latents=preliminary_global_noise_pred,
                    windows=windows,
                    latent_crop_size=latent_crop_size,
                )
                agreement_residual = torch.sqrt(
                    self.masked_crop_channel_mse(
                        values=crop_noise_pred - preliminary_noise_crops,
                        spatial_mask=crop_update_masks,
                    )
                    + 1e-8,
                )
                agreement_weights = self.residuals_to_confidence_weights(
                    residuals=agreement_residual,
                    valid_mask=(crop_overlap_counts > 1.0)
                    & (crop_update_masks > 0.0),
                    min_weight=confidence_min_weight,
                )

                crop_confidence_weights = base_crop_weights * torch.pow(
                    agreement_weights,
                    crop_generation_confidence * confidence_agreement_strength,
                )

                crop_pred_original_sample = self.predict_original_sample(
                    sample=crop_latents,
                    model_output=crop_noise_pred,
                    timestep=t,
                )
                if (
                    previous_global_pred_original_sample is not None
                    and crop_pred_original_sample is not None
                    and confidence_temporal_strength > 0.0
                ):
                    previous_pred_original_crops = self.crop_latents_from_global(
                        global_latents=previous_global_pred_original_sample,
                        windows=windows,
                        latent_crop_size=latent_crop_size,
                    )
                    temporal_residual = torch.sqrt(
                        self.masked_crop_channel_mse(
                            values=(
                                crop_pred_original_sample
                                - previous_pred_original_crops
                            ),
                            spatial_mask=crop_update_masks,
                        )
                        + 1e-8,
                    )
                    temporal_weights = self.residuals_to_confidence_weights(
                        residuals=temporal_residual,
                        valid_mask=(crop_overlap_counts > 1.0)
                        & (crop_update_masks > 0.0),
                        min_weight=confidence_min_weight,
                    )
                    crop_confidence_weights = crop_confidence_weights * torch.pow(
                        temporal_weights,
                        crop_generation_confidence * confidence_temporal_strength,
                    )

                global_noise_pred = self.fuse_crop_tensors_to_global(
                    crop_tensors=crop_noise_pred,
                    crop_weights=crop_confidence_weights,
                    windows=windows,
                    latent_h=latent_h,
                    latent_w=latent_w,
                    latent_crop_size=latent_crop_size,
                )

                if crop_pred_original_sample is not None:
                    previous_global_pred_original_sample = (
                        self.predict_original_sample(
                            sample=global_latents,
                            model_output=global_noise_pred,
                            timestep=t,
                        )
                    )

                global_latents = self.scheduler.step(
                    global_noise_pred,
                    t,
                    global_latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )[0]

                # 用同一噪声级别的输入 latent 恢复非修复区域。
                if step_index < len(timesteps) - 1:
                    next_t = timesteps[step_index + 1]
                    next_t_batch = torch.as_tensor(next_t, device=device).reshape(1)
                    input_latents_next = self.scheduler.add_noise(
                        original_samples=original_latents_full,
                        noise=global_noise,
                        timesteps=next_t_batch,
                    )
                else:
                    input_latents_next = original_latents_full
                global_latents = (
                    state_inpaint_mask_latent * global_latents
                    + (1.0 - state_inpaint_mask_latent) * input_latents_next
                )

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

                    global_latents = callback_outputs.pop("global_latents", global_latents)

                if step_index == len(timesteps) - 1 or (
                    (step_index + 1) > num_warmup_steps
                    and (step_index + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                    if callback is not None and callback_steps is not None and step_index % callback_steps == 0:
                        callback(step_index, t, global_latents)

        if output_type != "latent":
            output_image_tensor = self.vae.decode(
                global_latents / self.vae.config.scaling_factor,
                return_dict=False,
            )[0]

            output_image_tensor, has_nsfw_concept = self.run_safety_checker(
                output_image_tensor,
                device,
                dtype,
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
            images = global_latents
            has_nsfw_concept = None

        self.maybe_free_model_hooks()

        if not return_dict:
            return images, has_nsfw_concept

        return StableDiffusionPipelineOutput(
            images=images,
            nsfw_content_detected=has_nsfw_concept,
        )

    def run_conditioned_unet_in_batches(
        self,
        latent_model_input,
        timestep,
        image_condition,
        prompt_embeds,
        process_batch_size,
        cross_attention_kwargs,
    ):
        """只向 UNet 传入加噪 crop、均值填充图的 crop 和文本。"""
        channels = self.vae.config.latent_channels
        if (
            latent_model_input.shape != image_condition.shape
            or latent_model_input.shape[1] != channels
            or self.unet.config.in_channels != 2 * channels
            or prompt_embeds.shape[0] != latent_model_input.shape[0]
        ):
            raise ValueError("Baseline 需要对齐的 4+4 通道 crop 和对应的文本 batch。")

        predictions = []
        for start in range(0, latent_model_input.shape[0], process_batch_size):
            batch = slice(start, start + process_batch_size)
            model_input = torch.cat(
                (latent_model_input[batch], image_condition[batch]), dim=1
            )
            predictions.append(
                self.unet(
                    model_input,
                    timestep,
                    encoder_hidden_states=prompt_embeds[batch],
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]
            )
        return torch.cat(predictions, dim=0)


class FixedTileInstructPix2PixPipeline(LocalConditionInstructPix2PixPipeline):
    """固定网格内独立去噪，按坐标直接写回每块 latent。"""

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        image: Image.Image,
        inpaint_mask: Image.Image,
        crop_size: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        image_guidance_scale: float = 2.0,
        crop_process_batch_size: int = 4,
        generator: torch.Generator | None = None,
    ):
        scale = self.vae_scale_factor
        if self.vae.config.latent_channels != 4 or self.unet.config.in_channels != 8:
            raise ValueError("此流程需要 4 通道 VAE latent 和 8 通道 UNet 输入。")
        if crop_size <= 0 or crop_size % scale or crop_process_batch_size <= 0:
            raise ValueError("分块尺寸必须是 VAE 缩放倍数，且分块批大小必须大于 0。")
        if image.size != inpaint_mask.size:
            raise ValueError("输入图和修复掩码尺寸必须一致。")

        source_image = image.convert("RGB")
        pixel_mask = inpaint_mask.convert("L").point(lambda value: 255 if value >= 128 else 0)
        if pixel_mask.getbbox() is None:
            return StableDiffusionPipelineOutput(images=[source_image], nsfw_content_detected=None)

        self._guidance_scale = guidance_scale
        self._image_guidance_scale = image_guidance_scale
        use_guidance = self.do_classifier_free_guidance
        device = self._execution_device
        prompt_embeds = self._encode_prompt(prompt, device, 1, use_guidance)
        dtype = prompt_embeds.dtype
        width, height = image.size
        pad_width, pad_height = (-width) % scale, (-height) % scale
        if pad_width or pad_height:
            image = Image.fromarray(np.pad(
                np.asarray(source_image),
                ((0, pad_height), (0, pad_width), (0, 0)),
                mode="edge",
            ))
            mask = np.pad(
                np.asarray(pixel_mask),
                ((0, pad_height), (0, pad_width)),
            )
            inpaint_mask = Image.fromarray(mask)
        else:
            image = source_image
            inpaint_mask = pixel_mask

        pixels = self.image_processor.preprocess(image)
        condition = self.encode_image_full_to_latents(pixels, dtype, device)
        latent_h, latent_w = condition.shape[-2:]
        repair = self.preprocess_mask_to_tensor(
            inpaint_mask, pixels.shape[-2], pixels.shape[-1], device, dtype
        )
        repair = F.max_pool2d((repair >= 0.5).to(dtype), scale, stride=scale)

        # 用同样的整图种子抽取初始噪声，再独立处理各块。
        initial, noise = self.prepare_global_latents(
            latent_h, latent_w, 4, dtype, device, generator
        )
        tile = crop_size // scale
        padding = (0, (-latent_w) % tile, 0, (-latent_h) % tile)
        condition = F.pad(condition, padding, mode="replicate")
        initial = F.pad(initial, padding)
        noise = F.pad(noise, padding)
        repair = F.pad(repair, padding)
        stitched = condition.clone()
        # 一次性筛出含修复像素的块，避免把空块混进 UNet 的批次。
        active_tiles = F.max_pool2d(repair, tile, stride=tile)[0, 0]
        tile_regions = [
            (slice(row * tile, (row + 1) * tile), slice(col * tile, (col + 1) * tile))
            for row, col in active_tiles.nonzero().tolist()
        ]

        for start in range(0, len(tile_regions), crop_process_batch_size):
            regions = tile_regions[start:start + crop_process_batch_size]
            tile_mask = torch.cat([repair[..., y, x] for y, x in regions])
            tile_condition = torch.cat([condition[..., y, x] for y, x in regions])
            tile_noise = torch.cat([noise[..., y, x] for y, x in regions])
            latents = torch.cat([initial[..., y, x] for y, x in regions])
            image_condition = self.make_cfg_condition(tile_condition) if use_guidance else tile_condition
            text_condition = self.repeat_cfg_tensor_for_crops(
                prompt_embeds, len(regions), use_guidance
            )

            # 每批 tile 重新开始 scheduler，历史状态仅属于本批的独立块。
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps
            step_kwargs = self.prepare_extra_step_kwargs(generator, eta=0.0)
            for step, timestep in enumerate(timesteps):
                model_latents = torch.cat([latents] * 3) if use_guidance else latents
                model_latents = self.scheduler.scale_model_input(model_latents, timestep)
                prediction = self.run_conditioned_unet_in_batches(
                    model_latents, timestep, image_condition, text_condition,
                    crop_process_batch_size, None,
                )
                if use_guidance:
                    text, image_only, uncond = prediction.chunk(3)
                    prediction = (
                        uncond + guidance_scale * (text - image_only)
                        + image_guidance_scale * (image_only - uncond)
                    )
                latents = self.scheduler.step(
                    prediction, timestep, latents, **step_kwargs, return_dict=False
                )[0]
                known = tile_condition
                if step + 1 < len(timesteps):
                    known = self.scheduler.add_noise(
                        tile_condition, tile_noise, timesteps[step + 1].reshape(1)
                    )
                latents = torch.where(tile_mask.bool(), latents, known)

            for index, (y, x) in enumerate(regions):
                stitched[..., y, x] = latents[index:index + 1]

        decoded = self.vae.decode(
            stitched[..., :latent_h, :latent_w] / self.vae.config.scaling_factor,
            return_dict=False,
        )[0]
        generated = self.image_processor.postprocess(decoded[..., :height, :width], output_type="pil")[0]
        result = Image.composite(generated, source_image, pixel_mask)
        self.maybe_free_model_hooks()
        return StableDiffusionPipelineOutput(images=[result], nsfw_content_detected=None)
