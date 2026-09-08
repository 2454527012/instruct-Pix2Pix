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
import cv2
import numpy as np

class SlidingWindowInstructPix2PixPipeline(BaseInstructPix2PixPipeline):
    """
    新版逻辑：

    以前：
        1. 像素空间切 crop
        2. 每个 crop 单独 VAE encode
        3. 每个 crop 单独 denoise
        4. 每个 crop 单独 VAE decode
        5. 像素空间拼图

    现在：
        1. 整张图先 VAE encode 成整图 latent
        2. 初始化一张整图 noisy latent
        3. 每一步 denoising 时，使用新的受限随机窗口从整图 latent 切 crop
        4. 每个 latent crop 单独 UNet 预测噪声
        5. 把所有 crop 噪声预测写回整图噪声图
        6. 重叠 latent 区域平均
        7. 对整图 latent 执行一次 scheduler.step
        8. 最后只对整图 latent 做一次 VAE decode
    """

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str] = None,
        image: PipelineImageInput = None,
        mask: PipelineImageInput = None,
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
        global_context_strength: float = 0.85,
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

        if mask is None:
            raise ValueError("`mask` cannot be None，因为训练时使用了 mask 条件。")

        if inpaint_mask is None:
            raise ValueError("`inpaint_mask` cannot be None，因为融合可信度需要修复区域。")

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

        if not 0.0 <= global_context_strength <= 1.0:
            raise ValueError("global_context_strength 必须在 [0, 1] 内。")

        if not 0.0 <= window_jitter_ratio <= 0.5:
            raise ValueError("window_jitter_ratio 必须在 [0, 0.5] 内。")

        if not 0.0 <= mask_aware_min_known_fraction <= 1.0:
            raise ValueError("mask_aware_min_known_fraction 必须在 [0, 1] 内。")

        if mask_aware_jitter_attempts < 1:
            raise ValueError("mask_aware_jitter_attempts 必须大于等于 1。")

        self._guidance_scale = guidance_scale
        self._image_guidance_scale = image_guidance_scale

        device = self._execution_device

        # ------------------------------------------------------------
        # 1. batch size
        # ------------------------------------------------------------
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError("这里建议直接传 prompt 字符串或 prompt list。")

        batch_size = batch_size * num_images_per_prompt

        if batch_size != 1:
            raise ValueError("这版先按单张图推理写，batch_size 目前保持 1。")

        # ------------------------------------------------------------
        # 2. 文本编码
        #
        # InstructPix2Pix CFG 顺序：
        #   text condition
        #   image condition
        #   unconditional
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

        dtype = prompt_embeds.dtype

        # ------------------------------------------------------------
        # 3. 预处理整图 image
        #
        # image:
        #   [1, 3, H, W]
        #   range [-1, 1]
        # ------------------------------------------------------------
        image = self.image_processor.preprocess(image)
        image = image.to(device=device, dtype=dtype)

        _, _, pixel_h, pixel_w = image.shape

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

        # ------------------------------------------------------------
        # 4. 预处理 mask
        #
        # mask 会先整图 VAE encode 一次，随后每一步随随机窗口在 latent
        # 空间切出 mask crop，避免重复 encode 和 crop 边缘的 VAE padding。
        # ------------------------------------------------------------
        mask_image = self.preprocess_mask_to_tensor(
            mask=mask,
            height=pixel_h,
            width=pixel_w,
            device=device,
            dtype=dtype,
        )

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

        # 距离到白色区域
        dist = cv2.distanceTransform(
            1 - mask_np,
            cv2.DIST_L2,
            5,
        )

        # 过渡宽度（像素）
        transition_width = 128

        weight = np.clip(
            1.0 - dist / transition_width,
            0.0,
            1.0,
        )

        # 保证白色区域仍然是1
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

        # ------------------------------------------------------------
        # 5. 生成可随机偏移的基础窗口网格
        #
        # 注意：
        #   窗口仍然用像素坐标保存。
        #   后面切 latent 时再除以 vae_scale_factor。
        # ------------------------------------------------------------
        # 内部窗口每一步会使用同一个随机相位整体偏移；首尾窗口始终
        # 贴住图像边界，保证完整覆盖。最小 overlap 保留为原 overlap 的
        # 四分之一，避免随机偏移后出现覆盖空洞。
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

        # mask-aware routing 不改变全图 latent 的状态表示；它只减少 UNet
        # 实际处理的窗口。每个保留窗口都与修复区域相交，边界窗口还必须
        # 看到一定比例的已知像素。大区域的深中心窗口会作为桥接窗口保留，
        # 因此不会因为缺少已知像素而出现未覆盖的修复位置。
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

        # ------------------------------------------------------------
        # 5.1 预计算融合可信度的整图静态部分
        #
        # 不再使用“crop 中心高、边缘低”的位置先验。每个 crop 的基础
        # 可信度只由它在当前窗口内真正看见的非修复区域决定。
        # repair_mask/distance 是整图坐标系下的静态图；context_support 与
        # overlap coverage 会在每个 timestep 依据当前随机窗口重新计算。
        # ------------------------------------------------------------
        (
            repair_mask_latent,
            repair_distance_latent,
        ) = self.prepare_global_confidence_maps(
            inpaint_mask=inpaint_mask_tensor,
            latent_h=latent_h,
            latent_w=latent_w,
        )
        # 对 mask-aware 路径，任何被至少一个修复像素覆盖的 latent 单元
        # 都允许更新；其余单元在每一步都回写为输入图的带噪 latent。
        # ``area > 0`` 保留像素 mask 到 latent 网格的边界覆盖，避免细小
        # 修复区域在下采样时消失。
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
        # ------------------------------------------------------------
        # 6. 整图 VAE encode
        #
        # 这是新版最关键的第一处变化：
        #
        #   不再先切 image crop 再分别 VAE encode。
        #   而是直接把整张 image 编码成：
        #
        #       original_latents_full: [1, 4, H/8, W/8]
        #
        # 后面每个 crop 的 image condition 都从这个整图 latent 上切。
        # ------------------------------------------------------------
        original_latents_full = self.encode_image_full_to_latents(
            image=image,
            dtype=dtype,
            device=device,
        )

        # ------------------------------------------------------------
        # 7. 整图 mask 只 VAE encode 一次。
        #
        # 之后每个 timestep 都从 mask_latents_full 按新的随机窗口切条件，
        # 不会因窗口移动而重复 VAE encode，也避免 crop 边缘的人为 padding。
        # ------------------------------------------------------------
        mask_latents_full = self.encode_mask_to_latents(
            mask_images=mask_image,
            dtype=dtype,
            device=device,
            encode_batch_size=crop_process_batch_size,
        )

        # ------------------------------------------------------------
        # 7.1 全图低分辨率结构分支的条件
        #
        # 整图 image/mask 已经完成 VAE encode；这里仅在 latent 空间映射到
        # 一个与训练 crop 相同的方形网格。这样全图分支能在同一个
        # UNet 感受野里看到 mask 四周的真实道路、草地和边线，不会再次
        # VAE encode。
        # ------------------------------------------------------------
        (
            coarse_latent_h,
            coarse_latent_w,
        ) = self.get_global_context_latent_size(
            latent_h=latent_h,
            latent_w=latent_w,
            max_latent_side=latent_crop_size,
        )
        coarse_original_image_latents = F.interpolate(
            original_latents_full,
            size=(coarse_latent_h, coarse_latent_w),
            mode="bilinear",
            align_corners=False,
        )
        coarse_image_latents = coarse_original_image_latents
        coarse_mask_latents = F.interpolate(
            mask_latents_full,
            size=(coarse_latent_h, coarse_latent_w),
            mode="bilinear",
            align_corners=False,
        )
        coarse_center_maps = torch.full(
            (1, 2, coarse_latent_h, coarse_latent_w),
            0.5,
            device=device,
            dtype=dtype,
        )
        coarse_original_latents_full = original_latents_full
        coarse_prompt_embeds = prompt_embeds

        if self.do_classifier_free_guidance:
            coarse_image_latents = self.make_cfg_condition(
                coarse_image_latents
            )
            coarse_mask_latents = self.make_cfg_condition(
                coarse_mask_latents
            )
            coarse_center_maps = self.make_cfg_condition(
                coarse_center_maps
            )
            coarse_original_latents_full = self.make_cfg_condition(
                coarse_original_latents_full
            )

        # ------------------------------------------------------------
        # 8. 保存固定的整图/文本条件
        #
        # mask-aware routing 在不同 timestep 可能因覆盖补偿而拥有不同数量
        # 的 crop；因此整图 latent 与 prompt 的按-crop 扩展必须放到循环内。
        # ------------------------------------------------------------
        base_prompt_embeds = prompt_embeds

        # ------------------------------------------------------------
        # 13. 初始化整图 noisy latent
        #
        # 这是新版最关键的第二处变化：
        #
        # 旧版：
        #   crop_latents = [N, 4, 64, 64]
        #
        # 新版：
        #   global_latents = [1, 4, H/8, W/8]
        #
        # 每一步 denoise 时临时从 global_latents 切 crop。
        # ------------------------------------------------------------
        global_latents , global_noise= self.prepare_global_latents(
            latent_h=latent_h,
            latent_w=latent_w,
            num_channels_latents=self.vae.config.latent_channels,
            dtype=dtype,
            device=device,
            generator=generator,
        )

        # ------------------------------------------------------------
        # 14. 动态通道检查
        # ------------------------------------------------------------
        expected_in_channels = self.vae.config.latent_channels
        if self.unet.config.in_channels != expected_in_channels:
            raise ValueError(
                "新版 UNet 输入通道不匹配：\n"
                f"  noisy latent channels: {expected_in_channels}\n"
                f"  unet.config.in_channels: {self.unet.config.in_channels}\n"
                "新版 ContentAwarePositionGatedConvIn "
                "应将 unet.config.in_channels 注册为 4。"
            )

        # ------------------------------------------------------------
        # 15. scheduler
        # ------------------------------------------------------------
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 全图低分辨率分支必须使用独立 scheduler 状态。某些 scheduler
        # 会在 step() 内保存历史项；若与局部分支共用，会破坏两条轨迹。
        global_context_enabled = global_context_strength > 0.0
        if global_context_enabled:
            coarse_scheduler = self.scheduler.__class__.from_config(
                self.scheduler.config
            )
            coarse_scheduler.set_timesteps(
                num_inference_steps,
                device=device,
            )
            coarse_generator = self.create_window_generator(
                generator=generator,
                window_seed=window_seed,
                seed_offset=1,
            )
            coarse_extra_step_kwargs = self.prepare_extra_step_kwargs(
                coarse_generator,
                eta,
            )
            coarse_noise = F.interpolate(
                global_noise,
                size=(coarse_latent_h, coarse_latent_w),
                mode="area",
            )
            coarse_noise = self.normalize_noise_to_unit_variance(
                coarse_noise
            )
            coarse_latents = (
                coarse_noise * coarse_scheduler.init_noise_sigma
            )
            coarse_inpaint_mask_latent = F.interpolate(
                inpaint_mask_latent,
                size=(coarse_latent_h, coarse_latent_w),
                mode="bilinear",
                align_corners=False,
            )

        num_warmup_steps = (
            len(timesteps) - num_inference_steps * self.scheduler.order
        )
        previous_global_pred_original_sample = None

        # ------------------------------------------------------------
        # 16. denoising loop
        #
        # 每一步流程：
        #
        #   global_latents
        #       ↓ 按当前随机窗口切 latent crop
        #   crop_latents
        #       ↓ UNet 预测 noise
        #   crop_noise_pred
        #       ↓ 融合为整图 noise，重叠区域平均
        #   global_noise_pred
        #       ↓ scheduler.step
        #   global_latents
        #
        # 所以最后 global_latents 本身就是完整大图 latent。
        # ------------------------------------------------------------
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for step_index, t in enumerate(timesteps):
                # ----------------------------------------------------
                # 16.1 全图低分辨率结构预测
                #
                # 该分支在一个 512 级别的完整视野中预测噪声，因此对
                # 1024 级大 mask 的中心也能同时看到四周真实结构。
                # ----------------------------------------------------
                if global_context_enabled:
                    if self.do_classifier_free_guidance:
                        coarse_model_input = torch.cat(
                            [
                                coarse_latents,
                                coarse_latents,
                                coarse_latents,
                            ],
                            dim=0,
                        )
                    else:
                        coarse_model_input = coarse_latents

                    coarse_model_input = coarse_scheduler.scale_model_input(
                        coarse_model_input,
                        t,
                    )
                    coarse_noise_pred = self.run_conditioned_unet_in_batches(
                        latent_model_input=coarse_model_input,
                        timestep=t,
                        image_condition=coarse_image_latents,
                        mask_condition=coarse_mask_latents,
                        full_image_condition=coarse_original_latents_full,
                        center_condition=coarse_center_maps,
                        prompt_embeds=coarse_prompt_embeds,
                        process_batch_size=crop_process_batch_size,
                        cross_attention_kwargs=cross_attention_kwargs,
                    )

                    if self.do_classifier_free_guidance:
                        (
                            coarse_noise_pred_text,
                            coarse_noise_pred_image,
                            coarse_noise_pred_uncond,
                        ) = coarse_noise_pred.chunk(3)
                        coarse_noise_pred = (
                            coarse_noise_pred_uncond
                            + self.guidance_scale
                            * (
                                coarse_noise_pred_text
                                - coarse_noise_pred_image
                            )
                            + self.image_guidance_scale
                            * (
                                coarse_noise_pred_image
                                - coarse_noise_pred_uncond
                            )
                        )

                # ----------------------------------------------------
                # 16.2 为当前 timestep 采样新的受限随机窗口
                #
                # 常规路径仍使用整图网格。mask-aware 路径仅随机采样与
                # repair mask 相交的窗口，并保持边界窗口对已知区域的覆盖；
                # 抖动后若出现修复像素漏覆盖，会回退补入对应基础窗口。
                # ----------------------------------------------------
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
                original_latents_full_per_crop = original_latents_full.expand(
                    num_crops,
                    -1,
                    -1,
                    -1,
                )
                if self.do_classifier_free_guidance:
                    original_latents_full_per_crop = self.make_cfg_condition(
                        original_latents_full_per_crop
                    )
                crop_prompt_embeds = self.repeat_cfg_tensor_for_crops(
                    tensor=base_prompt_embeds,
                    num_crops=num_crops,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                )

                # ----------------------------------------------------
                # 16.3 从整图 latent 切出当前窗口的所有条件
                #
                # crop_latents:
                #   [N, 4, 64, 64]
                #
                # image/mask 都已整图 VAE encode，只在 latent 空间切片。
                # center map 与窗口坐标相关，因此当前步重新生成。
                # ----------------------------------------------------
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
                mask_crop_latents = self.crop_latents_from_global(
                    global_latents=mask_latents_full,
                    windows=windows,
                    latent_crop_size=latent_crop_size,
                )

                center_maps = torch.cat(
                    [
                        self.prepare_center_maps_for_pixel_window(
                            pixel_top=win["pixel_top"],
                            pixel_left=win["pixel_left"],
                            crop_size=crop_size,
                            image_h=pixel_h,
                            image_w=pixel_w,
                            latent_crop_size=latent_crop_size,
                            dtype=dtype,
                            device=device,
                        )
                        for win in windows
                    ],
                    dim=0,
                )

                if self.do_classifier_free_guidance:
                    image_crop_latents = self.make_cfg_condition(
                        image_crop_latents
                    )
                    mask_crop_latents = self.make_cfg_condition(
                        mask_crop_latents
                    )
                    center_maps = self.make_cfg_condition(center_maps)

                # ----------------------------------------------------
                # 16.3 CFG 扩展
                #
                # latent_model_input:
                #   [3N, 4, 64, 64]
                # ----------------------------------------------------
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

                # ----------------------------------------------------
                # 16.4 分批跑 UNet
                #
                # 注意：
                #   这里分批的是 3N 维度。
                #   所以所有条件 tensor 都必须已经扩展到 3N。
                # ----------------------------------------------------
                crop_noise_pred = self.run_conditioned_unet_in_batches(
                    latent_model_input=latent_model_input,
                    timestep=t,
                    image_condition=image_crop_latents,
                    mask_condition=mask_crop_latents,
                    full_image_condition=original_latents_full_per_crop,
                    center_condition=center_maps,
                    prompt_embeds=crop_prompt_embeds,
                    process_batch_size=crop_process_batch_size,
                    cross_attention_kwargs=cross_attention_kwargs,
                )

                # ----------------------------------------------------
                # 16.4 InstructPix2Pix CFG
                #
                # crop_noise_pred:
                #   [3N, 4, 64, 64]
                #
                # 合成后：
                #   [N, 4, 64, 64]
                # ----------------------------------------------------
                if self.do_classifier_free_guidance:
                    noise_pred_text, noise_pred_image, noise_pred_uncond = crop_noise_pred.chunk(3)

                    crop_noise_pred = (
                        noise_pred_uncond
                        + self.guidance_scale * (noise_pred_text - noise_pred_image)
                        + self.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                    )

                # 有效 crop 内仍会计算完整 512x512 UNet 输出，以便利用已知
                # 像素作为条件上下文；但 mask-aware 路径只让修复区域参与
                # 后续一致性、融合和 scheduler 更新。
                crop_update_masks = self.crop_latents_from_global(
                    global_latents=routing_update_mask_latent,
                    windows=windows,
                    latent_crop_size=latent_crop_size,
                )

                # ----------------------------------------------------
                # 16.5 计算当前 timestep 的动态可信度
                #
                # mask 边缘在早期就保持高可信；可信前沿随后以“前快后慢”
                # 的速度向中心推进。它同时决定：何时启用局部预测的一致性
                # /时间稳定性，以及深中心何时从全图结构预测切换到局部细节。
                # ----------------------------------------------------
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

                # ----------------------------------------------------
                # 16.6 先用静态可信度得到基准融合预测
                #
                # 基础权重只由当前 crop 内可见的真实上下文决定：
                #
                #   可见真实区域越多、离当前位置越近 -> 权重越高
                #   看不见真实区域                     -> 仅保留最小权重
                #
                # 不再因像素落在 crop 中心或边缘而额外加、降权。
                # 后期仍保留一半上下文影响；一致性和时间稳定性在后面
                # 逐步加入，而不是让“crop 中心”取代真实上下文。
                # ----------------------------------------------------
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

                # ----------------------------------------------------
                # 16.7 预测一致性：偏离基准融合预测的 crop 降权
                # ----------------------------------------------------
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

                # ----------------------------------------------------
                # 16.8 时间稳定性：以相邻 timestep 的 x0 预测为参照
                #
                # 不直接比较不同 timestep 的 noise_pred，避免不同噪声
                # 尺度导致的误判。scheduler 不支持 x0 推导时自动跳过。
                # ----------------------------------------------------
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

                # ----------------------------------------------------
                # 16.9 按多维可信度融合噪声预测
                # ----------------------------------------------------
                global_noise_pred = self.fuse_crop_tensors_to_global(
                    crop_tensors=crop_noise_pred,
                    crop_weights=crop_confidence_weights,
                    windows=windows,
                    latent_h=latent_h,
                    latent_w=latent_w,
                    latent_crop_size=latent_crop_size,
                )

                # ----------------------------------------------------
                # 16.10 按距 mask 边缘的传播进度混合全图/局部预测
                #
                # global_generation_confidence:
                #   mask 边缘早期接近 1 -> 相信局部真实上下文
                #   mask 深中心早期接近 0 -> 相信全图结构预测
                #
                # 因此全图分支先把道路和边线的整体信息送进中心，随后
                # 局部随机窗口逐渐接手高频纹理。
                # ----------------------------------------------------
                if global_context_enabled:
                    coarse_noise_pred_full = F.interpolate(
                        coarse_noise_pred,
                        size=(latent_h, latent_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    coarse_blend_weight = global_context_strength * (
                        1.0 - global_generation_confidence
                    )
                    global_noise_pred = (
                        (1.0 - coarse_blend_weight) * global_noise_pred
                        + coarse_blend_weight * coarse_noise_pred_full
                    )

                if crop_pred_original_sample is not None:
                    previous_global_pred_original_sample = (
                        self.predict_original_sample(
                            sample=global_latents,
                            model_output=global_noise_pred,
                            timestep=t,
                        )
                    )

                # ----------------------------------------------------
                # 16.11 对整张图只执行一次 scheduler.step
                #
                # 此时 model_output 与 sample 都是整图 latent，因此
                # scheduler 的状态和随机项也只在整图尺度更新一次。
                # ----------------------------------------------------
                global_latents = self.scheduler.step(
                    global_noise_pred,
                    t,
                    global_latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )[0]


                    # ------------------------------------------------------------
                    # 用输入图在下一时间步的加噪 latent 覆盖非修复区域
                    #
                    # scheduler.step 后：
                    #     global_latents 已经从 x_t 更新到下一状态
                    #
                    # 因此输入图也应加噪到下一个 timestep，而不是当前 t。
                    #
                    # 白色区域 mask=1：
                    #     保留生成 latent
                    #
                    # 黑色区域 mask=0：
                    #     使用输入图对应噪声级别的 latent
                    # ------------------------------------------------------------
                if inpaint_mask is not None:
                    if step_index < len(timesteps) - 1:
                        next_t = timesteps[step_index + 1]
                        # add_noise 要求 timestep 至少是一维，
                        # 对当前单张整图构造 [1]。
                        if torch.is_tensor(next_t):
                            next_t_batch = next_t.reshape(1).to(
                                device=original_latents_full.device
                            )
                        else:
                            next_t_batch = torch.tensor(
                                [next_t],
                                device=original_latents_full.device,
                                dtype=torch.long,
                            )
                        input_latents_noisy = self.scheduler.add_noise(
                            original_samples=original_latents_full,
                            noise=global_noise,
                            timesteps=next_t_batch,
                        )
                    else:
                        # 最后一步去噪完成后，非修复区域直接恢复干净输入 latent。
                        input_latents_noisy = original_latents_full
                    global_latents = (
                        state_inpaint_mask_latent * global_latents
                        + (
                            1.0
                            - state_inpaint_mask_latent
                        )
                        * input_latents_noisy
                    )

                # ----------------------------------------------------
                # 16.12 更新全图低分辨率 latent，并固定其非修复区域。
                # 它与局部分支使用独立 scheduler 状态，但始终处于同一
                # timestep；下一步会继续为 mask 深中心提供全图结构信息。
                # ----------------------------------------------------
                if global_context_enabled:
                    coarse_latents = coarse_scheduler.step(
                        coarse_noise_pred,
                        t,
                        coarse_latents,
                        **coarse_extra_step_kwargs,
                        return_dict=False,
                    )[0]
                    if step_index < len(timesteps) - 1:
                        coarse_input_latents_noisy = (
                            coarse_scheduler.add_noise(
                                original_samples=coarse_original_image_latents,
                                noise=coarse_noise,
                                timesteps=next_t_batch,
                            )
                        )
                    else:
                        coarse_input_latents_noisy = (
                            coarse_original_image_latents
                        )
                    coarse_latents = (
                        coarse_inpaint_mask_latent * coarse_latents
                        + (1.0 - coarse_inpaint_mask_latent)
                        * coarse_input_latents_noisy
                    )

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

                    global_latents = callback_outputs.pop("global_latents", global_latents)

                if step_index == len(timesteps) - 1 or (
                    (step_index + 1) > num_warmup_steps
                    and (step_index + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                    if callback is not None and callback_steps is not None and step_index % callback_steps == 0:
                        callback(step_index, t, global_latents)

        # ------------------------------------------------------------
        # 17. 最后只 decode 一次整图 latent
        #
        # 旧版：
        #   decode 每个 crop latent
        #   stitch crop image
        #
        # 新版：
        #   decode global_latents
        #
        # output_image_tensor:
        #   [1, 3, H, W]
        # ------------------------------------------------------------
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

    # ================================================================
    # mask 预处理
    # ================================================================
    def preprocess_mask_to_tensor(
        self,
        mask,
        height: int,
        width: int,
        device,
        dtype,
    ):
        """
        把 mask 转成与训练代码一致的格式：

            shape: [1, 1, H, W]
            range: [0, 1]

        不进行 [-1,1] 归一化。
        [-1,1] 转换放在 VAE 编码前完成。
        """

        if isinstance(mask, PIL.Image.Image):
            mask = mask.convert("L")

            mask = mask.resize(
                (width, height),
                resample=PIL.Image.Resampling.NEAREST,
            )

            import numpy as np

            mask_array = np.asarray(
                mask,
                dtype=np.float32,
            )

            mask_tensor = torch.from_numpy(
                mask_array
            )[None, None]

            mask_tensor = mask_tensor / 255.0

        elif torch.is_tensor(mask):
            mask_tensor = mask.detach().float()

            if mask_tensor.ndim == 2:
                # [H,W] -> [1,1,H,W]
                mask_tensor = mask_tensor[None, None]

            elif mask_tensor.ndim == 3:
                # 支持 [C,H,W]
                if mask_tensor.shape[0] in (1, 3, 4):
                    mask_tensor = mask_tensor[None]

                # 支持 [H,W,C]
                elif mask_tensor.shape[-1] in (1, 3, 4):
                    mask_tensor = mask_tensor.permute(
                        2,
                        0,
                        1,
                    )[None]

                else:
                    raise ValueError(
                        f"无法判断三维 mask 的通道位置：{mask_tensor.shape}"
                    )

            elif mask_tensor.ndim != 4:
                raise ValueError(
                    f"mask Tensor 应为 2/3/4 维，实际为 {mask_tensor.shape}"
                )

            if mask_tensor.shape[0] != 1:
                raise ValueError(
                    "当前推理代码只支持单张 mask，"
                    f"实际 batch={mask_tensor.shape[0]}"
                )

            # RGB/RGBA mask 只取一个通道
            if mask_tensor.shape[1] != 1:
                mask_tensor = mask_tensor[:, :1]

            if mask_tensor.max().item() > 1.0:
                mask_tensor = mask_tensor / 255.0

            mask_tensor = F.interpolate(
                mask_tensor,
                size=(height, width),
                mode="nearest",
            )

        else:
            raise TypeError(
                "mask 目前只支持 PIL.Image 或 torch.Tensor，"
                f"实际类型为 {type(mask)}"
            )

        mask_tensor = mask_tensor.clamp(
            0.0,
            1.0,
        )

        return mask_tensor.to(
            device=device,
            dtype=dtype,
        )
    # ================================================================
    # 整图 mask -> 整图 latent
    # ================================================================
    def encode_mask_to_latents(
        self,
        mask_images: torch.Tensor,
        dtype,
        device,
        encode_batch_size: int = 8,
    ):
        """
        将整图 mask VAE encode 一次。

        输入：
            mask_images:
                [N, 1, H, W]
                range [0, 1]

        处理：
            1. 重复成三通道
            2. 从 [0,1] 映射到 [-1,1]
            3. 使用 VAE 编码
            4. 使用 latent distribution 的 mode
            5. 乘一次 vae.config.scaling_factor

        输出：
            mask_latents:
                [N, 4, H/8, W/8]
        """

        if mask_images.ndim != 4:
            raise ValueError(
                f"mask_images 应为 [N,1,H,W]，实际为 {mask_images.shape}"
            )

        if mask_images.shape[1] != 1:
            raise ValueError(
                f"mask_images 应为单通道，实际通道数为 {mask_images.shape[1]}"
            )

        if encode_batch_size <= 0:
            raise ValueError(
                f"encode_batch_size 必须大于 0，实际为 {encode_batch_size}"
            )

        mask_images = mask_images.to(
            device=device,
            dtype=dtype,
        )

        # 避免输入 mask 因格式问题超过 [0,1]
        mask_images = mask_images.clamp(0.0, 1.0)

        mask_latent_chunks = []

        for start in range(0, mask_images.shape[0], encode_batch_size):
            end = min(
                start + encode_batch_size,
                mask_images.shape[0],
            )

            mask_chunk = mask_images[start:end]

            # [B,1,H,W] -> [B,3,H,W]
            mask_3ch = mask_chunk.repeat(
                1,
                3,
                1,
                1,
            )

            # [0,1] -> [-1,1]
            mask_3ch = mask_3ch * 2.0 - 1.0

            mask_latents = (
                self.vae.encode(mask_3ch)
                .latent_dist.mode()
            )

            # 与训练保持一致，只乘一次
            mask_latents = (
                mask_latents
                * self.vae.config.scaling_factor
            )

            mask_latent_chunks.append(mask_latents)

        return torch.cat(
            mask_latent_chunks,
            dim=0,
        )


    # ================================================================
    # 整图 image -> 整图 latent
    # ================================================================
    def encode_image_full_to_latents(
        self,
        image: torch.Tensor,
        dtype,
        device,
    ):
        """
        整张图一次性 VAE encode。

        输入：
            image: [1, 3, H, W], range [-1, 1]

        输出：
            latents: [1, 4, H/8, W/8]

        注意：
            这里乘 vae.config.scaling_factor，
            和训练代码保持一致。
        """

        image = image.to(device=device, dtype=dtype)

        latents = retrieve_latents(
            self.vae.encode(image),
            sample_mode="argmax",
        )

        latents = latents * self.vae.config.scaling_factor

        return latents

    # ================================================================
    # 初始化整图 noisy latent
    # ================================================================
    def prepare_global_latents(
        self,
        latent_h: int,
        latent_w: int,
        num_channels_latents: int,
        dtype,
        device,
        generator,
    ):
        """
        初始化整图 noisy latent。

        输出：
            [1, 4, latent_h, latent_w]

        之后每一步都从这张整图 latent 上切 crop。
        """

        shape = (
            1,
            num_channels_latents,
            latent_h,
            latent_w,
        )

        noisy = randn_tensor(
            shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )

        latents = noisy * self.scheduler.init_noise_sigma

        return latents, noisy

    # ================================================================
    # 全图低分辨率结构分支
    # ================================================================
    def get_global_context_latent_size(
        self,
        latent_h: int,
        latent_w: int,
        max_latent_side: int,
    ):
        """返回与训练 crop 一致的方形全图结构网格尺寸。"""

        if max_latent_side <= 0:
            raise ValueError("max_latent_side 必须大于 0。")

        downsample_alignment = 2 ** (
            len(self.unet.config.block_out_channels) - 1
        )
        if max_latent_side < downsample_alignment:
            raise ValueError(
                "全图分支的 latent 尺寸过小，"
                f"至少需要 {downsample_alignment}。"
            )

        if max_latent_side % downsample_alignment != 0:
            raise ValueError(
                "全图结构分支尺寸必须满足 UNet 下采样对齐，"
                f"实际为 {max_latent_side}，对齐要求为 "
                f"{downsample_alignment}。"
            )

        # ContentAwarePositionGatedConvIn 的 global_projector 输出固定为
        # 训练 crop 尺寸；全图分支必须使用相同方形尺寸，避免条件特征与
        # noisy latent 的空间尺寸不一致。
        return max_latent_side, max_latent_side

    def normalize_noise_to_unit_variance(self, noise: torch.Tensor):
        """补偿 area 下采样造成的噪声方差下降。"""

        noise_std = noise.flatten(start_dim=1).std(
            dim=1,
            unbiased=False,
        ).view(-1, 1, 1, 1)
        return noise / noise_std.clamp(min=1e-6)

    def run_conditioned_unet_in_batches(
        self,
        latent_model_input: torch.Tensor,
        timestep,
        image_condition: torch.Tensor,
        mask_condition: torch.Tensor,
        full_image_condition: torch.Tensor,
        center_condition: torch.Tensor,
        prompt_embeds: torch.Tensor,
        process_batch_size: int,
        cross_attention_kwargs: dict[str, Any] | None,
    ):
        """以相同的条件注入方式运行局部或全图 UNet 分支。"""

        total_batch = latent_model_input.shape[0]
        condition_tensors = {
            "image_condition": image_condition,
            "mask_condition": mask_condition,
            "full_image_condition": full_image_condition,
            "center_condition": center_condition,
            "prompt_embeds": prompt_embeds,
        }
        for name, tensor in condition_tensors.items():
            if tensor.shape[0] != total_batch:
                raise ValueError(
                    f"{name} 的 batch={tensor.shape[0]}，"
                    f"但 latent batch={total_batch}。"
                )

        noise_pred_chunks = []
        for start in range(0, total_batch, process_batch_size):
            end = min(start + process_batch_size, total_batch)
            self.unet.conv_in.set_condition_context(
                original_crop_latents=image_condition[start:end],
                mask_embeds=mask_condition[start:end],
                original_latents_full=full_image_condition[start:end],
                center_maps=center_condition[start:end],
            )

            try:
                noise_pred_chunk = self.unet(
                    latent_model_input[start:end],
                    timestep,
                    encoder_hidden_states=prompt_embeds[start:end],
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]
            finally:
                self.unet.conv_in.clear_condition_context()

            noise_pred_chunks.append(noise_pred_chunk)

        return torch.cat(noise_pred_chunks, dim=0)

    # ================================================================
    # 从整图 latent 上切 crop
    # ================================================================
    def crop_latents_from_global(
        self,
        global_latents: torch.Tensor,
        windows: list[dict],
        latent_crop_size: int,
    ):
        """
        从整图 latent 上，根据像素窗口切出 latent crop。

        global_latents:
            [1, 4, H_lat, W_lat]

        return:
            [N, 4, latent_crop_size, latent_crop_size]
        """

        crops = []

        for win in windows:
            latent_top = win["pixel_top"] // self.vae_scale_factor
            latent_left = win["pixel_left"] // self.vae_scale_factor

            crop = global_latents[
                :,
                :,
                latent_top : latent_top + latent_crop_size,
                latent_left : latent_left + latent_crop_size,
            ]

            crops.append(crop)

        return torch.cat(crops, dim=0)

    # ================================================================
    # 从整图 mask 上切 crop
    # ================================================================
    def crop_mask_from_global(
        self,
        mask_image: torch.Tensor,
        windows: list[dict],
        crop_size: int,
    ):
        """
        从整图 mask 上切出像素级 mask crop。

        mask_image:
            [1, 1, H, W]

        return:
            [N, 1, crop_size, crop_size]
        """

        crops = []

        for win in windows:
            top = win["pixel_top"]
            left = win["pixel_left"]

            crop = mask_image[
                :,
                :,
                top : top + crop_size,
                left : left + crop_size,
            ]

            crops.append(crop)

        return torch.cat(crops, dim=0)

    # ================================================================
    # crop tensor 融合到整图 tensor
    # ================================================================
    def fuse_crop_tensors_to_global(
        self,
        crop_tensors: torch.Tensor,
        windows: list[dict],
        latent_h: int,
        latent_w: int,
        latent_crop_size: int,
        crop_weights: torch.Tensor | None = None,
        return_weight_sum: bool = False,
    ):
        """
        把每个 crop tensor 写回整图 tensor。

        对重叠区域做逐像素加权平均。

        crop_tensors:
            [N, C, latent_crop_size, latent_crop_size]

        crop_weights:
            None 或 [N, 1, latent_crop_size, latent_crop_size]。
            None 时等价于简单平均。

        return:
            默认返回 [1, C, latent_h, latent_w]。
            return_weight_sum=True 时额外返回 [1, 1, latent_h, latent_w]
            的权重和。
        """

        num_crops, channels, _, _ = crop_tensors.shape
        device = crop_tensors.device
        dtype = crop_tensors.dtype

        if crop_weights is None:
            crop_weights = torch.ones(
                num_crops,
                1,
                latent_crop_size,
                latent_crop_size,
                device=device,
                dtype=dtype,
            )
        elif crop_weights.shape != (
            num_crops,
            1,
            latent_crop_size,
            latent_crop_size,
        ):
            raise ValueError(
                "crop_weights 形状应为 "
                f"[{num_crops}, 1, {latent_crop_size}, {latent_crop_size}]，"
                f"实际为 {tuple(crop_weights.shape)}。"
            )
        elif crop_weights.device != device:
            raise ValueError("crop_weights 与 crop_tensors 必须在同一 device 上。")
        elif crop_weights.dtype != dtype:
            crop_weights = crop_weights.to(dtype=dtype)

        global_sum = torch.zeros(
            1,
            channels,
            latent_h,
            latent_w,
            device=device,
            dtype=dtype,
        )

        global_weight_sum = torch.zeros(
            1,
            1,
            latent_h,
            latent_w,
            device=device,
            dtype=dtype,
        )

        for idx, win in enumerate(windows):
            latent_top = win["pixel_top"] // self.vae_scale_factor
            latent_left = win["pixel_left"] // self.vae_scale_factor

            global_sum[
                :,
                :,
                latent_top : latent_top + latent_crop_size,
                latent_left : latent_left + latent_crop_size,
            ] += crop_tensors[idx : idx + 1] * crop_weights[idx : idx + 1]

            global_weight_sum[
                :,
                :,
                latent_top : latent_top + latent_crop_size,
                latent_left : latent_left + latent_crop_size,
            ] += crop_weights[idx : idx + 1]

        global_tensor = global_sum / global_weight_sum.clamp(min=1e-6)

        if return_weight_sum:
            return global_tensor, global_weight_sum

        return global_tensor

    def masked_crop_channel_mse(
        self,
        values: torch.Tensor,
        spatial_mask: torch.Tensor,
    ):
        """计算仅在修复区域有效的逐像素通道平均 MSE。

        ``values`` 是 [N,C,H,W]，而 ``spatial_mask`` 是 [N,1,H,W]。
        这样 mask-aware 路径中的一致性/时间稳定性不会被已知背景的预测
        差异主导。
        """

        if values.ndim != 4 or spatial_mask.ndim != 4:
            raise ValueError("values 与 spatial_mask 都必须是四维 Tensor。")
        if (
            values.shape[0] != spatial_mask.shape[0]
            or values.shape[-2:] != spatial_mask.shape[-2:]
            or spatial_mask.shape[1] != 1
        ):
            raise ValueError(
                "values 应为 [N,C,H,W]、spatial_mask 应为 [N,1,H,W]，"
                f"实际为 {tuple(values.shape)} 与 {tuple(spatial_mask.shape)}。"
            )

        # 保持 [N,1,H,W] 形状，以便 residuals_to_confidence_weights 能够
        # 针对同一空间位置的重叠预测计算自适应尺度。mask 外值会被置零，
        # 并由调用方的 valid_mask 排除，不参与统计或融合。
        return values.square().mean(dim=1, keepdim=True) * spatial_mask

    # ================================================================
    # 多维融合可信度：整图静态部分
    # ================================================================
    def prepare_global_confidence_maps(
        self,
        inpaint_mask: torch.Tensor,
        latent_h: int,
        latent_w: int,
    ):
        """
        创建不随 timestep 变化的整图可信度图。

        返回：
            repair_mask_latent:
                [1, 1, H, W]，修复区为 1。
            repair_distance_latent:
                [1, 1, H, W]，修复区到最近非修复区的距离。
        """

        if inpaint_mask.shape[0] != 1 or inpaint_mask.shape[1] != 1:
            raise ValueError(
                "inpaint_mask 应为 [1, 1, H, W]，"
                f"实际为 {tuple(inpaint_mask.shape)}。"
            )

        device = inpaint_mask.device
        dtype = inpaint_mask.dtype

        repair_mask_pixels = (inpaint_mask >= 0.5).to(dtype=dtype)
        repair_mask_latent = F.interpolate(
            repair_mask_pixels,
            size=(latent_h, latent_w),
            mode="area",
        ).clamp(0.0, 1.0)

        repair_mask_np = (
            repair_mask_pixels[0, 0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        if np.all(repair_mask_np == 1):
            # 没有已知区域时无法定义空间传播方向，退化为仅随时间增长
            # 的可信度；一致性和时间稳定性权重仍会继续工作。
            repair_distance_pixels = np.zeros_like(
                repair_mask_np,
                dtype=np.float32,
            )
        else:
            repair_distance_pixels = cv2.distanceTransform(
                repair_mask_np,
                cv2.DIST_L2,
                5,
            )
        repair_distance_latent = F.interpolate(
            torch.from_numpy(repair_distance_pixels)[None, None],
            size=(latent_h, latent_w),
            mode="bilinear",
            align_corners=False,
        ).to(device=device, dtype=dtype)

        return repair_mask_latent, repair_distance_latent

    def prepare_crop_context_support(
        self,
        repair_mask_latent: torch.Tensor,
        windows: list[dict],
        latent_crop_size: int,
        context_radius_pixels: int,
    ):
        """
        基于当前窗口，计算每个 crop 可见的非修复区比例。

        repair_mask_latent 是整图坐标系下的静态图；窗口随机移动时，只需
        重新从它切取 crop 并做局部池化，无需重新 VAE encode mask。
        """

        known_latent = 1.0 - repair_mask_latent
        known_crops = self.crop_latents_from_global(
            global_latents=known_latent,
            windows=windows,
            latent_crop_size=latent_crop_size,
        )

        context_radius_latent = max(
            int(round(context_radius_pixels / self.vae_scale_factor)),
            0,
        )
        if context_radius_latent == 0:
            return known_crops

        kernel_size = 2 * context_radius_latent + 1
        return F.avg_pool2d(
            known_crops,
            kernel_size=kernel_size,
            stride=1,
            padding=context_radius_latent,
            count_include_pad=False,
        )

    # ================================================================
    # 多维融合可信度：由外向内的时间进度
    # ================================================================
    def build_generation_confidence_map(
        self,
        repair_mask_latent: torch.Tensor,
        repair_distance_latent: torch.Tensor,
        step_index: int,
        num_steps: int,
        propagation_gamma: float,
    ):
        """
        修复区域的动态可信度图。

        mask 边缘从第一个 timestep 起就保持高可信；随后可信前沿按
        “前快后慢”的曲线快速向修复区中心推进。非修复区恒为 1。
        """

        if num_steps <= 0:
            raise ValueError("num_steps 必须大于 0。")

        if propagation_gamma <= 0.0:
            raise ValueError("propagation_gamma 必须大于 0。")

        progress = (
            1.0
            if num_steps == 1
            else step_index / (num_steps - 1)
        )
        max_distance = repair_distance_latent.max()
        transition_width = torch.clamp(max_distance * 0.05, min=1.0)
        if max_distance <= 1e-6:
            repair_confidence = torch.full_like(
                repair_mask_latent,
                progress,
            )
        else:
            # t=0 时，已知边界内侧约 3 个过渡带宽已可用；gamma > 1
            # 使前沿在早期迅速推进，避免 1024 级大 mask 的中心直到
            # 最后几步才开始得到可靠结构信息。
            accelerated_progress = 1.0 - (1.0 - progress) ** (
                propagation_gamma
            )
            initial_radius = 3.0 * transition_width
            final_radius = max_distance + 3.0 * transition_width
            propagation_radius = initial_radius + (
                final_radius - initial_radius
            ) * accelerated_progress
            repair_confidence = torch.sigmoid(
                (propagation_radius - repair_distance_latent)
                / transition_width
            )

        return (
            1.0 - repair_mask_latent
            + repair_mask_latent * repair_confidence
        )

    # ================================================================
    # 多维融合可信度：预测残差转权重
    # ================================================================
    def residuals_to_confidence_weights(
        self,
        residuals: torch.Tensor,
        valid_mask: torch.Tensor,
        min_weight: float,
    ):
        """
        用重叠区域的中位残差作自适应尺度，将预测残差映射为 [min_weight, 1]
        的相对可信度。没有重叠的位置固定为 1，不影响单 crop 区域。
        """

        if residuals.shape != valid_mask.shape:
            raise ValueError(
                "residuals 与 valid_mask 形状必须一致："
                f"{tuple(residuals.shape)} vs {tuple(valid_mask.shape)}。"
            )

        valid_residuals = residuals[valid_mask]
        if valid_residuals.numel() == 0:
            return torch.ones_like(residuals)

        scale = valid_residuals.median()
        if scale <= 1e-6:
            return torch.ones_like(residuals)

        weights = torch.exp(-0.5 * (residuals / scale).square())
        weights = weights.clamp(min=min_weight, max=1.0)

        return torch.where(valid_mask, weights, torch.ones_like(weights))

    # ================================================================
    # 多维融合可信度：将模型输出转为 x0 预测
    # ================================================================
    def predict_original_sample(
        self,
        sample: torch.Tensor,
        model_output: torch.Tensor,
        timestep,
    ):
        """
        将 epsilon/sample/v_prediction 转为 x0 预测，供跨 timestep 的稳定性
        比较使用。对于不暴露 alphas_cumprod 的 scheduler 返回 None。
        """

        alphas_cumprod = getattr(self.scheduler, "alphas_cumprod", None)
        if alphas_cumprod is None:
            return None

        if torch.is_tensor(timestep):
            if timestep.numel() != 1 or timestep.is_floating_point():
                return None
            timestep_index = int(timestep.item())
        else:
            timestep_index = int(timestep)

        if not 0 <= timestep_index < len(alphas_cumprod):
            return None

        alpha_prod_t = alphas_cumprod[timestep_index].to(
            device=sample.device,
            dtype=sample.dtype,
        )
        beta_prod_t = 1.0 - alpha_prod_t
        sqrt_alpha_prod_t = torch.sqrt(alpha_prod_t)
        sqrt_beta_prod_t = torch.sqrt(beta_prod_t)

        scheduler_config = getattr(self.scheduler, "config", None)
        prediction_type = getattr(
            scheduler_config,
            "prediction_type",
            "epsilon",
        )

        if prediction_type == "epsilon":
            return (
                sample - sqrt_beta_prod_t * model_output
            ) / sqrt_alpha_prod_t

        if prediction_type == "sample":
            return model_output

        if prediction_type == "v_prediction":
            return (
                sqrt_alpha_prod_t * sample
                - sqrt_beta_prod_t * model_output
            )

        return None

    # ================================================================
    # 生成 CFG 条件
    # ================================================================
    def make_cfg_condition(self, tensor: torch.Tensor):
        """
        把普通条件 [N, C, H, W] 扩展成 CFG 条件：

            [condition, condition, zero]

        对应 InstructPix2Pix 的三路：
            text branch
            image branch
            uncond branch
        """

        zero_tensor = torch.zeros_like(tensor)

        return torch.cat(
            [
                tensor,
                tensor,
                zero_tensor,
            ],
            dim=0,
        )

    # ================================================================
    # prompt CFG tensor 复制到每个 crop
    # ================================================================
    def repeat_cfg_tensor_for_crops(
        self,
        tensor: torch.Tensor,
        num_crops: int,
        do_classifier_free_guidance: bool,
    ):
        """
        原始 prompt_embeds：

            CFG:
                [3, seq, dim]

        复制后：

            [3 * num_crops, seq, dim]

        顺序必须保持：
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
    # center map
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
    ):
        """
        生成当前 crop 的位置条件。

        和训练代码里的 centers 对齐：

            cx = crop中心点x / 整图宽度
            cy = crop中心点y / 整图高度

        输出：
            [1, 2, latent_crop_size, latent_crop_size]
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

        return center_maps

    # ================================================================
    # 随机窗口：基础网格、受限随机相位与完整覆盖
    # ================================================================
    def get_randomizable_window_starts(
        self,
        image_h: int,
        image_w: int,
        crop_size: int,
        target_overlap: int,
        minimum_overlap: int,
    ):
        """
        创建固定边界、内部可随机平移的基础网格。

        当某个方向原本只有首尾两个窗口时，尽可能插入一个中间窗口，
        使该方向也能在每个 timestep 产生不同视角。
        """

        tops = self.get_adaptive_starts(
            length=image_h,
            crop=crop_size,
            target_overlap=target_overlap,
            alignment=self.vae_scale_factor,
        )
        lefts = self.get_adaptive_starts(
            length=image_w,
            crop=crop_size,
            target_overlap=target_overlap,
            alignment=self.vae_scale_factor,
        )

        return (
            self.ensure_randomizable_starts(
                starts=tops,
                crop_size=crop_size,
                minimum_overlap=minimum_overlap,
            ),
            self.ensure_randomizable_starts(
                starts=lefts,
                crop_size=crop_size,
                minimum_overlap=minimum_overlap,
            ),
        )

    def ensure_randomizable_starts(
        self,
        starts: list[int],
        crop_size: int,
        minimum_overlap: int,
    ):
        """为仅有两个边界窗口的方向补充一个可移动的中间窗口。"""

        if len(starts) != 2:
            return starts

        max_start = starts[-1]
        if max_start == 0:
            return starts

        alignment = self.vae_scale_factor
        middle = round(max_start / 2 / alignment) * alignment
        max_stride = crop_size - minimum_overlap

        if (
            middle <= 0
            or middle >= max_start
            or middle > max_stride
            or max_start - middle > max_stride
        ):
            return starts

        return [0, int(middle), max_start]

    def make_pixel_windows(
        self,
        tops: list[int],
        lefts: list[int],
    ):
        """由纵横起点生成按行优先排序的像素窗口。"""

        return [
            {
                "pixel_top": top,
                "pixel_left": left,
            }
            for top in tops
            for left in lefts
        ]

    def create_window_generator(
        self,
        generator: torch.Generator | list[torch.Generator] | None,
        window_seed: int | None,
        seed_offset: int = 0,
    ):
        """
        建立独立的 CPU 随机数发生器，避免窗口随机性消耗 diffusion 的
        noise generator，从而保持噪声采样与窗口采样相互独立。
        """

        if window_seed is None:
            source_generator = (
                generator[0]
                if isinstance(generator, list) and len(generator) > 0
                else generator
            )
            if isinstance(source_generator, torch.Generator):
                window_seed = source_generator.initial_seed()
            else:
                window_seed = torch.initial_seed()

        window_generator = torch.Generator(device="cpu")
        window_generator.manual_seed(
            (int(window_seed) + seed_offset) % (2**63 - 1)
        )

        return window_generator

    def sample_jittered_pixel_windows(
        self,
        base_tops: list[int],
        base_lefts: list[int],
        crop_size: int,
        minimum_overlap: int,
        jitter_pixels: int,
        generator: torch.Generator,
    ):
        """
        对 x/y 内部窗口采样独立的共享随机相位。

        同一方向的所有内部窗口采用同一个 offset，既能让每一步窗口
        位置变化，又不会破坏内部窗口间距。首尾窗口固定在边界，且根据
        minimum_overlap 限制 offset，保证不会产生覆盖空洞。
        """

        tops = self.sample_jittered_starts(
            base_starts=base_tops,
            crop_size=crop_size,
            minimum_overlap=minimum_overlap,
            jitter_pixels=jitter_pixels,
            generator=generator,
        )
        lefts = self.sample_jittered_starts(
            base_starts=base_lefts,
            crop_size=crop_size,
            minimum_overlap=minimum_overlap,
            jitter_pixels=jitter_pixels,
            generator=generator,
        )

        return self.make_pixel_windows(tops, lefts)

    def sample_jittered_starts(
        self,
        base_starts: list[int],
        crop_size: int,
        minimum_overlap: int,
        jitter_pixels: int,
        generator: torch.Generator,
    ):
        """在不破坏覆盖和 VAE 对齐的前提下，随机偏移内部窗口。"""

        if len(base_starts) <= 2 or jitter_pixels == 0:
            return list(base_starts)

        alignment = self.vae_scale_factor
        max_start = base_starts[-1]
        max_stride = crop_size - minimum_overlap

        lower_offset = max(
            -jitter_pixels,
            max_start - max_stride - base_starts[-2],
            alignment - base_starts[1],
        )
        upper_offset = min(
            jitter_pixels,
            max_stride - base_starts[1],
            max_start - alignment - base_starts[-2],
        )

        lower_offset = (
            (lower_offset + alignment - 1) // alignment
        ) * alignment
        upper_offset = (upper_offset // alignment) * alignment

        if lower_offset > upper_offset:
            return list(base_starts)

        lower_units = lower_offset // alignment
        upper_units = upper_offset // alignment
        offset = int(
            torch.randint(
                lower_units,
                upper_units + 1,
                (1,),
                generator=generator,
            ).item()
        ) * alignment

        return [
            base_starts[0],
            *[start + offset for start in base_starts[1:-1]],
            base_starts[-1],
        ]

    # ================================================================
    # mask-aware 窗口路由
    # ================================================================
    def build_mask_aware_base_windows(
        self,
        repair_mask: np.ndarray,
        base_tops: list[int],
        base_lefts: list[int],
        crop_size: int,
        min_known_fraction: float,
    ):
        """从规则网格筛选与修复区相关的基础窗口。

        所有保留窗口都至少包含一个修复像素。包含足够已知像素的窗口被
        标记为边界锚点；后续随机抖动时仍必须保持这些上下文。完全位于
        大 mask 内部的窗口作为桥接窗口保留，保证深中心也能被覆盖。
        """

        if repair_mask.ndim != 2:
            raise ValueError(
                f"repair_mask 必须是 [H,W]，实际为 {repair_mask.shape}。"
            )
        if not 0.0 <= min_known_fraction <= 1.0:
            raise ValueError("min_known_fraction 必须在 [0, 1] 内。")

        repair_mask = repair_mask.astype(bool, copy=False)
        windows = []
        for top in base_tops:
            for left in base_lefts:
                mask_crop = repair_mask[
                    top : top + crop_size,
                    left : left + crop_size,
                ]
                if not np.any(mask_crop):
                    continue
                known_fraction = 1.0 - float(mask_crop.mean())
                windows.append(
                    {
                        "pixel_top": int(top),
                        "pixel_left": int(left),
                        "requires_known_context": (
                            known_fraction >= min_known_fraction
                        ),
                    }
                )
        return windows

    def sample_mask_aware_pixel_windows(
        self,
        base_windows: list[dict],
        repair_mask: np.ndarray,
        image_h: int,
        image_w: int,
        crop_size: int,
        jitter_pixels: int,
        min_known_fraction: float,
        max_attempts: int,
        generator: torch.Generator,
    ):
        """随机抖动有效窗口，并补回抖动造成的修复区覆盖空洞。

        与常规整图滑窗不同，此函数不会产生 mask 外窗口。对于被标记为
        边界锚点的基础窗口，随机位置必须同时包含修复区和足够已知上下文；
        若找不到合法偏移，则保留其基础位置。
        """

        if not base_windows:
            raise ValueError("base_windows 不能为空。")
        if repair_mask.shape != (image_h, image_w):
            raise ValueError(
                "repair_mask 尺寸必须与图像一致："
                f"{repair_mask.shape} vs {(image_h, image_w)}。"
            )
        if max_attempts < 1:
            raise ValueError("max_attempts 必须大于等于 1。")

        repair_mask = repair_mask.astype(bool, copy=False)
        alignment = self.vae_scale_factor
        jitter_units = jitter_pixels // alignment
        selected_windows = []
        selected_positions: set[tuple[int, int]] = set()

        def is_valid(top: int, left: int, requires_known_context: bool) -> bool:
            mask_crop = repair_mask[
                top : top + crop_size,
                left : left + crop_size,
            ]
            if not np.any(mask_crop):
                return False
            if not requires_known_context:
                return True
            known_fraction = 1.0 - float(mask_crop.mean())
            return known_fraction >= min_known_fraction

        def append_window(top: int, left: int) -> None:
            key = (top, left)
            if key not in selected_positions:
                selected_positions.add(key)
                selected_windows.append(
                    {"pixel_top": int(top), "pixel_left": int(left)}
                )

        for base_window in base_windows:
            base_top = int(base_window["pixel_top"])
            base_left = int(base_window["pixel_left"])
            requires_known_context = bool(
                base_window["requires_known_context"]
            )
            selected_top, selected_left = base_top, base_left

            for _ in range(max_attempts):
                if jitter_units == 0:
                    break
                top_offset = int(
                    torch.randint(
                        -jitter_units,
                        jitter_units + 1,
                        (1,),
                        generator=generator,
                    ).item()
                ) * alignment
                left_offset = int(
                    torch.randint(
                        -jitter_units,
                        jitter_units + 1,
                        (1,),
                        generator=generator,
                    ).item()
                ) * alignment
                top = min(max(base_top + top_offset, 0), image_h - crop_size)
                left = min(max(base_left + left_offset, 0), image_w - crop_size)
                if is_valid(top, left, requires_known_context):
                    selected_top, selected_left = top, left
                    break

            # 基础窗口来自完整覆盖网格，且已经验证过含 mask，因此无论抖动
            # 是否成功都可以作为安全回退。
            append_window(selected_top, selected_left)

        # 独立抖动会让少量 mask 像素暂时落在所有窗口外。扫描并补入能够
        # 覆盖这些空洞的基础窗口，确保每个修复像素每步至少有一个预测。
        uncovered = repair_mask.copy()
        for window in selected_windows:
            top, left = window["pixel_top"], window["pixel_left"]
            uncovered[top : top + crop_size, left : left + crop_size] = False

        if np.any(uncovered):
            for base_window in base_windows:
                top = int(base_window["pixel_top"])
                left = int(base_window["pixel_left"])
                if np.any(
                    uncovered[
                        top : top + crop_size,
                        left : left + crop_size,
                    ]
                ):
                    append_window(top, left)
                    uncovered[
                        top : top + crop_size,
                        left : left + crop_size,
                    ] = False
                if not np.any(uncovered):
                    break

        if np.any(uncovered):
            raise RuntimeError("mask-aware 窗口未能覆盖全部修复像素。")

        return selected_windows

    # ================================================================
    # 像素空间窗口生成
    # ================================================================
    def get_adaptive_pixel_windows(
        self,
        image_h: int,
        image_w: int,
        crop_size: int,
        target_overlap: int,
    ):
        """
        生成像素空间 crop 窗口。

        虽然新版是在 latent 空间切 crop，
        但窗口仍然先用像素坐标描述。

        好处：
            1. crop_size / overlap 仍然直观，比如 512 / 128
            2. mask crop 仍然可以直接用同一个窗口
            3. latent crop 时只需要除以 vae_scale_factor
        """

        tops = self.get_adaptive_starts(
            length=image_h,
            crop=crop_size,
            target_overlap=target_overlap,
            alignment=self.vae_scale_factor,
        )

        lefts = self.get_adaptive_starts(
            length=image_w,
            crop=crop_size,
            target_overlap=target_overlap,
            alignment=self.vae_scale_factor,
        )

        return self.make_pixel_windows(tops, lefts)

    def get_adaptive_starts(
        self,
        length: int,
        crop: int,
        target_overlap: int,
        alignment: int = 1,
    ):
        """
        生成一个方向上的 crop 起点。

        目标：
            1. 第一块贴边
            2. 最后一块贴边
            3. 完整覆盖
            4. overlap 尽量接近 target_overlap
        """

        if length < crop:
            raise ValueError(f"length={length} 小于 crop={crop}。")

        if length == crop:
            return [0]

        target_stride = crop - target_overlap

        if target_stride <= 0:
            raise ValueError("target_overlap 太大。")

        if alignment <= 0:
            raise ValueError("alignment 必须大于 0。")

        if length % alignment != 0 or crop % alignment != 0:
            raise ValueError(
                "length 和 crop 必须能被 alignment 整除，"
                f"实际为 length={length}, crop={crop}, alignment={alignment}。"
            )

        max_start = length - crop

        num_intervals = (max_start + target_stride - 1) // target_stride
        num_intervals = max(num_intervals, 1)

        starts = []

        for i in range(num_intervals + 1):
            start = round(i * max_start / num_intervals)
            start = round(start / alignment) * alignment
            starts.append(int(start))

        starts = sorted(set(starts))

        if starts[0] != 0:
            starts.insert(0, 0)

        if starts[-1] != max_start:
            starts.append(max_start)

        return starts
