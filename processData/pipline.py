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
        3. 每一步 denoising 时，从整图 noisy latent 上切 latent crop
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
        **kwargs,
    ):
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if image is None:
            raise ValueError("`image` cannot be None.")

        if mask is None:
            raise ValueError("`mask` cannot be None，因为训练时使用了 mask 条件。")

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
        # 这里不再用 VAE encode mask。
        # 按你最新训练代码，mask crop 会进入：
        #   unet.conv_in.project_mask_crop(mask_crop)
        #
        # mask_crop:
        #   [N, 1, 512, 512]
        #   range [0, 1]
        # ------------------------------------------------------------
        mask_image = self.preprocess_mask_to_tensor(
            mask=mask,
            height=pixel_h,
            width=pixel_w,
            device=device,
            dtype=dtype,
        )

        inpaint_mask_tensor=self.preprocess_mask_to_tensor(
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
        # 5. 生成像素窗口
        #
        # 注意：
        #   窗口仍然用像素坐标保存。
        #   后面切 latent 时再除以 vae_scale_factor。
        # ------------------------------------------------------------
        windows = self.get_adaptive_pixel_windows(
            image_h=pixel_h,
            image_w=pixel_w,
            crop_size=crop_size,
            target_overlap=crop_overlap,
        )

        num_crops = len(windows)

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
        # 7. 从整图 condition latent 上切出每个 crop 的 image condition
        #
        # image_crop_latents:
        #   [N, 4, 64, 64]
        # ------------------------------------------------------------
        image_crop_latents = self.crop_latents_from_global(
            global_latents=original_latents_full,
            windows=windows,
            latent_crop_size=latent_crop_size,
        )

        # ------------------------------------------------------------
        # 8. 从整图 mask 上切 mask crop，然后通过训练时的投影层
        #
        # mask_crops:
        #   [N, 1, 512, 512]
        #
        # mask_crop_latents:
        #   [N, 4, 64, 64]
        # ------------------------------------------------------------
        mask_crops = self.crop_mask_from_global(
            mask_image=mask_image,
            windows=windows,
            crop_size=crop_size,
        )

        mask_crop_latents = self.encode_mask_crops_to_latents(
            mask_crops=mask_crops,
            dtype=dtype,
            device=device,
            encode_batch_size=crop_process_batch_size,
        )
        # ------------------------------------------------------------
        # 9. center map
        #
        # center_maps:
        #   [N, 2, 64, 64]
        #
        # 和训练代码一致：
        #   cx = crop center x / image_w
        #   cy = crop center y / image_h
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
                dtype=dtype,
                device=device,
            )
            center_maps.append(cm)

        center_maps = torch.cat(center_maps, dim=0)

        original_latents_full_per_crop = original_latents_full.expand(
            num_crops,
            -1,
            -1,
            -1,
        )

        # ------------------------------------------------------------
        # 11. CFG 条件扩展
        #
        # 普通条件是 N 个 crop：
        #   image_crop_latents
        #   mask_crop_latents
        #   projected_global_latents
        #   center_maps
        #
        # CFG 后变成 3N：
        #   text branch:  使用条件
        #   image branch: 使用条件
        #   uncond branch: 条件置零
        # ------------------------------------------------------------
        if self.do_classifier_free_guidance:
            image_crop_latents = self.make_cfg_condition(
                image_crop_latents
            )
            mask_crop_latents = self.make_cfg_condition(
                mask_crop_latents
            )
            original_latents_full_per_crop = self.make_cfg_condition(
                original_latents_full_per_crop
            )
            center_maps = self.make_cfg_condition(
                center_maps
            )

        # ------------------------------------------------------------
        # 12. prompt embeds 复制到每个 crop
        #
        # 原始：
        #   CFG: [3, seq, dim]
        #
        # 变成：
        #   CFG: [3N, seq, dim]
        # ------------------------------------------------------------
        prompt_embeds = self.repeat_cfg_tensor_for_crops(
            tensor=prompt_embeds,
            num_crops=num_crops,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )

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
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        # ------------------------------------------------------------
        # 16. denoising loop
        #
        # 每一步流程：
        #
        #   global_latents
        #       ↓ 切 latent crop
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
                # 16.1 从当前整图 latent 切出所有 crop latent
                #
                # crop_latents:
                #   [N, 4, 64, 64]
                # ----------------------------------------------------
                crop_latents = self.crop_latents_from_global(
                    global_latents=global_latents,
                    windows=windows,
                    latent_crop_size=latent_crop_size,
                )

                # ----------------------------------------------------
                # 16.2 CFG 扩展
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

                noise_pred_chunks = []
                total_model_batch = latent_model_input.shape[0]

                # ----------------------------------------------------
                # 16.3 分批跑 UNet
                #
                # 注意：
                #   这里分批的是 3N 维度。
                #   所以所有条件 tensor 都必须已经扩展到 3N。
                # ----------------------------------------------------
                for start in range(
                    0,
                    total_model_batch,
                    crop_process_batch_size,
                ):
                    end = min(
                        start + crop_process_batch_size,
                        total_model_batch,
                    )

                    # ----------------------------------------------------
                    # 当前分块的所有输入
                    # ----------------------------------------------------
                    latent_chunk = latent_model_input[start:end]

                    image_cond_chunk = image_crop_latents[start:end]

                    mask_cond_chunk = mask_crop_latents[start:end]

                    full_latent_chunk = (
                        original_latents_full_per_crop[start:end]
                    )

                    center_chunk = center_maps[start:end]

                    prompt_chunk = prompt_embeds[start:end]

                    # ----------------------------------------------------
                    # 把当前 chunk 的条件临时保存到 conv_in
                    #
                    # 此处只保存 Tensor 引用，不执行可训练计算。
                    # global_projector 会在 conv_in.forward() 内执行。
                    # ----------------------------------------------------
                    
                    self.unet.conv_in.set_condition_context(
                        original_crop_latents=image_cond_chunk,
                        mask_embeds=mask_cond_chunk,
                        original_latents_full=full_latent_chunk,
                        center_maps=center_chunk,
                    )

                    try:
                        # ------------------------------------------------
                        # 新版 UNet 只直接接收4通道 noisy latent
                        #
                        # 不再构造20通道 model_input。
                        # ------------------------------------------------
                        noise_pred_chunk = self.unet(
                            latent_chunk,
                            t,
                            encoder_hidden_states=prompt_chunk,
                            cross_attention_kwargs=cross_attention_kwargs,
                            return_dict=False,
                        )[0]

                    finally:
                        # 防止长期持有 Tensor 或计算图引用。
                        # 即使 UNet forward 报错，也会执行清理。
                        self.unet.conv_in.clear_condition_context()

                    noise_pred_chunks.append(
                        noise_pred_chunk
                    )

                crop_noise_pred = torch.cat(noise_pred_chunks, dim=0)

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

                # ----------------------------------------------------
                # 16.5 先融合所有 crop 的噪声预测
                #
                # 对重叠位置做逐通道简单平均，得到与整图 latent
                # 尺寸相同的噪声预测：
                #   global_noise_pred: [1, 4, H_lat, W_lat]
                # ----------------------------------------------------
                global_noise_pred = self.fuse_crop_tensors_to_global(
                    crop_tensors=crop_noise_pred,
                    windows=windows,
                    latent_h=latent_h,
                    latent_w=latent_w,
                    latent_crop_size=latent_crop_size,
                )

                # ----------------------------------------------------
                # 16.6 对整张图只执行一次 scheduler.step
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
                        inpaint_mask_latent * global_latents
                        + (1.0 - inpaint_mask_latent) * input_latents_noisy
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
    # mask crop -> mask latent
    # ================================================================
    def encode_mask_crops_to_latents(
        self,
        mask_crops: torch.Tensor,
        dtype,
        device,
        encode_batch_size: int = 8,
    ):
        """
        与训练代码中的 mask 编码过程完全一致。

        输入：
            mask_crops:
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

        if mask_crops.ndim != 4:
            raise ValueError(
                f"mask_crops 应为 [N,1,H,W]，实际为 {mask_crops.shape}"
            )

        if mask_crops.shape[1] != 1:
            raise ValueError(
                f"mask_crops 应为单通道，实际通道数为 {mask_crops.shape[1]}"
            )

        if encode_batch_size <= 0:
            raise ValueError(
                f"encode_batch_size 必须大于 0，实际为 {encode_batch_size}"
            )

        mask_crops = mask_crops.to(
            device=device,
            dtype=dtype,
        )

        # 避免输入 mask 因格式问题超过 [0,1]
        mask_crops = mask_crops.clamp(0.0, 1.0)

        mask_latent_chunks = []

        for start in range(0, mask_crops.shape[0], encode_batch_size):
            end = min(
                start + encode_batch_size,
                mask_crops.shape[0],
            )

            mask_chunk = mask_crops[start:end]

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
    ):
        """
        把每个 crop tensor 写回整图 tensor。

        重叠区域平均。

        crop_tensors:
            [N, C, latent_crop_size, latent_crop_size]

        return:
            [1, C, latent_h, latent_w]
        """

        num_crops, channels, _, _ = crop_tensors.shape
        device = crop_tensors.device
        dtype = crop_tensors.dtype

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

        for idx, win in enumerate(windows):
            latent_top = win["pixel_top"] // self.vae_scale_factor
            latent_left = win["pixel_left"] // self.vae_scale_factor

            global_sum[
                :,
                :,
                latent_top : latent_top + latent_crop_size,
                latent_left : latent_left + latent_crop_size,
            ] += crop_tensors[idx : idx + 1]

            global_count[
                :,
                :,
                latent_top : latent_top + latent_crop_size,
                latent_left : latent_left + latent_crop_size,
            ] += 1.0

        global_tensor = global_sum / global_count.clamp(min=1e-6)

        return global_tensor

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

        max_start = length - crop

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
