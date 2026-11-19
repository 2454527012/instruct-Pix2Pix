"""512x512 letterboxed whole-image inpainting pipeline.

This is intentionally separate from ``pipline.py``.  It runs only the
whole-image branch used during mixed training: a letterboxed 512x512 input,
its mask condition, the native-resolution filled-image latent context, and a
zero position condition.
"""

import inspect
from typing import Any

import PIL.Image
import torch
import torch.nn.functional as F

from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

from processData.pipline import SlidingWindowInstructPix2PixPipeline


class ThumbnailInstructPix2PixPipeline(SlidingWindowInstructPix2PixPipeline):
    """Generate one completed 512x512 thumbnail before any crop refinement."""

    def _get_thumbnail_scheduler(self):
        """Build the same independent DDIM scheduler used during training.

        Training first loads the base scheduler as ``DDPMScheduler`` and then
        constructs ``DDIMScheduler`` from that config.  Keep this scheduler
        separate from ``self.scheduler`` so Stage 1 cannot consume or mutate
        the scheduler state used later by Stage 2.
        """

        thumbnail_scheduler = self.__dict__.get("_thumbnail_ddim_scheduler")
        if thumbnail_scheduler is None:
            training_noise_scheduler = DDPMScheduler.from_config(
                self.scheduler.config
            )
            thumbnail_scheduler = DDIMScheduler.from_config(
                training_noise_scheduler.config
            )
            self._thumbnail_ddim_scheduler = thumbnail_scheduler
        return thumbnail_scheduler

    def _get_letterbox_geometry(self, image_size: tuple[int, int], size: int):
        image_width, image_height = image_size
        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"Invalid input image size: {image_size}")

        scale = min(size / image_width, size / image_height)
        resized_width = max(1, min(size, round(image_width * scale)))
        resized_height = max(1, min(size, round(image_height * scale)))
        left = (size - resized_width) // 2
        top = (size - resized_height) // 2
        return resized_width, resized_height, left, top

    def _resize_and_letterbox(
        self,
        image: PIL.Image.Image,
        geometry: tuple[int, int, int, int],
        resample,
        size: int,
    ) -> PIL.Image.Image:
        resized_width, resized_height, left, top = geometry
        resized = image.resize((resized_width, resized_height), resample=resample)
        canvas = PIL.Image.new(image.mode, (size, size), color=0)
        canvas.paste(resized, (left, top))
        return canvas

    def _make_valid_mask(
        self,
        geometry: tuple[int, int, int, int],
        size: int,
    ) -> PIL.Image.Image:
        resized_width, resized_height, left, top = geometry
        valid_mask = PIL.Image.new("L", (size, size), color=0)
        valid_mask.paste(255, (left, top, left + resized_width, top + resized_height))
        return valid_mask

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        image: PIL.Image.Image,
        mask: PIL.Image.Image,
        inpaint_mask: PIL.Image.Image,
        thumbnail_size: int = 512,
        num_inference_steps: int = 10,
        guidance_scale: float = 3.0,
        image_guidance_scale: float = 2.0,
        negative_prompt: str | None = None,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Repair one full-image thumbnail.

        Args:
            image: Native-resolution *filled* image.  It is letterboxed for
                the direct 512 condition while its native VAE latent remains
                the global-context condition, matching the training branch.
            mask: Native-resolution dataset mask used as the mask condition.
            inpaint_mask: Native-resolution binary repair mask.  White/1 is
                generated; black/0 is restored from ``image`` every step.
        """

        if not isinstance(prompt, str):
            raise TypeError("thumbnail inference currently expects one prompt string")
        if not isinstance(image, PIL.Image.Image):
            raise TypeError("image must be a PIL.Image.Image")
        if not isinstance(mask, PIL.Image.Image):
            raise TypeError("mask must be a PIL.Image.Image")
        if not isinstance(inpaint_mask, PIL.Image.Image):
            raise TypeError("inpaint_mask must be a PIL.Image.Image")
        if thumbnail_size % self.vae_scale_factor != 0:
            raise ValueError(
                f"thumbnail_size={thumbnail_size} must be divisible by "
                f"the VAE scale factor {self.vae_scale_factor}"
            )

        image = image.convert("RGB")
        mask = mask.convert("L")
        inpaint_mask = inpaint_mask.convert("L")
        if image.size != mask.size or image.size != inpaint_mask.size:
            raise ValueError(
                "image, mask, and inpaint_mask must have identical native sizes; "
                f"got {image.size}, {mask.size}, and {inpaint_mask.size}"
            )

        try:
            image_resample = PIL.Image.Resampling.BICUBIC
            mask_resample = PIL.Image.Resampling.NEAREST
        except AttributeError:
            image_resample = PIL.Image.BICUBIC
            mask_resample = PIL.Image.NEAREST

        geometry = self._get_letterbox_geometry(image.size, thumbnail_size)
        thumbnail_image = self._resize_and_letterbox(
            image, geometry, image_resample, thumbnail_size
        )
        thumbnail_mask = self._resize_and_letterbox(
            mask, geometry, mask_resample, thumbnail_size
        )
        thumbnail_inpaint_mask = self._resize_and_letterbox(
            inpaint_mask, geometry, mask_resample, thumbnail_size
        )
        thumbnail_valid_mask = self._make_valid_mask(geometry, thumbnail_size)

        self._guidance_scale = guidance_scale
        self._image_guidance_scale = image_guidance_scale
        device = self._execution_device
        thumbnail_scheduler = self._get_thumbnail_scheduler()

        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            1,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
        )
        dtype = prompt_embeds.dtype

        native_image_tensor = self.image_processor.preprocess(image).to(
            device=device,
            dtype=dtype,
        )
        thumbnail_image_tensor = self.image_processor.preprocess(thumbnail_image).to(
            device=device,
            dtype=dtype,
        )

        if thumbnail_image_tensor.shape[-2:] != (thumbnail_size, thumbnail_size):
            raise ValueError(
                "The image processor changed the thumbnail size to "
                f"{tuple(thumbnail_image_tensor.shape[-2:])}; expected "
                f"({thumbnail_size}, {thumbnail_size})."
            )
        if (
            native_image_tensor.shape[-2] % self.vae_scale_factor != 0
            or native_image_tensor.shape[-1] % self.vae_scale_factor != 0
        ):
            raise ValueError("Native image size must be divisible by the VAE scale factor")

        thumbnail_mask_tensor = self.preprocess_mask_to_tensor(
            thumbnail_mask,
            height=thumbnail_size,
            width=thumbnail_size,
            device=device,
            dtype=dtype,
        )
        thumbnail_inpaint_tensor = self.preprocess_mask_to_tensor(
            thumbnail_inpaint_mask,
            height=thumbnail_size,
            width=thumbnail_size,
            device=device,
            dtype=dtype,
        )
        thumbnail_inpaint_condition = (
            thumbnail_inpaint_tensor >= 0.5
        ).to(dtype=dtype)
        thumbnail_valid_tensor = self.preprocess_mask_to_tensor(
            thumbnail_valid_mask,
            height=thumbnail_size,
            width=thumbnail_size,
            device=device,
            dtype=dtype,
        )

        # These four conditions exactly correspond to the global sample in
        # training: letterboxed image, letterboxed mask, native full-image
        # latent, and zero position.
        native_full_latents = self.encode_image_full_to_latents(
            native_image_tensor,
            dtype=dtype,
            device=device,
        )
        thumbnail_image_latents = self.encode_image_full_to_latents(
            thumbnail_image_tensor,
            dtype=dtype,
            device=device,
        )
        thumbnail_mask_latents = self.encode_mask_to_latents(
            mask_images=thumbnail_mask_tensor,
            inpaint_masks=thumbnail_inpaint_condition,
            dtype=dtype,
            device=device,
            encode_batch_size=1,
        )

        latent_size = thumbnail_size // self.vae_scale_factor
        center_maps = torch.zeros(
            1,
            2,
            latent_size,
            latent_size,
            device=device,
            dtype=dtype,
        )
        if self.do_classifier_free_guidance:
            thumbnail_image_latents = self.make_cfg_condition(thumbnail_image_latents)
            thumbnail_mask_latents = self.make_cfg_condition(thumbnail_mask_latents)
            native_full_latents = self.make_cfg_condition(native_full_latents)
            center_maps = self.make_cfg_condition(center_maps)

        if self.unet.config.in_channels != self.vae.config.latent_channels:
            raise ValueError(
                "The thumbnail pipeline expects a 4-channel custom UNet input; "
                f"got unet.config.in_channels={self.unet.config.in_channels}."
            )

        latents, fixed_noise = self.prepare_global_latents(
            latent_h=latent_size,
            latent_w=latent_size,
            num_channels_latents=self.vae.config.latent_channels,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        # ``prepare_global_latents`` uses the refinement scheduler's sigma.
        # Re-apply the independent DDIM scheduler's sigma for Stage 1.
        latents = fixed_noise * thumbnail_scheduler.init_noise_sigma
        repair_mask_latent = F.interpolate(
            thumbnail_inpaint_tensor,
            size=(latent_size, latent_size),
            mode="nearest",
        )

        thumbnail_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = thumbnail_scheduler.timesteps
        step_parameters = set(
            inspect.signature(thumbnail_scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if "eta" in step_parameters:
            extra_step_kwargs["eta"] = eta
        if "generator" in step_parameters:
            extra_step_kwargs["generator"] = generator
        warmup_steps = (
            len(timesteps)
            - num_inference_steps * thumbnail_scheduler.order
        )

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for step_index, timestep in enumerate(timesteps):
                if self.do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents, latents, latents], dim=0)
                else:
                    latent_model_input = latents
                latent_model_input = thumbnail_scheduler.scale_model_input(
                    latent_model_input,
                    timestep,
                )

                noise_pred = self.run_conditioned_unet_in_batches(
                    latent_model_input=latent_model_input,
                    timestep=timestep,
                    image_condition=thumbnail_image_latents,
                    mask_condition=thumbnail_mask_latents,
                    full_image_condition=native_full_latents,
                    center_condition=center_maps,
                    prompt_embeds=prompt_embeds,
                    process_batch_size=3 if self.do_classifier_free_guidance else 1,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
                if self.do_classifier_free_guidance:
                    noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                    noise_pred = (
                        noise_pred_uncond
                        + self.guidance_scale * (noise_pred_text - noise_pred_image)
                        + self.image_guidance_scale
                        * (noise_pred_image - noise_pred_uncond)
                    )

                latents = thumbnail_scheduler.step(
                    noise_pred,
                    timestep,
                    latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )[0]

                # Keep known pixels identical to the filled input throughout
                # sampling.  inpaint_mask == 1 is the repair region.
                if step_index < len(timesteps) - 1:
                    next_timestep = timesteps[step_index + 1]
                    if torch.is_tensor(next_timestep):
                        next_timestep = next_timestep.reshape(1).to(
                            device=device,
                            dtype=torch.long,
                        )
                    else:
                        next_timestep = torch.tensor(
                            [next_timestep],
                            device=device,
                            dtype=torch.long,
                        )
                    known_latents = thumbnail_scheduler.add_noise(
                        original_samples=thumbnail_image_latents[:1],
                        noise=fixed_noise,
                        timesteps=next_timestep,
                    )
                else:
                    known_latents = thumbnail_image_latents[:1]
                latents = (
                    repair_mask_latent * latents
                    + (1.0 - repair_mask_latent) * known_latents
                )

                if step_index == len(timesteps) - 1 or (
                    (step_index + 1) > warmup_steps
                    and (step_index + 1) % thumbnail_scheduler.order == 0
                ):
                    progress_bar.update()

        if output_type == "latent":
            images = latents
            has_nsfw_concept = None
        else:
            decoded = self.vae.decode(
                latents / self.vae.config.scaling_factor,
                return_dict=False,
            )[0]
            decoded = (decoded / 2.0 + 0.5).clamp(0.0, 1.0)
            thumbnail_input_01 = (thumbnail_image_tensor / 2.0 + 0.5).clamp(0.0, 1.0)

            # Pixel-space compositing makes the known image pixels and the
            # artificial black letterbox border exact in the saved preview.
            decoded = (
                thumbnail_inpaint_tensor * decoded
                + (1.0 - thumbnail_inpaint_tensor) * thumbnail_input_01
            )
            decoded = decoded * thumbnail_valid_tensor
            images = self.image_processor.postprocess(
                decoded * 2.0 - 1.0,
                output_type=output_type,
                do_denormalize=[True],
            )
            has_nsfw_concept = None

        self.maybe_free_model_hooks()
        if not return_dict:
            return images, has_nsfw_concept
        return StableDiffusionPipelineOutput(
            images=images,
            nsfw_content_detected=has_nsfw_concept,
        )
