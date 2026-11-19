"""Two-stage whole-image restoration aligned with the mixed training setup.

Stage 1 restores one aspect-ratio-preserving 512x512 thumbnail. Stage 2 removes
the artificial letterbox, enlarges the restored thumbnail to native size, and
uses it only inside the repair mask as both the local crop condition and the
full-image condition of the sliding-window pipeline.
"""

from dataclasses import dataclass
from typing import Any

import PIL.Image
import torch
from diffusers.utils import BaseOutput

from processData.pipline import SlidingWindowInstructPix2PixPipeline
from processData.pipline_thumbnail import ThumbnailInstructPix2PixPipeline


@dataclass
class TwoStageInstructPix2PixPipelineOutput(BaseOutput):
    """Final result together with the two intermediate pixel-space images."""

    images: Any
    nsfw_content_detected: list[bool] | None = None
    thumbnail_images: list[PIL.Image.Image] | None = None
    upsampled_thumbnail_images: list[PIL.Image.Image] | None = None
    refinement_condition_images: list[PIL.Image.Image] | None = None


class TwoStageInstructPix2PixPipeline(ThumbnailInstructPix2PixPipeline):
    """Run the trained thumbnail branch, then the trained crop branch."""

    def build_refinement_condition(
        self,
        filled_image: PIL.Image.Image,
        inpaint_mask: PIL.Image.Image,
        thumbnail_result: PIL.Image.Image,
        thumbnail_size: int,
    ) -> tuple[PIL.Image.Image, PIL.Image.Image]:
        """Convert the letterboxed stage-1 result into the stage-2 condition.

        ``inpaint_mask`` follows the project convention: white/255 is the
        repair region and black/0 is the known region. The returned condition
        therefore contains the enlarged thumbnail prediction only in the
        repair region; every known pixel is copied exactly from ``filled_image``.
        """

        if thumbnail_result.size != (thumbnail_size, thumbnail_size):
            raise ValueError(
                "Stage-1 output has an unexpected size: "
                f"{thumbnail_result.size}; expected "
                f"({thumbnail_size}, {thumbnail_size})."
            )

        try:
            image_resample = PIL.Image.Resampling.BICUBIC
        except AttributeError:
            image_resample = PIL.Image.BICUBIC

        resized_width, resized_height, left, top = self._get_letterbox_geometry(
            filled_image.size,
            thumbnail_size,
        )
        thumbnail_content = thumbnail_result.convert("RGB").crop(
            (
                left,
                top,
                left + resized_width,
                top + resized_height,
            )
        )
        upsampled_thumbnail = thumbnail_content.resize(
            filled_image.size,
            resample=image_resample,
        )
        refinement_condition = PIL.Image.composite(
            upsampled_thumbnail,
            filled_image,
            inpaint_mask,
        )
        return upsampled_thumbnail, refinement_condition

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        image: PIL.Image.Image,
        mask: PIL.Image.Image,
        inpaint_mask: PIL.Image.Image,
        thumbnail_size: int = 512,
        thumbnail_num_inference_steps: int = 10,
        num_inference_steps: int = 50,
        thumbnail_guidance_scale: float = 3.0,
        thumbnail_image_guidance_scale: float = 2.0,
        guidance_scale: float = 3.0,
        image_guidance_scale: float = 2.0,
        crop_size: int = 512,
        crop_overlap: int = 128,
        crop_process_batch_size: int = 4,
        global_context_strength: float = 0.0,
        negative_prompt: str | None = None,
        thumbnail_eta: float = 0.0,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
        thumbnail_generator: torch.Generator | None = None,
        refinement_generator: torch.Generator | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        preserve_known_pixels: bool = True,
        cross_attention_kwargs: dict[str, Any] | None = None,
        **refinement_kwargs,
    ):
        """Restore one native-resolution image in two serial stages.

        The default ``global_context_strength=0`` is intentional. The first
        stage already provides the trained global structure prior; enabling
        the older simultaneous coarse branch would introduce a condition that
        was not used by this serial two-stage training path.
        """

        if not isinstance(prompt, str):
            raise TypeError("Two-stage inference currently expects one prompt string")
        if not isinstance(image, PIL.Image.Image):
            raise TypeError("image must be a PIL.Image.Image")
        if not isinstance(mask, PIL.Image.Image):
            raise TypeError("mask must be a PIL.Image.Image")
        if not isinstance(inpaint_mask, PIL.Image.Image):
            raise TypeError("inpaint_mask must be a PIL.Image.Image")

        filled_image = image.convert("RGB")
        dataset_mask = mask.copy()
        repair_mask = inpaint_mask.convert("L")
        if (
            filled_image.size != dataset_mask.size
            or filled_image.size != repair_mask.size
        ):
            raise ValueError(
                "image, mask, and inpaint_mask must have identical sizes; "
                f"got {filled_image.size}, {dataset_mask.size}, and "
                f"{repair_mask.size}."
            )
        if (
            filled_image.width % self.vae_scale_factor != 0
            or filled_image.height % self.vae_scale_factor != 0
        ):
            raise ValueError(
                "Native image width and height must be divisible by the VAE "
                f"scale factor {self.vae_scale_factor}; got {filled_image.size}."
            )

        if thumbnail_generator is None:
            thumbnail_generator = generator
        if refinement_generator is None:
            refinement_generator = generator

        # Stage 1: exactly the 1/8 whole-image branch used during training.
        thumbnail_output = ThumbnailInstructPix2PixPipeline.__call__(
            self,
            prompt=prompt,
            image=filled_image,
            mask=dataset_mask,
            inpaint_mask=repair_mask,
            thumbnail_size=thumbnail_size,
            num_inference_steps=thumbnail_num_inference_steps,
            guidance_scale=thumbnail_guidance_scale,
            image_guidance_scale=thumbnail_image_guidance_scale,
            negative_prompt=negative_prompt,
            eta=thumbnail_eta,
            generator=thumbnail_generator,
            output_type="pil",
            return_dict=True,
            cross_attention_kwargs=cross_attention_kwargs,
        )
        thumbnail_image = thumbnail_output.images[0].convert("RGB")

        # Match crop training in pixel space:
        #   condition = enlarged_thumbnail inside mask + filled input outside.
        (
            upsampled_thumbnail,
            refinement_condition,
        ) = self.build_refinement_condition(
            filled_image=filled_image,
            inpaint_mask=repair_mask,
            thumbnail_result=thumbnail_image,
            thumbnail_size=thumbnail_size,
        )

        # Stage 2: the same condition image is encoded once and used for both
        # original_crop_latents and original_latents_full, exactly as in the
        # seven crop rows of training.
        refinement_output = SlidingWindowInstructPix2PixPipeline.__call__(
            self,
            prompt=prompt,
            image=refinement_condition,
            mask=dataset_mask,
            inpaint_mask=repair_mask,
            crop_size=crop_size,
            crop_overlap=crop_overlap,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            negative_prompt=negative_prompt,
            eta=eta,
            generator=refinement_generator,
            output_type=output_type,
            return_dict=True,
            cross_attention_kwargs=cross_attention_kwargs,
            crop_process_batch_size=crop_process_batch_size,
            global_context_strength=global_context_strength,
            **refinement_kwargs,
        )

        final_images = refinement_output.images
        if preserve_known_pixels and output_type == "pil":
            final_images = [
                PIL.Image.composite(
                    result.convert("RGB"),
                    filled_image,
                    repair_mask,
                )
                for result in final_images
            ]

        if not return_dict:
            return (
                final_images,
                refinement_output.nsfw_content_detected,
                [thumbnail_image],
                [upsampled_thumbnail],
                [refinement_condition],
            )

        return TwoStageInstructPix2PixPipelineOutput(
            images=final_images,
            nsfw_content_detected=refinement_output.nsfw_content_detected,
            thumbnail_images=[thumbnail_image],
            upsampled_thumbnail_images=[upsampled_thumbnail],
            refinement_condition_images=[refinement_condition],
        )
