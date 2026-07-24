import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalLatentProjector(nn.Module):
    """
    多尺度全局上下文投影模块。

    输入:
        original_crop_latents: [B, 4, 64, 64]
        original_latents_full: [B, 4, H_lat, W_lat]
        center_maps:           [B, 2, 64, 64]

    输出:
        projected_global_latents: [B, 6, 64, 64]
    """

    def __init__(
        self,
        out_channels=6,
        hidden_channels=64,
        out_size=64,
        context_sizes=(128, 192),
    ):
        super().__init__()

        self.out_size = out_size
        self.context_sizes = context_sizes

        # crop latent 4
        # 多尺度 context latent 4 * len(context_sizes)
        # global latent 4
        # center maps 2
        in_channels = 4 + 4 * len(context_sizes) + 4 + 2

        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),

            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),

            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),

            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
        )

        self._init_weights()

    def _init_weights(self):
        """
        初始化 global projector。

        最后一层置零:
            让新增 global 条件一开始输出接近 0，
            避免训练初期直接破坏原始 UNet 的输入分布。
        """

        with torch.no_grad():
            for module in self.fuse:
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                    if module.bias is not None:
                        module.bias.zero_()

            final_conv = self.fuse[-1]
            final_conv.weight.zero_()

            if final_conv.bias is not None:
                final_conv.bias.zero_()

    def crop_by_center(
        self,
        full_latents: torch.Tensor,
        centers: torch.Tensor,
        crop_size: int,
    ) -> torch.Tensor:
        """
        从整图 latent 中，以 center 为中心裁剪上下文区域。

        参数:
            full_latents:
                [B, 4, H_lat, W_lat]

            centers:
                [B, 2]
                centers[:, 0] 是 x 相对坐标，范围 [0, 1]
                centers[:, 1] 是 y 相对坐标，范围 [0, 1]

            crop_size:
                latent 空间裁剪尺寸。

        返回:
            [B, 4, crop_size, crop_size]
        """

        bsz, _, h_lat, w_lat = full_latents.shape

        # 如果整图 latent 比 crop_size 小，直接 resize 兜底。
        if h_lat < crop_size or w_lat < crop_size:
            return F.interpolate(
                full_latents,
                size=(crop_size, crop_size),
                mode="bilinear",
                align_corners=False,
            )

        half = crop_size / 2.0
        crops = []

        for i in range(bsz):
            # 防止 center 意外越界。
            cx_rel = float(centers[i, 0].clamp(0, 1).item())
            cy_rel = float(centers[i, 1].clamp(0, 1).item())

            # 相对坐标转 latent 绝对坐标。
            cx = cx_rel * w_lat
            cy = cy_rel * h_lat

            # 以 center 为中心计算左上角。
            left = int(round(cx - half))
            top = int(round(cy - half))

            # 边界修正，保证裁剪窗口完整在图内。
            left = max(0, min(left, w_lat - crop_size))
            top = max(0, min(top, h_lat - crop_size))

            crop = full_latents[
                i:i + 1,
                :,
                top:top + crop_size,
                left:left + crop_size,
            ]

            crops.append(crop)

        return torch.cat(crops, dim=0)

    def forward(
        self,
        original_crop_latents: torch.Tensor,
        original_latents_full: torch.Tensor,
        center_maps: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播。

        输入:
            original_crop_latents: [B, 4, 64, 64]
            original_latents_full: [B, 4, H_lat, W_lat]
            center_maps:           [B, 2, 64, 64]

        输出:
            [B, 6, 64, 64]
        """

        # center_maps 每个空间位置值都一样，取任意一个位置即可。
        centers = center_maps[:, :, 0, 0]

        features = []

        # 1. 当前 crop latent。
        if original_crop_latents.shape[-2:] != (self.out_size, self.out_size):
            original_crop_latents = F.interpolate(
                original_crop_latents,
                size=(self.out_size, self.out_size),
                mode="bilinear",
                align_corners=False,
            )

        features.append(original_crop_latents)

        # 2. 多尺度上下文 latent。
        for crop_size in self.context_sizes:
            context = self.crop_by_center(
                full_latents=original_latents_full,
                centers=centers,
                crop_size=crop_size,
            )

            context = F.interpolate(
                context,
                size=(self.out_size, self.out_size),
                mode="bilinear",
                align_corners=False,
            )

            features.append(context)

        # 3. 整图 global latent。
        global_latent = F.interpolate(
            original_latents_full,
            size=(self.out_size, self.out_size),
            mode="bilinear",
            align_corners=False,
        )

        features.append(global_latent)

        # 4. center 位置编码。
        features.append(center_maps)

        # context_sizes=(64,128,192) 时:
        # 4 + 4 + 4 + 4 + 4 + 2 = 22 channels
        x = torch.cat(features, dim=1)

        return self.fuse(x)

class ContentAwarePositionGatedConvIn(nn.Module):
    """
    替换 UNet 原始 conv_in。

    外部送入 UNet 的输入：
        noisy_latents: [B, 4, 64, 64]

    在调用 UNet 前，通过 set_condition_context() 设置：
        original_crop_latents: [B, 4, 64, 64]
        mask_embeds:           [B, 4, 64, 64]
        original_latents_full: [B, 4, H_lat, W_lat]
        center_maps:           [B, 2, 64, 64]

    conv_in.forward() 内部会：
        1. 计算全局条件特征；
        2. 分别投影 noisy、crop、mask、global、center；
        3. 将各分支结果相加。

    最终输出：
        [B, unet_conv_in.out_channels, 64, 64]
    """

    def __init__(self, unet_conv_in: nn.Conv2d):
        super().__init__()

        out_channels = unet_conv_in.out_channels

        # =====================================================
        # 全局上下文投影器
        # 该模块必须在 forward 内部被调用，
        # 才能稳定地参与 DDP 梯度同步。
        # =====================================================
        self.global_projector = GlobalLatentProjector(
            out_channels=6,
            hidden_channels=64,
            out_size=64,
            context_sizes=(64, 128, 192),
        )

        # =====================================================
        # 各输入分支
        # =====================================================
        self.noise_branch = nn.Conv2d(
            4,
            out_channels,
            kernel_size=unet_conv_in.kernel_size,
            stride=unet_conv_in.stride,
            padding=unet_conv_in.padding,
            dilation=unet_conv_in.dilation,
            groups=unet_conv_in.groups,
            bias=unet_conv_in.bias is not None,
            padding_mode=unet_conv_in.padding_mode,
        )

        self.crop_branch = nn.Conv2d(
            4,
            out_channels,
            kernel_size=unet_conv_in.kernel_size,
            stride=unet_conv_in.stride,
            padding=unet_conv_in.padding,
            dilation=unet_conv_in.dilation,
            groups=unet_conv_in.groups,
            bias=False,
            padding_mode=unet_conv_in.padding_mode,
        )

        self.mask_branch = nn.Conv2d(
            4,
            out_channels,
            kernel_size=unet_conv_in.kernel_size,
            stride=unet_conv_in.stride,
            padding=unet_conv_in.padding,
            dilation=unet_conv_in.dilation,
            groups=unet_conv_in.groups,
            bias=False,
            padding_mode=unet_conv_in.padding_mode,
        )

        self.global_branch = nn.Conv2d(
            6,
            out_channels,
            kernel_size=unet_conv_in.kernel_size,
            stride=unet_conv_in.stride,
            padding=unet_conv_in.padding,
            dilation=unet_conv_in.dilation,
            groups=unet_conv_in.groups,
            bias=False,
            padding_mode=unet_conv_in.padding_mode,
        )

        self.center_branch = nn.Conv2d(
            2,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        # =====================================================
        # 临时条件上下文
        #
        # 它们不是 Parameter，也不注册为 buffer，
        # 只在当前 forward 前临时设置。
        # =====================================================
        self._original_crop_latents = None
        self._mask_embeds = None
        self._original_latents_full = None
        self._center_maps = None

        self._init_weights(unet_conv_in)

    def _init_weights(self, unet_conv_in: nn.Conv2d):
        """
        初始化新增分支。

        参数保持 fp32；
        混合精度由 accelerator.autocast() 控制。
        """

        with torch.no_grad():
            # noisy 分支继承预训练 Stable Diffusion conv_in。
            self.noise_branch.weight.copy_(unet_conv_in.weight)

            if unet_conv_in.bias is not None:
                self.noise_branch.bias.copy_(unet_conv_in.bias)

            # crop latent 同样是4通道 latent，
            # 以较小权重复制原始 conv_in。
            self.crop_branch.weight.copy_(unet_conv_in.weight)
            self.crop_branch.weight.mul_(0.1)

            # 新增条件分支从弱影响开始。
            self.mask_branch.weight.zero_()

            nn.init.normal_(
                self.global_branch.weight,
                mean=0.0,
                std=1e-3,
            )

            self.center_branch.weight.zero_()

    def set_condition_context(
        self,
        original_crop_latents: torch.Tensor,
        mask_embeds: torch.Tensor,
        original_latents_full: torch.Tensor,
        center_maps: torch.Tensor,
    ) -> None:
        """
        在调用完整 UNet 前设置当前 batch 的条件。

        注意：
            这里只保存 Tensor 引用，不执行含可训练参数的运算。
            global_projector 的运算会在 forward() 中执行。
        """

        if original_crop_latents.ndim != 4:
            raise ValueError(
                "original_crop_latents must be a 4D tensor, "
                f"got shape {original_crop_latents.shape}."
            )

        if mask_embeds.ndim != 4:
            raise ValueError(
                "mask_embeds must be a 4D tensor, "
                f"got shape {mask_embeds.shape}."
            )

        if original_latents_full.ndim != 4:
            raise ValueError(
                "original_latents_full must be a 4D tensor, "
                f"got shape {original_latents_full.shape}."
            )

        if center_maps.ndim != 4:
            raise ValueError(
                "center_maps must be a 4D tensor, "
                f"got shape {center_maps.shape}."
            )

        batch_size = original_crop_latents.shape[0]

        if mask_embeds.shape[0] != batch_size:
            raise ValueError(
                "mask_embeds batch size does not match "
                "original_crop_latents."
            )

        if original_latents_full.shape[0] != batch_size:
            raise ValueError(
                "original_latents_full batch size does not match "
                "original_crop_latents."
            )

        if center_maps.shape[0] != batch_size:
            raise ValueError(
                "center_maps batch size does not match "
                "original_crop_latents."
            )

        if original_crop_latents.shape[1] != 4:
            raise ValueError(
                "original_crop_latents must have 4 channels, "
                f"got {original_crop_latents.shape[1]}."
            )

        if mask_embeds.shape[1] != 4:
            raise ValueError(
                "mask_embeds must have 4 channels, "
                f"got {mask_embeds.shape[1]}."
            )

        if original_latents_full.shape[1] != 4:
            raise ValueError(
                "original_latents_full must have 4 channels, "
                f"got {original_latents_full.shape[1]}."
            )

        if center_maps.shape[1] != 2:
            raise ValueError(
                "center_maps must have 2 channels, "
                f"got {center_maps.shape[1]}."
            )

        self._original_crop_latents = original_crop_latents
        self._mask_embeds = mask_embeds
        self._original_latents_full = original_latents_full
        self._center_maps = center_maps

    def clear_condition_context(self) -> None:
        """
        清除当前 batch 的条件引用，避免长期持有计算图或大 Tensor。
        """

        self._original_crop_latents = None
        self._mask_embeds = None
        self._original_latents_full = None
        self._center_maps = None

    def _check_condition_context(self, noisy_latents: torch.Tensor) -> None:
        """
        检查调用 forward 前是否已经设置条件。
        """

        missing = []

        if self._original_crop_latents is None:
            missing.append("original_crop_latents")

        if self._mask_embeds is None:
            missing.append("mask_embeds")

        if self._original_latents_full is None:
            missing.append("original_latents_full")

        if self._center_maps is None:
            missing.append("center_maps")

        if missing:
            raise RuntimeError(
                "Condition context has not been set. "
                "Call conv_in.set_condition_context(...) before UNet forward. "
                f"Missing: {', '.join(missing)}."
            )

        batch_size = noisy_latents.shape[0]

        if self._original_crop_latents.shape[0] != batch_size:
            raise ValueError(
                "The condition batch size does not match noisy_latents: "
                f"{self._original_crop_latents.shape[0]} vs {batch_size}."
            )

    def forward(self, noisy_latents: torch.Tensor) -> torch.Tensor:
        """
        参数：
            noisy_latents: [B, 4, 64, 64]

        返回：
            [B, out_channels, 64, 64]
        """

        if noisy_latents.ndim != 4:
            raise ValueError(
                "noisy_latents must be a 4D tensor, "
                f"got shape {noisy_latents.shape}."
            )

        if noisy_latents.shape[1] != 4:
            raise ValueError(
                "ContentAwarePositionGatedConvIn expects "
                "4-channel noisy latents as the direct UNet input, "
                f"got {noisy_latents.shape[1]} channels."
            )

        self._check_condition_context(noisy_latents)

        original_crop_latents = self._original_crop_latents
        mask_embeds = self._mask_embeds
        original_latents_full = self._original_latents_full
        center_maps = self._center_maps

        # 尺寸统一。
        target_size = noisy_latents.shape[-2:]

        if original_crop_latents.shape[-2:] != target_size:
            original_crop_latents = F.interpolate(
                original_crop_latents,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        if mask_embeds.shape[-2:] != target_size:
            mask_embeds = F.interpolate(
                mask_embeds,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        if center_maps.shape[-2:] != target_size:
            center_maps = F.interpolate(
                center_maps,
                size=target_size,
                mode="nearest",
            )

        # =====================================================
        # 关键：
        # global_projector 在完整 UNet forward 内部执行。
        # DDP 可以正常追踪这些参数。
        # =====================================================
        projected_global_latents = self.global_projector(
            original_crop_latents=original_crop_latents,
            original_latents_full=original_latents_full,
            center_maps=center_maps,
        )

        noise_feat = self.noise_branch(noisy_latents)

        condition_feat = (
            self.crop_branch(original_crop_latents)
            + self.mask_branch(mask_embeds)
            + self.global_branch(projected_global_latents)
            + self.center_branch(center_maps)
        )

        return noise_feat + condition_feat

    @classmethod
    def apply_to_unet(cls, unet):
        """
        替换 UNet 原始 conv_in。

        注意：
            修改后，UNet 的直接输入仍然是4通道 noisy_latents。
            其他条件通过 set_condition_context() 传入。
        """

        unet.conv_in = cls(unet.conv_in)

        # 外部调用 unet 时只传4通道 noisy_latents。
        unet.register_to_config(in_channels=4)

        return unet