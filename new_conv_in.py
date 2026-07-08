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

    输入通道:
        0:4      noisy_latents              [B, 4, 64, 64]
        4:8      crop/original latents      [B, 4, 64, 64]
        8:10     mask embeds               [B, 2, 64, 64]
        10:16    projected global latents   [B, 6, 64, 64]
        16:18    center maps               [B, 2, 64, 64]

    总输入:
        18 channels
    """

    def __init__(self, unet_conv_in: nn.Conv2d):
        super().__init__()

        out_channels = unet_conv_in.out_channels

        self.global_projector = GlobalLatentProjector(
            out_channels=6,
            hidden_channels=64,
            out_size=64,
            context_sizes=(64, 128, 192),
        )

        self.mask_projector = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),   # 512 -> 256
            nn.SiLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 256 -> 128
            nn.SiLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 128 -> 64
            nn.SiLU(),

            nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1),   # [B, 2, 64, 64]
        )

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
            2,
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

        self._init_weights(unet_conv_in)

    def _init_weights(self, unet_conv_in: nn.Conv2d):
        """
        初始化新增 conv_in 分支。

        重点:
            不要把可训练参数手动转 fp16。
            mixed precision 训练时，参数保持 fp32，
            前向计算由 accelerator.autocast() 自动处理 fp16。
        """

        with torch.no_grad():
            # noisy 分支继承原始 conv_in。
            self.noise_branch.weight.copy_(unet_conv_in.weight)

            if unet_conv_in.bias is not None:
                self.noise_branch.bias.copy_(unet_conv_in.bias)

            # crop 分支也是 4 通道 latent，可以复制原始 conv_in。
            self.crop_branch.weight.copy_(unet_conv_in.weight)
            self.crop_branch.weight.mul_(0.1)

            # mask/global 是新条件，输入通道和原始 conv_in 不一致，不能 copy。
            # 这里建议初始化为 0，让它们从无影响开始学习。
            self.mask_branch.weight.zero_()
            self.global_branch.weight.zero_()

            # center 位置条件也从 0 开始学习。
            self.center_branch.weight.zero_()

    def project_global_latents(
        self,
        original_crop_latents: torch.Tensor,
        original_latents_full: torch.Tensor,
        center_maps: torch.Tensor,
    ) -> torch.Tensor:
        """
        生成全局上下文条件特征。

        注意:
            这里不要写 self.global_projector.to(dtype=...)
            否则会把可训练参数真的变成 fp16，
            导致 accelerate 报:
                Attempting to unscale FP16 gradients.
        """

        return self.global_projector(
            original_crop_latents=original_crop_latents,
            original_latents_full=original_latents_full,
            center_maps=center_maps,
        )

    def project_mask_crop(self, mask_crop: torch.Tensor) -> torch.Tensor:
        """
        将单通道 mask crop 编码成 2 通道条件。

        输入:
            mask_crop: [B, 1, 512, 512]

        输出:
            mask_embeds: [B, 2, 64, 64]

        注意:
            这里也不要手动把 mask_projector 转 fp16。
            让 accelerator.autocast() 负责前向精度。
        """

        return self.mask_projector(mask_crop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 18, 64, 64]
        """

        if x.shape[1] != 18:
            raise ValueError(
                f"ContentAwarePositionGatedConvIn expects 18 input channels, "
                f"got {x.shape[1]}."
            )

        noisy_latents = x[:, 0:4]
        crop_latents = x[:, 4:8]
        mask_embeds = x[:, 8:10]
        global_latents = x[:, 10:16]
        center_maps = x[:, 16:18]

        noise_feat = self.noise_branch(noisy_latents)

        condition_feat = (
            self.crop_branch(crop_latents)
            + self.mask_branch(mask_embeds)
            + self.global_branch(global_latents)
            + self.center_branch(center_maps)
        )

        return noise_feat + condition_feat

    @classmethod
    def apply_to_unet(cls, unet):
        """
        替换 UNet 的 conv_in，并更新配置输入通道数。
        """

        unet.conv_in = cls(unet.conv_in)
        unet.register_to_config(in_channels=18)

        return unet