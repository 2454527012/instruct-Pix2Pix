import torch
import torch.nn as nn


class ContentAwarePositionGatedConvIn(nn.Module):
    """
    Content-aware Position Gated ConvIn.

    输入通道：
        0:4      noisy_latents
        4:8      crop/local/original latents
        8:12     mask latents
        12:16    global/square latents
        16:18    center maps

    总输入：
        18 channels
    """

    def __init__(self, unet_conv_in: nn.Conv2d):
        super().__init__()

        out_channels = unet_conv_in.out_channels

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

        # crop_latents 4 + mask_latents 4 + global_latents 4 + center_maps 2 = 14
        self.gate_branch = nn.Sequential(
            nn.Conv2d(14, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self._init_weights(unet_conv_in)

    def _init_weights(self, unet_conv_in: nn.Conv2d):
        with torch.no_grad():
            self.noise_branch.weight.copy_(unet_conv_in.weight)

            if unet_conv_in.bias is not None:
                self.noise_branch.bias.copy_(unet_conv_in.bias)

            self.crop_branch.weight.zero_()
            self.mask_branch.weight.zero_()
            self.global_branch.weight.zero_()

            for module in self.gate_branch:
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                    if module.bias is not None:
                        module.bias.zero_()

            final_gate_conv = self.gate_branch[-2]
            final_gate_conv.weight.zero_()
            final_gate_conv.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 18:
            raise ValueError(
                f"ContentAwarePositionGatedConvIn expects 18 input channels, "
                f"but got {x.shape[1]}."
            )

        noisy_latents = x[:, 0:4, :, :]
        crop_latents = x[:, 4:8, :, :]
        mask_latents = x[:, 8:12, :, :]
        global_latents = x[:, 12:16, :, :]
        center_maps = x[:, 16:18, :, :]

        noise_feat = self.noise_branch(noisy_latents)
        crop_feat = self.crop_branch(crop_latents)
        mask_feat = self.mask_branch(mask_latents)
        global_feat = self.global_branch(global_latents)

        gate_input = torch.cat(
            [crop_latents, mask_latents, global_latents, center_maps],
            dim=1,
        )

        gate = self.gate_branch(gate_input)

        condition_feat = gate * crop_feat + (1.0 - gate) * global_feat

        condition_feat = condition_feat + mask_feat

        return noise_feat + condition_feat

    @classmethod
    def apply_to_unet(cls, unet):
        unet.conv_in = cls(unet.conv_in)
        unet.register_to_config(in_channels=18)
        return unet