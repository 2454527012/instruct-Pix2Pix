import sys
sys.path.append("/root/autodl-tmp/instruct/instruct-pix2pix")

import PIL.Image
import torch
from safetensors.torch import load_file
from diffusers import UNet2DConditionModel

from processData.pipline_14 import SlidingWindowInstructPix2PixPipeline
from new_conv_in import ContentAwarePositionGatedConvIn
from collections import Counter

base_model_path = "/root/autodl-tmp/instruct/instruct-pix2pix/models/stable-diffusion-v1-5"
model_path = "/root/autodl-tmp/instruct/instruct-pix2pix/outputs/ip2p_gate_512_20000"
image_path = "/root/autodl-tmp/instruct/instruct-pix2pix/datasets/UAVPDD/patched_images/lr_00001_bottom_left.jpg"
output_path = "/root/autodl-tmp/instruct/instruct-pix2pix/result_lr_00001_bottom_left_20000.png"


def check_model_weights(model):
    for name, p in model.named_parameters():
        if torch.isnan(p).any() or torch.isinf(p).any():
            raise RuntimeError(f"UNet weight NaN/Inf: {name}")
    print("UNet weights OK")


original_image = PIL.Image.open(image_path).convert("RGB")

# 1. 先从原始 SD1.5 加载 4 通道 UNet
unet = UNet2DConditionModel.from_pretrained(
    base_model_path,
    subfolder="unet",
    torch_dtype=torch.get_float32_matmul_precision,
    low_cpu_mem_usage=False,
)

# 2. 替换成你的自定义 14 通道 conv_in
ContentAwarePositionGatedConvIn.apply_to_unet(unet)

# 3. 加载你训练好的权重
state_dict = load_file(
    f"{model_path}/unet/diffusion_pytorch_model.safetensors"
)
print(Counter(v.dtype for v in state_dict.values()))
missing, unexpected = unet.load_state_dict(state_dict, strict=False)

print("missing keys:", missing)
print("unexpected keys:", unexpected)
print("conv_in type:", type(unet.conv_in))
print(unet.conv_in)

check_model_weights(unet)

# 4. 加载 pipeline，并传入自定义 unet
pipe = SlidingWindowInstructPix2PixPipeline.from_pretrained(
    model_path,
    unet=unet,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=False,
    device_map=None,
)

pipe = pipe.to("cuda")
pipe.set_progress_bar_config(disable=False)

# 5. 推理
result = pipe(
    prompt="Restore to aerial view of road damage taken by drone",
    image=original_image,
    crop_size=512,
    crop_overlap=128,
    guidance_scale=7.5,
    image_guidance_scale=1.5,
    num_inference_steps=50,
    crop_process_batch_size=8,
    safety_checker=None,
).images[0]

result.save(output_path)
print(f"saved to {output_path}")