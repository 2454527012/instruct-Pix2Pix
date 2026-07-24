import sys
import gc
from pathlib import Path

import torch
import PIL.Image

from diffusers import UNet2DConditionModel
from safetensors.torch import load_file


# ============================================================
# path
# ============================================================

PROJECT_ROOT = Path(
    "/root/autodl-tmp/instruct/instruct-pix2pix"
)

sys.path.append(
    str(PROJECT_ROOT)
)


from processData.pipline import (
    SlidingWindowInstructPix2PixPipeline
)

from new_conv_in import (
    ContentAwarePositionGatedConvIn
)

from process_image import (
    fill_box_with_mean_color
)



# ============================================================
# config
# ============================================================

BASE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "stable-diffusion-v1-5"
)


MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "ip2p_use_latent_20_new"
)


IMAGE_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "UAVPDD"
    / "JPEGImages"
)


XML_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "UAVPDD"
    / "Annotations"
)


MASK_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "UAVPDD"
    / "masks"
)


VAL_TXT = (
    PROJECT_ROOT
    / "datasets"
    / "UAVPDD"
    / "ImageSets"
    / "Main"
    / "val.txt"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "results_global_latent_test_70000"
)



PROMPT = (
    "Road damage images acquired by unmanned aerial vehicles"
)


DEVICE = "cuda"

DTYPE = torch.float32


CROP_SIZE = 512

CROP_OVERLAP = 128


GUIDANCE_SCALE = 3.0

IMAGE_GUIDANCE_SCALE = 2.0


STEPS = 50


CROP_BATCH = 4


SEED = 42


MAX_IMAGES = 1000


SKIP_EXISTING = True



# ============================================================
# utils
# ============================================================

def clear_memory():

    gc.collect()

    torch.cuda.empty_cache()



def find_file(folder, name):

    for ext in [
        ".jpg",
        ".jpeg",
        ".png",
        ".JPG",
        ".PNG"
    ]:

        path = (
            folder
            /
            f"{name}{ext}"
        )

        if path.exists():
            return path

    return None



def crop_to_8(image):

    w,h = image.size

    w = w - w % 8
    h = h - h % 8

    return image.crop(
        (
            0,
            0,
            w,
            h
        )
    )



# ============================================================
# load model
# ============================================================

def load_pipeline():
    print("Loading UNet...")
    unet = (
        UNet2DConditionModel
        .from_pretrained(
            str(BASE_MODEL_PATH),
            subfolder="unet",
            torch_dtype=DTYPE,
        )
    )
    # replace conv_in
    ContentAwarePositionGatedConvIn.apply_to_unet(
        unet
    )
    weight_file = (
        MODEL_PATH
        /
        "unet"
        /
        "diffusion_pytorch_model.safetensors"
    )
    state_dict = load_file(
        str(weight_file)
    )
    unet.load_state_dict(
        state_dict,
        strict=False
    )
    print("Loading pipeline...")
    pipe = (
        SlidingWindowInstructPix2PixPipeline
        .from_pretrained(
            str(MODEL_PATH),
            unet=unet,
            torch_dtype=DTYPE,
            safety_checker=None,
        )
    )
    pipe.to(
        DEVICE
    )
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    return pipe




# ============================================================
# prepare input
# ============================================================


def prepare_input(
    image_file,
    xml_file,
    mask_file
):


    original = (
        PIL.Image
        .open(image_file)
        .convert("RGB")
    )


    edited,_ = (
        fill_box_with_mean_color(
            str(xml_file),
            str(image_file),
            large_region_prob=0.5
        )
    )


    edited = edited.convert(
        "RGB"
    )


    mask = (
        PIL.Image
        .open(mask_file)
        .convert("L")
    )


    original = crop_to_8(
        original
    )

    edited = crop_to_8(
        edited
    )

    mask = crop_to_8(
        mask
    )


    return (
        original,
        edited,
        mask
    )




# ============================================================
# inference one image
# ============================================================


@torch.inference_mode()
def inference_one(
    pipe,
    name,
    image_file,
    xml_file,
    mask_file,
    index
):


    save_dir = OUTPUT_DIR


    output_file = (
        save_dir
        /
        f"{name}_generate.png"
    )


    if (
        SKIP_EXISTING
        and output_file.exists()
    ):
        print(f'{name} has existed. skip')
        return



    original,edited,mask = prepare_input(
        image_file,
        xml_file,
        mask_file
    )


    generator = (
        torch.Generator(
            device="cpu"
        )
        .manual_seed(
            SEED+index
        )
    )



    result = pipe(

        prompt=PROMPT,

        image=edited,

        mask=mask,

        crop_size=CROP_SIZE,

        crop_overlap=CROP_OVERLAP,

        guidance_scale=GUIDANCE_SCALE,

        image_guidance_scale=
            IMAGE_GUIDANCE_SCALE,

        num_inference_steps=STEPS,

        crop_process_batch_size=
            CROP_BATCH,

        generator=generator,

        output_type="pil",

    ).images[0]



    original.save(
        save_dir /
        f"{name}_original.png"
    )


    edited.save(
        save_dir /
        f"{name}_input.png"
    )


    mask.save(
        save_dir /
        f"{name}_mask.png"
    )


    result.save(
        output_file
    )


    print(
        "done:",
        name
    )





# ============================================================
# main
# ============================================================


def main():


    OUTPUT_DIR.mkdir(
        exist_ok=True,
        parents=True
    )


    with open(
        VAL_TXT,
        "r"
    ) as f:

        names=[
            x.strip()
            for x in f
            if x.strip()
        ]



    if MAX_IMAGES:

        names = names[:MAX_IMAGES]



    pipe = load_pipeline()



    for idx,name in enumerate(names):


        try:


            image_file = find_file(
                IMAGE_DIR,
                name
            )


            xml_file = (
                XML_DIR
                /
                f"{name}.xml"
            )


            mask_file = find_file(
                MASK_DIR,
                name
            )



            if (
                image_file is None
                or not xml_file.exists()
                or mask_file is None
            ):

                print(
                    "missing:",
                    name
                )

                continue



            inference_one(
                pipe,
                name,
                image_file,
                xml_file,
                mask_file,
                idx
            )



        except Exception as e:

            print(
                "failed:",
                name,
                e
            )



        if idx % 10 == 0:

            clear_memory()



if __name__=="__main__":

    main()