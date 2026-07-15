import os
import cv2
import random
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np


def parse_boxes_from_xml(xml_path):
    boxes = []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue

            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))

            boxes.append((xmin, ymin, xmax, ymax))

    except Exception as e:
        print(f"Warning: failed to parse {xml_path}: {e}")

    return boxes


def fill_one_box(image, inpaint_mask, box):
    img_h, img_w = image.shape[:2]
    xmin, ymin, xmax, ymax = box

    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img_w, xmax)
    ymax = min(img_h, ymax)

    if xmax <= xmin or ymax <= ymin:
        return

    roi = image[ymin:ymax, xmin:xmax]

    if roi.size == 0:
        return

    mean_color = roi.reshape(-1, 3).mean(axis=0).astype(np.uint8)

    image[ymin:ymax, xmin:xmax] = mean_color
    inpaint_mask[ymin:ymax, xmin:xmax] = 255


def fill_all_boxes(image, inpaint_mask, boxes):
    for box in boxes:
        fill_one_box(image, inpaint_mask, box)


def fill_large_region_near_box(
    image,
    inpaint_mask,
    boxes,
    large_region_size=1024,
):
    img_h, img_w = image.shape[:2]

    xmin, ymin, xmax, ymax = random.choice(boxes)

    box_cx = (xmin + xmax) / 2.0
    box_cy = (ymin + ymax) / 2.0

    offset_x = random.uniform(
        -large_region_size * 0.25,
        large_region_size * 0.25,
    )
    offset_y = random.uniform(
        -large_region_size * 0.25,
        large_region_size * 0.25,
    )

    region_cx = box_cx + offset_x
    region_cy = box_cy + offset_y

    actual_region_w = min(large_region_size, img_w)
    actual_region_h = min(large_region_size, img_h)

    left = int(region_cx - actual_region_w / 2)
    top = int(region_cy - actual_region_h / 2)

    left = max(0, min(left, img_w - actual_region_w))
    top = max(0, min(top, img_h - actual_region_h))

    right = left + actual_region_w
    bottom = top + actual_region_h

    region = image[top:bottom, left:right]

    if region.size == 0:
        return

    mean_color = region.reshape(-1, 3).mean(axis=0).astype(np.uint8)

    image[top:bottom, left:right] = mean_color
    inpaint_mask[top:bottom, left:right] = 255


def fill_box_with_mean_color(
    xml_path,
    image_path,
    large_region_prob=0.5,
    large_region_size=1024,
):
    """
    返回:
        filled_image:
            PIL.Image.Image, RGB
            均值填充后的输入图像

        inpaint_mask:
            PIL.Image.Image, L
            白色 255 = 需要修复 / 允许修改
            黑色 0   = 保留原图 / 不允许修改

    逻辑:
        1. 读取 XML 中所有 box。
        2. 如果没有 box，返回原图和全黑 mask。
        3. 如果进入大区域模式:
            - 随机选一个 box；
            - 在它附近生成 large_region_size × large_region_size 区域；
            - 填充这个大区域；
            - 同时所有 box 也继续填充。
        4. 如果不进入大区域模式:
            - 只填充所有 box。
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    img_h, img_w = image.shape[:2]
    boxes = parse_boxes_from_xml(xml_path)

    inpaint_mask = np.zeros((img_h, img_w), dtype=np.uint8)

    if len(boxes) == 0:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image_rgb), Image.fromarray(inpaint_mask, mode="L")

    use_large_region = random.random() < large_region_prob

    if use_large_region:
        fill_large_region_near_box(
            image=image,
            inpaint_mask=inpaint_mask,
            boxes=boxes,
            large_region_size=large_region_size,
        )

        # 关键修改：大区域模式下，其他所有 box 也要填充
        fill_all_boxes(
            image=image,
            inpaint_mask=inpaint_mask,
            boxes=boxes,
        )

    else:
        fill_all_boxes(
            image=image,
            inpaint_mask=inpaint_mask,
            boxes=boxes,
        )

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    filled_image = Image.fromarray(image_rgb)
    inpaint_mask_pil = Image.fromarray(inpaint_mask, mode="L")

    return filled_image, inpaint_mask_pil


if __name__ == "__main__":
    xml_path = "/root/autodl-tmp/instruct/instruct-pix2pix/datasets/UAVPDD/Annotations/lr_00001_bottom_left.xml"
    image_path = "/root/autodl-tmp/instruct/instruct-pix2pix/datasets/UAVPDD/JPEGImages/lr_00001_bottom_left.jpg"

    result, inpaint_mask = fill_box_with_mean_color(
        xml_path=xml_path,
        image_path=image_path,
        large_region_prob=0.5,
        large_region_size=1024,
    )

    save_path = "/root/autodl-tmp/instruct/instruct-pix2pix/test_mean_fill.jpg"
    mask_save_path = "/root/autodl-tmp/instruct/instruct-pix2pix/test_inpaint_mask.png"

    result.save(save_path)
    inpaint_mask.save(mask_save_path)

    print(f"均值填充图已保存到: {save_path}")
    print(f"inpaint_mask 已保存到: {mask_save_path}")