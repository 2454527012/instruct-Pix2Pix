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


def fill_box_with_mean_color(
    xml_path,
    image_path,
    large_region_prob=0.2,
    large_region_size=1024,
):
    """
    80%:
        按原流程，把所有 box 区域填充为各自 box 内平均颜色。

    20%:
        随机选择一个 box，在该 box 附近取一个 1024×1024 区域，
        将整个区域填充为该区域内像素平均颜色。

    返回:
        PIL.Image.Image
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    img_h, img_w = image.shape[:2]

    boxes = parse_boxes_from_xml(xml_path)

    if len(boxes) == 0:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image_rgb)

    use_large_region = random.random() < large_region_prob

    # =========================
    # 20%：填充 box 附近 1024×1024 区域
    # =========================
    if use_large_region:
        xmin, ymin, xmax, ymax = random.choice(boxes)

        box_cx = (xmin + xmax) / 2.0
        box_cy = (ymin + ymax) / 2.0

        # 在 box 中心附近随机偏移一点
        offset_x = random.uniform(-large_region_size * 0.25, large_region_size * 0.25)
        offset_y = random.uniform(-large_region_size * 0.25, large_region_size * 0.25)

        region_cx = box_cx + offset_x
        region_cy = box_cy + offset_y

        left = int(region_cx - large_region_size / 2)
        top = int(region_cy - large_region_size / 2)

        # 防止越界
        left = max(0, min(left, img_w - large_region_size))
        top = max(0, min(top, img_h - large_region_size))

        right = min(left + large_region_size, img_w)
        bottom = min(top + large_region_size, img_h)

        region = image[top:bottom, left:right]

        if region.size > 0:
            mean_color = region.reshape(-1, 3).mean(axis=0).astype(np.uint8)
            image[top:bottom, left:right] = mean_color

    # =========================
    # 80%：原流程，只填充 box 区域
    # =========================
    else:
        for xmin, ymin, xmax, ymax in boxes:
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(img_w, xmax)
            ymax = min(img_h, ymax)

            if xmax <= xmin or ymax <= ymin:
                continue

            roi = image[ymin:ymax, xmin:xmax]

            if roi.size == 0:
                continue

            mean_color = roi.reshape(-1, 3).mean(axis=0).astype(np.uint8)
            image[ymin:ymax, xmin:xmax] = mean_color

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)


if __name__ == "__main__":
    xml_path = "/root/autodl-tmp/instruct/instruct-pix2pix/datasets/UAVPDD/Annotations/lr_00001_bottom_left.xml"
    image_path = "/root/autodl-tmp/instruct/instruct-pix2pix/datasets/UAVPDD/JPEGImages/lr_00001_bottom_left.jpg"

    result = fill_box_with_mean_color(xml_path, image_path)

    # 显示图片
    result.show()

    # 保存测试结果
    save_path = "/root/autodl-tmp/instruct/instruct-pix2pix/test_mean_fill.jpg"
    result.save(save_path)

    print(f"已保存到: {save_path}")