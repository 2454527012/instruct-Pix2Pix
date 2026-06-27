import os
import cv2
import xml.etree.ElementTree as ET

# ================= 配置路径 =================
# 请根据你的实际路径修改这里
xml_dir = '/root/autodl-tmp/instruct/instruct-pix2pix/datasets/UAVPDD/Annotations'  # XML文件目录
image_dir = '/root/autodl-tmp/instruct/instruct-pix2pix/datasets/UAVPDD/JPEGImages' # 原图目录
mask_dir = '/root/autodl-tmp/instruct/instruct-pix2pix/datasets/UAVPDD/masks'        # 掩膜图目录

output_dir = '/root/autodl-tmp/instruct/instruct-pix2pix/datasets/UAVPDD/patched_images' # 结果保存目录
os.makedirs(output_dir, exist_ok=True)

# ================= 处理函数 =================
def process_dataset(xml_dir, img_dir, mask_dir, out_dir):
    xml_files = glob(os.path.join(xml_dir, '*.xml'))
    
    print(f"Found {len(xml_files)} XML files.")

    for xml_path in xml_files:
        # 解析 XML
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 获取文件名 (从XML的 <filename> 标签获取)
        filename_node = root.find('filename')
        if filename_node is None:
            print(f"Warning: No filename found in {xml_path}")
            continue
        
        filename = filename_node.text
        basename, _ = os.path.splitext(filename)
        
        # 构建原图和掩膜的路径
        # 假设原图是 .jpg，掩膜是 .png (根据你之前的代码)
        img_path = os.path.join(img_dir, filename)
        
        # 尝试查找掩膜文件（可能是jpg或png，这里优先尝试png，如果不存在尝试jpg）
        mask_path_png = os.path.join(mask_dir, basename + ".png")
        mask_path_jpg = os.path.join(mask_dir, basename + ".jpg")
        
        if os.path.exists(mask_path_png):
            mask_path = mask_path_png
        elif os.path.exists(mask_path_jpg):
            mask_path = mask_path_jpg
        else:
            print(f"Mask not found for {filename}, skipping...")
            continue

        # 读取图像
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # 读取为灰度图

        if image is None:
            print(f"Failed to read image: {img_path}")
            continue
        if mask is None:
            print(f"Failed to read mask: {mask_path}")
            continue

        img_h, img_w = image.shape[:2]

        # 遍历 XML 中的所有 object
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            if bndbox is None:
                continue

            try:
                # 获取坐标并转为整数
                xmin = int(float(bndbox.find('xmin').text))
                ymin = int(float(bndbox.find('ymin').text))
                xmax = int(float(bndbox.find('xmax').text))
                ymax = int(float(bndbox.find('ymax').text))
            except ValueError:
                print(f"Invalid coordinates in {xml_path}")
                continue

            # 边界检查，防止坐标超出图像范围
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(img_w, xmax)
            ymax = min(img_h, ymax)

            if xmax <= xmin or ymax <= ymin:
                continue

            # 核心操作：替换区域
            # 1. 提取掩膜的 ROI (Region of Interest)
            mask_roi = mask[ymin:ymax, xmin:xmax]
            
            # 2. 将单通道掩膜转换为 3通道 (BGR)，以便能赋值给 3通道 的原图
            # 如果不转换，直接赋值会因为维度不匹配报错，或者产生奇怪的颜色
            mask_roi_bgr = cv2.cvtColor(mask_roi, cv2.COLOR_GRAY2BGR)

            # 3. 替换原图中的对应区域
            image[ymin:ymax, xmin:xmax] = mask_roi_bgr
            
            # 可选：画一个矩形框标示替换区域（为了看清边界）
            # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        # 保存结果
        save_path = os.path.join(out_dir, filename)
        cv2.imwrite(save_path, image)
        print(f"Processed: {filename}")

# ================= 执行 =================
if __name__ == '__main__':
    from glob import glob # 需要导入glob
    process_dataset(xml_dir, image_dir, mask_dir, output_dir)
    print("All processing completed.")
