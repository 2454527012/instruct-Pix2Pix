import os
import json

# ================= 配置路径 =================
txt_file = '/root/autodl-tmp/instruct/instruct-pix2pix/datasets/UAVPDD/ImageSets/Main/train.txt'

edit_dir = '/root/autodl-tmp/instruct/instruct-pix2pix/datasets/UAVPDD/JPEGImages'
mask_dir = '/root/autodl-tmp/instruct/instruct-pix2pix/datasets/UAVPDD/masks'
box_dir = '/root/autodl-tmp/instruct/instruct-pix2pix/datasets/UAVPDD/Annotations'

output_jsonl = '/root/autodl-tmp/instruct/instruct-pix2pix/datasets/UAVPDD/dataset_train.jsonl'


def find_image_file(folder, basename):
    """自动查找 jpg/png/jpeg 文件"""
    for ext in [".jpg", ".png", ".jpeg"]:
        path = os.path.join(folder, basename + ext)
        if os.path.exists(path):
            return path
    return None


def generate_jsonl(txt_file, edit_dir, mask_dir, box_dir, output_jsonl):
    """生成只包含 edit_image、mask_image、box_xml 的 jsonl 文件"""

    with open(txt_file, "r", encoding="utf-8") as f:
        basenames = [line.strip() for line in f if line.strip()]

    print(f"Found {len(basenames)} filenames in txt file")

    total_count = 0
    missing_count = 0

    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        for basename in basenames:
            edit_path = find_image_file(edit_dir, basename)
            mask_path = find_image_file(mask_dir, basename)
            box_path = os.path.join(box_dir, basename + ".xml")

            if edit_path is None:
                print(f"Edit image not found: {basename}")
                missing_count += 1
                continue

            if mask_path is None:
                print(f"Mask image not found: {basename}")
                missing_count += 1
                continue

            if not os.path.exists(box_path):
                print(f"Box xml not found: {box_path}")
                missing_count += 1
                continue

            entry = {
                "edit_image": edit_path,
                "mask_image": mask_path,
                "box_xml": box_path,
                "prompt":"Road damage images acquired by unmanned aerial vehicles (UAVs)"
            }

            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            total_count += 1

            print(f"✓ {basename}")

    print(f"\n{'=' * 50}")
    print(f"Completed! Total: {total_count} entries")
    print(f"Missing: {missing_count} entries")
    print(f"Output saved to: {output_jsonl}")


if __name__ == "__main__":
    generate_jsonl(
        txt_file=txt_file,
        edit_dir=edit_dir,
        mask_dir=mask_dir,
        box_dir=box_dir,
        output_jsonl=output_jsonl,
    )