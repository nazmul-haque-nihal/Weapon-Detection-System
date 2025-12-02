# utils/voc_to_yolo.py
import os
import shutil
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET

CLASS_MAPPING = {'knife': 0, 'pistol': 1}

def parse_voc_xml(xml_path):
    """Parse VOC XML and return list of (class_id, bbox)"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        if size is None:
            return [], (0, 0)
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        boxes = []
        for obj in root.findall('object'):
            name_elem = obj.find('name')
            if name_elem is None:
                continue
            name = name_elem.text.strip()
            if name not in CLASS_MAPPING:
                continue

            class_id = CLASS_MAPPING[name]
            bndbox = obj.find('bndbox')
            if bndbox is None:
                continue

            try:
                xmin = int(float(bndbox.find('xmin').text))
                ymin = int(float(bndbox.find('ymin').text))
                xmax = int(float(bndbox.find('xmax').text))
                ymax = int(float(bndbox.find('ymax').text))
            except (ValueError, AttributeError, TypeError):
                continue

            # Clamp to image bounds
            xmin = max(0, min(xmin, width - 1))
            xmax = max(0, min(xmax, width - 1))
            ymin = max(0, min(ymin, height - 1))
            ymax = max(0, min(ymax, height - 1))

            if xmax <= xmin or ymax <= ymin:
                continue

            x_center = (xmin + xmax) / 2 / width
            y_center = (ymin + ymax) / 2 / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height

            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < w <= 1 and 0 < h <= 1):
                continue

            boxes.append((class_id, x_center, y_center, w, h))
        
        return boxes, (width, height)
    
    except Exception as e:
        print(f"❌ Failed to parse {xml_path}: {e}")
        return [], (0, 0)

def convert_dataset():
    img_train_dir = Path("data/Sohas_weapon-Detection/images")
    ann_train_dir = Path("data/Sohas_weapon-Detection/annotations/xmls")
    img_test_dir = Path("data/Sohas_weapon-Detection/images_test")
    ann_test_dir = Path("data/Sohas_weapon-Detection/annotations_test/xmls")

    out_train_img = Path("data/yolo_weapon/images/train")
    out_train_lbl = Path("data/yolo_weapon/labels/train")
    out_val_img = Path("data/yolo_weapon/images/val")
    out_val_lbl = Path("data/yolo_weapon/labels/val")

    for p in [out_train_img, out_train_lbl, out_val_img, out_val_lbl]:
        p.mkdir(parents=True, exist_ok=True)

    # Process training set
    processed = 0
    for xml_path in ann_train_dir.glob("*.xml"):
        stem = xml_path.stem
        # Try multiple extensions for image
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            candidate = img_train_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            print(f"⚠️ No image for {stem}")
            continue

        boxes, (w, h) = parse_voc_xml(xml_path)
        if not boxes or w == 0 or h == 0:
            continue

        # Copy image
        shutil.copy(img_path, out_train_img / img_path.name)

        # Write YOLO label
        with open(out_train_lbl / f"{stem}.txt", "w") as f:
            for cls_id, xc, yc, bw, bh in boxes:
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
        processed += 1

    # Process test set as val
    for xml_path in ann_test_dir.glob("*.xml"):
        stem = xml_path.stem
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            candidate = img_test_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            print(f"⚠️ No image for {stem}")
            continue

        boxes, (w, h) = parse_voc_xml(xml_path)
        if not boxes or w == 0 or h == 0:
            continue

        shutil.copy(img_path, out_val_img / img_path.name)
        with open(out_val_lbl / f"{stem}.txt", "w") as f:
            for cls_id, xc, yc, bw, bh in boxes:
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"✅ Converted {processed} training images + test as val")

if __name__ == "__main__":
    convert_dataset()