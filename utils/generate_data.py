"""
Author: Vikram Sandu
Date: 2025-09-25
Description: 
    This script uses annotated data to generate synthetic data
    using copy-pase augmentation scheme. It randomly chooses annotated components
    from structural diagrams and pastes them in an empty white sheet.
"""
# Imports
import json
import os
from pathlib import Path
from PIL import Image
import random
from matplotlib import pyplot as plt
from tqdm import tqdm

from augmentations import augment_


def bbox_to_yolo(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    cx, cy = x1 + bw / 2, y1 + bh / 2
    return cx / img_w, cy / img_h, bw / img_w, bh / img_h


def resize_bboxes(labels, orig_size, new_size):
    """Resize YOLO-format bboxes when image is resized."""
    ow, oh = orig_size
    nw, nh = new_size
    scale_x, scale_y = nw / ow, nh / oh
    new_labels = []
    for cls, cx, cy, bw, bh in labels:
        # YOLO coords are relative, so first convert back to pixels
        cx, cy, bw, bh = cx * ow, cy * oh, bw * ow, bh * oh
        # resize
        cx, cy, bw, bh = cx * scale_x, cy * scale_y, bw * scale_x, bh * scale_y
        # back to YOLO format
        new_labels.append((cls, cx / nw, cy / nh, bw / nw, bh / nh))
    return new_labels


def place_component(canvas, comp_path, used_boxes):
    """Place component randomly, allowing small overlap probability"""
    comp = Image.open(comp_path).convert("RGBA")
    comp = augment_(comp)
    cw, ch = comp.size
    W, H = canvas.size
    max_x, max_y = W - cw, H - ch
    if max_x <= 0 or max_y <= 0:
        return None

    for _ in range(20):
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        bbox = (x, y, x + cw, y + ch)

        # check overlap
        overlap = any(
            not (bbox[2] <= ub[0] or bbox[0] >= ub[2] or bbox[3] <= ub[1] or bbox[1] >= ub[3])
            for ub in used_boxes
        )
        # allow overlap with small probability
        if not overlap:  # only accept non-overlap
            canvas.alpha_composite(comp, (x, y))
            return bbox
    return None


def extract_components(data_path: Path, save_path: Path):
    """
    Extract and Crop annotated components from the image
    given image and component coordinates.
    """
    # create base folder if it doesn't exist
    save_path.mkdir(parents=True, exist_ok=True)

    # create subfolders
    for sub in ["Component", "Table", "Others"]:
        (save_path / sub).mkdir(parents=True, exist_ok=True)

    with open(data_path / "annotations.json", "r") as f:
        data = json.load(f)

    # Loop through data.
    for idx, row in enumerate(data):
        # Get image name + Read
        image_name = row["file_upload"]
        # print(image_name)
        image = Image.open(data_path / "images" / image_name)

        # Get all annotations
        annotations = row["annotations"][0]["result"]

        # Loop through all annotated components
        for component in annotations:
            height, width = component["original_height"], component["original_width"]

            # Component Pixel coordinates.
            value = component["value"]
            x, y = int(value["x"] / 100 * width), int(value["y"] / 100 * height)
            c_width, c_height = int(value["width"] / 100 * width), int(value["height"] / 100 * height)

            # Label
            label = value["rectanglelabels"][0]

            # Crop and save.
            crop = image.crop((x, y, x + c_width, y + c_height))
            crop.save(save_path / label / f"{image_name}_{x}_{y}.png")


def generate_sheets(element_path: Path, save_path: Path,
                    canvas_size=(9934, 7016), output_size=(640, 640),
                    num_sheets: int = 2000, max_components_per_sheet: int = 30,
                    prefix=""
                    ):
    import shutil
    # If save_path exists, remove it and create fresh
    # if save_path.exists():
    #     shutil.rmtree(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    (save_path / "images/train").mkdir(parents=True, exist_ok=True)
    (save_path / "labels/train").mkdir(parents=True, exist_ok=True)

    # Assign class ids (exclude "Others" from labeling)
    subfolders = ["Component", "Table", "Others"]
    class_map = {cls: i for i, cls in enumerate(subfolders) if cls != "Others"}
    print("Class Map:", class_map)

    # Collect all component image paths per class
    components = {
        cls: list((element_path / cls).glob("*.png")) + list((element_path / cls).glob("*.jpg"))
        for cls in subfolders
    }

    # Probs
    class_probs = {
        "Component": 0.7,
        "Table": 0.25,
        "Others": 0.05
    }
    classes = list(class_probs.keys())
    probs = list(class_probs.values())

    # Generate dataset
    for i in tqdm(range(num_sheets)):
        # Random light shade for canvas (RGB 230-255)
        shade = random.randint(250, 255)
        canvas_color = (shade, shade, shade, 255)
        canvas = Image.new("RGBA", canvas_size, canvas_color)

        used_boxes, labels = [], []
        num_comp = random.randint(1, max_components_per_sheet)

        for _ in range(num_comp):
            cls = random.choices(classes, probs, k=1)[0]
            comp_path = random.choice(components[cls])
            result = place_component(canvas, comp_path, used_boxes)
            if result:
                bbox = result
                used_boxes.append(bbox)

                # Only add to labels if not "Others"
                if cls != "Others":
                    yolo_bbox = bbox_to_yolo(bbox, *canvas_size)
                    labels.append((class_map[cls], *yolo_bbox))

        # Save image
        img_out = save_path / "images/train" / f"sheet_{i}_{prefix}.png"
        lbl_out = save_path / "labels/train" / f"sheet_{i}_{prefix}.txt"

        # Resize and Save
        canvas_resized = canvas.resize(output_size, Image.Resampling.LANCZOS)
        canvas_resized.convert("RGB").save(img_out)

        # Save labels
        labels_resized = resize_bboxes(labels, canvas_size[::-1], output_size[::-1])
        with open(lbl_out, "w") as f:
            for lab in labels_resized:
                f.write(" ".join(map(str, lab)) + "\n")

    print("✅ Synthetic dataset generated with overlap, augmentations, and varied canvas shades at:", save_path)


def add_real_data(input_path: Path, output_path: Path, output_size=(640, 640)):
    import shutil
    # Input folders
    images_path = input_path / "images"
    json_path = input_path / "annotations.json"

    # Remove if exist
    # if (output_path / "images/train").exists():
    #     shutil.rmtree(output_path / "images/val")
    # if (output_path / "labels/train").exists():
    #     shutil.rmtree(output_path / "labels/val")

    # Output folders
    (output_path / "images/Level7").mkdir(parents=True, exist_ok=True)
    (output_path / "labels/Level7").mkdir(parents=True, exist_ok=True)

    # Load JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    # Assign class ids
    subfolders = ["Component", "Table"]
    class_map = {cls: i for i, cls in enumerate(subfolders)}

    for idx, row in enumerate(data):
    # for _ in os.listdir(images_path):
        # Image name + load
        image_name = row["file_upload"] # _
        image = Image.open(images_path / image_name).convert("RGB")
        orig_w, orig_h = image.size

        # Resize image to output_size
        image_resized = image.resize(output_size, Image.Resampling.LANCZOS)
        new_w, new_h = output_size

        # Collect labels
        annotations = row["annotations"][0]["result"]
        yolo_labels = []

        for component in annotations:
            value = component["value"]

            label_name = value["rectanglelabels"][0]
            if label_name == "Others":  # 🚫 skip Others
                continue

            # Original pixel coords
            x = int(value["x"] / 100 * orig_w)
            y = int(value["y"] / 100 * orig_h)
            c_w = int(value["width"] / 100 * orig_w)
            c_h = int(value["height"] / 100 * orig_h)

            # Bounding box in original image
            x1, y1 = x, y
            x2, y2 = x + c_w, y + c_h

            # Scale coords to new size
            scale_x, scale_y = new_w / orig_w, new_h / orig_h
            x1, y1 = x1 * scale_x, y1 * scale_y
            x2, y2 = x2 * scale_x, y2 * scale_y

            bw, bh = x2 - x1, y2 - y1
            cx, cy = x1 + bw / 2, y1 + bh / 2

            # Normalize for YOLO
            cx /= new_w
            cy /= new_h
            bw /= new_w
            bh /= new_h

            # Get class
            cls_id = class_map[label_name]

            yolo_labels.append((cls_id, cx, cy, bw, bh))

        # Save resized image
        img_out = output_path / "images/Level7" / f"{Path(image_name).stem}.png"
        image_resized.save(img_out)

        # Save labels
        lbl_out = output_path / "labels/Level7" / f"{Path(image_name).stem}.txt"
        with open(lbl_out, "w") as f:
            for lab in yolo_labels:
                f.write(" ".join(map(str, lab)) + "\n")

    print(f"✅ Saved {len(data)} images + labels to {output_path}")


# TODO: Generalize for multiple sheets.
if __name__ == "__main__":
    data_path = Path("../database/data/Residential_7_level/Struct")
    save_path = Path("../database/synthetic_data/elements/Residential_6_level/")

    # Step-1: Extract components and save.
    # extract_components(data_path, save_path)

    # Step-2: Generate sheets.
    sheet_save_path = Path("../database/synthetic_data/sheets/Residential_6_level/data_")
    # generate_sheets(element_path=save_path, save_path=sheet_save_path, prefix="level_6_", num_sheets=2000)

    # Step-3: Generate Validation Data
    data_val_path = Path("../database/data/Residential_7_level/Struct")
    add_real_data(data_val_path, sheet_save_path)
