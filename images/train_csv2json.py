import pandas as pd
import json
from PIL import Image
import os

# === CONFIG ===
CSV_PATH = "train.csv"
IMAGES_DIR = "train/"
DEFAULT_WIDTH = 2048
DEFAULT_HEIGHT = 2048

# === Fixed Class Mapping
fixed_classes = {
    "Aortic enlargement": 0,
    "Atelectasis": 1,
    "Calcification": 2,
    "Cardiomegaly": 3,
    "Consolidation": 4,
    "ILD": 5,
    "Infiltration": 6,
    "Lung Opacity": 7,
    "Nodule/Mass": 8,
    "Other lesion": 9,
    "Pleural effusion": 10,
    "Pleural thickening": 11,
    "Pneumothorax": 12,
    "Pulmonary fibrosis": 13,
    "No finding": 14
}

# === COCO categories
categories = [
    {"id": cid, "name": cname}
    for cname, cid in fixed_classes.items()
]

# === Read CSV
df = pd.read_csv(CSV_PATH)

# Filter only valid classes (if needed)
df = df[df['class_id'].isin(range(15))]

# === Image metadata
image_id_map = {}
images = []
annotations = []
image_counter = 1
annotation_id = 1

for img_name in df['image_id'].unique():
    file_jpg = os.path.join(IMAGES_DIR, img_name + ".jpg")
    file_png = os.path.join(IMAGES_DIR, img_name + ".png")

    if os.path.exists(file_jpg):
        width, height = Image.open(file_jpg).size
    elif os.path.exists(file_png):
        width, height = Image.open(file_png).size
    else:
        width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT

    image_id_map[img_name] = image_counter
    images.append({
        "id": image_counter,
        "file_name": img_name + ".jpg",
        "width": width,
        "height": height
    })
    image_counter += 1

# === Annotations
for _, row in df.iterrows():
    img_id = image_id_map[row['image_id']]
    class_id = int(row['class_id'])

    if class_id == 14:  # No finding
        bbox = [0, 0, 1, 1]
        area = 1
    else:
        try:
            x_min = float(row['x_min'])
            y_min = float(row['y_min'])
            x_max = float(row['x_max'])
            y_max = float(row['y_max'])
        except:
            continue

        if any(pd.isna([x_min, y_min, x_max, y_max])):
            continue

        width = x_max - x_min
        height = y_max - y_min
        area = width * height

        if area <= 0:
            continue

        bbox = [x_min, y_min, width, height]

    annotations.append({
        "id": annotation_id,
        "image_id": img_id,
        "category_id": class_id,
        "bbox": bbox,
        "area": area,
        "iscrowd": 0
    })
    annotation_id += 1

# === Final COCO JSON
coco_json = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}

with open("vindrcxr_train.json", "w") as f:
    json.dump(coco_json, f, indent=2)
