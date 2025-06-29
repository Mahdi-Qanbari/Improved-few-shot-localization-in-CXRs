import json
import os
import shutil

# Paths
ANNOT_PATH = "vindrcxr_train.json"
IMAGES_DIR = "train"
OUTPUT_DIR = "training/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fixed class mapping
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

# Load COCO JSON
with open(ANNOT_PATH, 'r') as f:
    coco = json.load(f)

images_by_id = {img['id']: img for img in coco['images']}

for class_name, class_id in fixed_classes.items():
    safe_class_name = class_name.replace(" ", "_").replace("/", "-")
    save_dir = os.path.join(OUTPUT_DIR, safe_class_name)
    os.makedirs(save_dir, exist_ok=True)

    # ❌ Get image IDs that have the class to exclude
    excluded_image_ids = {ann['image_id'] for ann in coco['annotations'] if ann['category_id'] == class_id}

    # ✅ Filter images and annotations to EXCLUDE the class
    included_images = [img for img in coco['images'] if img['id'] not in excluded_image_ids]
    included_image_ids = {img['id'] for img in included_images}
    included_annotations = [ann for ann in coco['annotations'] if ann['image_id'] in included_image_ids]

    # Save new JSON
    output_json = {
        "images": included_images,
        "annotations": included_annotations,
        "categories": coco["categories"]
    }

    output_json_path = os.path.join(save_dir, f"annotations_without_{safe_class_name}.json")
    with open(output_json_path, "w") as f:
        json.dump(output_json, f, indent=2)

    # Copy images that remain
    for img in included_images:
        src = os.path.join(IMAGES_DIR, img['file_name'])
        dst = os.path.join(save_dir, img['file_name'])
        if os.path.exists(src):
            shutil.copy2(src, dst)

    print(f"✅ Excluding class: {class_name}")
    print(f"Saved {len(included_images)} images (excluded {len(excluded_image_ids)}) to: {save_dir}\n")
