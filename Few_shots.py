import json
import os
import random
import shutil
from collections import defaultdict

# === Settings ===
FULL_ANN_FILE = "vindrcxr_train.json.json"     # your full annotation file
FULL_IMAGE_DIR = "train"              # your image folder
FEWSHOT_PER_CLASS = 5                  # how many few-shot images per class
OUTPUT_DIR = "output"                  # where to save results

# === Load full annotation file ===
with open(FULL_ANN_FILE) as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]
categories = data["categories"]

# === Mapping setup ===
cat_id_to_name = {c["id"]: c["name"] for c in categories}
image_to_anns = defaultdict(list)
class_to_image_ids = defaultdict(set)

for ann in annotations:
    image_to_anns[ann["image_id"]].append(ann)
    class_to_image_ids[ann["category_id"]].add(ann["image_id"])

# === Select few-shot image IDs ===
selected_image_ids = set()
for cat_id, image_ids in class_to_image_ids.items():
    sampled = random.sample(list(image_ids), min(FEWSHOT_PER_CLASS, len(image_ids)))
    selected_image_ids.update(sampled)

# === Split annotations ===
fewshot_images = [img for img in images if img["id"] in selected_image_ids]
fewshot_annots = [ann for ann in annotations if ann["image_id"] in selected_image_ids]

remaining_images = [img for img in images if img["id"] not in selected_image_ids]
remaining_annots = [ann for ann in annotations if ann["image_id"] not in selected_image_ids]

# === Prepare output folders ===
fewshot_dir = os.path.join(OUTPUT_DIR, "fewshot")
remaining_dir = os.path.join(OUTPUT_DIR, "remaining")

os.makedirs(os.path.join(fewshot_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(remaining_dir, "images"), exist_ok=True)

# === Save annotation files ===
with open(os.path.join(fewshot_dir, "annotations.json"), "w") as f:
    json.dump({
        "images": fewshot_images,
        "annotations": fewshot_annots,
        "categories": categories
    }, f, indent=2)

with open(os.path.join(remaining_dir, "annotations.json"), "w") as f:
    json.dump({
        "images": remaining_images,
        "annotations": remaining_annots,
        "categories": categories
    }, f, indent=2)

# === Copy images ===
def copy_images(image_list, dest_dir):
    for img in image_list:
        img_id = img["file_name"] if "file_name" in img else img["id"]
        src = os.path.join(FULL_IMAGE_DIR, img_id)
        dst = os.path.join(dest_dir, img_id)
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"⚠️ Warning: image not found: {src}")

copy_images(fewshot_images, os.path.join(fewshot_dir, "images"))
copy_images(remaining_images, os.path.join(remaining_dir, "images"))

# === Done ===
print(f"✅ Few-shot set: {len(fewshot_images)} images → {fewshot_dir}")
print(f"✅ Remaining set: {len(remaining_images)} images → {remaining_dir}")
