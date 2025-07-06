import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# --- CONFIGURATION ---
input_json_path = 'base_training/Atelectasis/annotations_without_Atelectasis.json'
train_output = "train_14cls.json"
val_output = "val_14cls.json"
val_size = 0.2
seed = 42
num_classes = 14
min_class_count = 5  # Not used directly, but consider checking post-split

# For image copying
image_root = Path("base_training/Atelectasis/")     # Folder with all VinDr-CXR images
train_img_dir = Path("data/train/")
val_img_dir = Path("data/val/")
train_img_dir.mkdir(parents=True, exist_ok=True)
val_img_dir.mkdir(parents=True, exist_ok=True)

# --- SET SEED ---
random.seed(seed)
np.random.seed(seed)

# --- LOAD COCO JSON ---
with open(input_json_path, 'r') as f:
    coco = json.load(f)

image_id_to_ann = defaultdict(list)
image_id_to_classes = defaultdict(set)

for ann in coco['annotations']:
    image_id_to_ann[ann['image_id']].append(ann)
    image_id_to_classes[ann['image_id']].add(ann['category_id'])

images = coco['images']
img_ids = [img['id'] for img in images]
img_id_to_info = {img['id']: img for img in images}

# --- BUILD MULTI-LABEL MATRIX ---
X = np.array(img_ids)
Y = np.zeros((len(X), num_classes), dtype=int)
id_to_idx = {img_id: idx for idx, img_id in enumerate(X)}

for img_id, class_ids in image_id_to_classes.items():
    for cls in class_ids:
        if cls < num_classes:
            Y[id_to_idx[img_id], cls] = 1

# --- STRATIFIED SPLIT ---
mskf = MultilabelStratifiedKFold(n_splits=int(1/val_size), shuffle=True, random_state=seed)
train_idx, val_idx = next(mskf.split(X, Y))

train_ids = set(X[train_idx])
val_ids = set(X[val_idx])

def split_coco(coco_data, image_ids):
    """Create a subset COCO dictionary."""
    images = [img for img in coco_data['images'] if img['id'] in image_ids]
    annots = [ann for ann in coco_data['annotations'] if ann['image_id'] in image_ids]
    return {
        'images': images,
        'annotations': annots,
        'categories': coco_data['categories']
    }

# --- SAVE SPLITS ---
with open(train_output, 'w') as f:
    json.dump(split_coco(coco, train_ids), f)

with open(val_output, 'w') as f:
    json.dump(split_coco(coco, val_ids), f)

print(f"✅ JSON done. Train: {len(train_ids)} images, Val: {len(val_ids)} images.")

# --- COPY IMAGES ---
def copy_images(image_ids, img_id_to_info, dest_dir):
    """Copy image files for the given image IDs."""
    for img_id in image_ids:
        file_name = img_id_to_info[img_id]['file_name']
        src_path = image_root / file_name
        dst_path = dest_dir / file_name
        if not dst_path.exists():
            shutil.copy(src_path, dst_path)

print("📁 Copying training images...")
copy_images(train_ids, img_id_to_info, train_img_dir)

print("📁 Copying validation images...")
copy_images(val_ids, img_id_to_info, val_img_dir)

print("✅ All images copied to train/ and val/ folders.")
