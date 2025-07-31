
import json
import random
import shutil
from pathlib import Path
from collections import defaultdict

# --- CONFIGURATION ---
input_json_path = 'TRAIN/train80p.json'
image_root = Path('TRAIN/train80p/')  # Folder with all images
fewshot_sizes = [15, 25, 35, 45, 55]
target_cls = 1  # The category_id you want few-shot samples from
# Atelectasis = 1     # Cardiomegaly = 3    # pneumothorax = 12    # Nodule/Mass = 8      # Lung Opacity = 7 # 


seed = 27

# --- Setup ---
random.seed(seed)

output_dir = Path(f'fewshot_class{target_cls}/')
output_dir.mkdir(parents=True, exist_ok=True)

# --- Load COCO JSON ---
with open(input_json_path, 'r') as f:
    coco = json.load(f)

# Map image_id → annotations and find images with the target class
img_id_to_ann = defaultdict(list)
img_id_to_has_target_class = dict()

for ann in coco['annotations']:
    img_id_to_ann[ann['image_id']].append(ann)
    if ann['category_id'] == target_cls:
        img_id_to_has_target_class[ann['image_id']] = True

# Filter images that contain the target class
target_images = [
    img for img in coco['images']
    if img['id'] in img_id_to_has_target_class
]

print(f"🔍 Found {len(target_images)} images containing class {target_cls}")

# --- Generate Few-Shot Subsets ---
for k in fewshot_sizes:
    selected_imgs = random.sample(target_images, k)
    selected_img_ids = {img['id'] for img in selected_imgs}

    # Filter annotations
    selected_annots = [ann for ann in coco['annotations'] if ann['image_id'] in selected_img_ids]

    # Save COCO JSON
    fewshot_json = {
        "images": selected_imgs,
        "annotations": selected_annots,
        "categories": coco['categories']
    }

    json_out_path = output_dir / f'fewshot_{k}.json'
    with open(json_out_path, 'w') as f:
        json.dump(fewshot_json, f)
    print(f"✅ Saved annotations to {json_out_path}")

    # Copy corresponding images
    img_out_dir = output_dir / f'images_{k}'
    img_out_dir.mkdir(parents=True, exist_ok=True)

    for img in selected_imgs:
        file_name = img['file_name']
        src_path = image_root / file_name
        dst_path = img_out_dir / file_name
        if not dst_path.exists():
            shutil.copy(src_path, dst_path)

    print(f"📁 Copied {k} images to {img_out_dir}")

print("✅ Few-shot extraction complete.")
