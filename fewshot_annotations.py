##########   Fewshot data +  annotations    ###########

import json
from collections import defaultdict
import random
import os


data_root = '/home/woody/iwi5/iwi5237h/VinDr-CXR/'
# /home/hpc/iwi5/iwi5237h/Few_Shot_DETR
ann_file = data_root + 'annotations.json'
# img_dir = data_root + 'images/train/'  # ✅ Make sure this is correct

# Load full annotation file
with open(ann_file) as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]
categories = data["categories"]


print("Total images:", len(images))
print("Total annotations:", len(annotations))
print("Total categories:", len(categories))


# Step 3: Sample N Images per Class
# Set few-shot image count per class
few_shot_per_class = 5

# Map category_id to class name
cat_id_to_name = {c["id"]: c["name"] for c in categories}
name_to_cat_id = {v: k for k, v in cat_id_to_name.items()}

# 1. Map image_id → list of annotations
image_to_anns = defaultdict(list)
for ann in annotations:
    image_to_anns[ann["image_id"]].append(ann)

# 2. For each class, collect images containing that class
class_to_image_ids = defaultdict(set)
for ann in annotations:
    class_to_image_ids[ann["category_id"]].add(ann["image_id"])

# 3. Sample few-shot images per class
selected_image_ids = set()
for cat_id, image_ids in class_to_image_ids.items():
    sampled = random.sample(list(image_ids), min(few_shot_per_class, len(image_ids)))
    selected_image_ids.update(sampled)







# Step 4: Extract Few-Shot Images + Annotations
# Filter images
few_shot_images = [img for img in images if img["id"] in selected_image_ids]

# Filter annotations
few_shot_annots = [ann for ann in annotations if ann["image_id"] in selected_image_ids]

few_shot_data = {
    "images": few_shot_images,
    "annotations": few_shot_annots,
    "categories": categories
}

# # Save to new file
# with open("few_shot_vindr_cxr.json", "w") as f:
#     json.dump(few_shot_data, f, indent=2)





# Directory where files are saved
save_dir = data_root
prefix = "fewshot_annotations"

# Find the next available index
existing = [
    fname for fname in os.listdir(save_dir)
    if fname.startswith(prefix) and fname.endswith(".json")
]

# Extract numeric suffixes
existing_indexes = []
for fname in existing:
    try:
        index = int(fname.replace(prefix, "").replace(".json", ""))
        existing_indexes.append(index)
    except ValueError:
        pass

next_index = max(existing_indexes, default=0) + 1
filename = f"{prefix}{next_index}.json"
filepath = os.path.join(save_dir, filename)

# Save to file
with open(filepath, "w") as f:
    json.dump(few_shot_data, f, indent=2)

print(f"Few-shot dataset saved to: {filename}")
print(f"Total unique few-shot images: {len(selected_image_ids)}")





##############     Adding corresponding images to the folder "fewshot_images        #######################


import shutil

# Make sure target folder exists
fewshot_img_dir = os.path.join(data_root, "fewshot_images")
os.makedirs(fewshot_img_dir, exist_ok=True)

# Define source image folder (adjust if needed)
original_img_dir = os.path.join(data_root, "images/train")  # or your actual path

# Copy each selected image
for img in few_shot_images:
    img_id = img["id"]
    filename = f"{img_id}.jpg"

    src_path = os.path.join(original_img_dir, filename)
    dst_path = os.path.join(fewshot_img_dir, filename)

    if os.path.exists(src_path):
        shutil.copyfile(src_path, dst_path)
    else:
        print(f"❌ Warning: Image not found -> {src_path}")

print(f"✅ Copied {len(few_shot_images)} images to: {fewshot_img_dir}")
