import os
import json

# Settings
IMAGE_FOLDER = "base_training/Aortic_enlargement"
JSON_PATH = "base_training/Aortic_enlargement/annotations_without_Aortic_enlargement.json"
EXCLUDED_CLASS = "Aortic_enlargement"  # Class you wanted to exclude

# Load annotation JSON
with open(JSON_PATH, 'r') as f:
    data = json.load(f)

# Get image filenames in the folder
image_filenames = set(os.listdir(IMAGE_FOLDER))
image_names = {os.path.splitext(f)[0] for f in image_filenames if f.lower().endswith(('.png', '.jpg', '.jpeg'))}

# Build image_id to file_name mapping (if needed)
image_id_to_filename = {img['id']: os.path.splitext(img['file_name'])[0] for img in data['images']}

# Reverse map: filename -> image_id
filename_to_image_id = {v: k for k, v in image_id_to_filename.items()}

# Get image_ids for the selected folder
selected_image_ids = {filename_to_image_id[name] for name in image_names if name in filename_to_image_id}

# Now, check annotations for these images
annotations = data['annotations']
images_with_class_X = []

for ann in annotations:
    if ann['image_id'] in selected_image_ids and ann['category_id'] == EXCLUDED_CLASS:
        images_with_class_X.append(ann['image_id'])

# Report
if not images_with_class_X:
    print("✅ All good! None of the selected images contain class X.")
else:
    print("❌ Found images that still contain class X annotations:")
    for img_id in set(images_with_class_X):
        print(f"- {image_id_to_filename[img_id]} (ID: {img_id})")
