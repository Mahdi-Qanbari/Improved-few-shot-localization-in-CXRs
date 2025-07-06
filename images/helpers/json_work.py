import json
import os

# Load your COCO JSON annotation file
with open("vindrcxr_train.json", "r") as f:
    coco_data = json.load(f)

# Count number of unique images
num_images = len(coco_data.get("images", []))

print(f"Number of images in annotation file: {num_images}")



#############################

IMAGES_DIR = "train/"

# Load COCO JSON
with open("vindrcxr_train.json", "r") as f:
    coco_data = json.load(f)

# Extract image filenames (without extension)
json_images = set(img['file_name'].rsplit('.', 1)[0] for img in coco_data.get("images", []))

# List all image files in the folder (consider jpg and png)
folder_images = set()
for fname in os.listdir(IMAGES_DIR):
    if fname.lower().endswith(('.jpg', '.png')):
        folder_images.add(fname.rsplit('.', 1)[0])

# Find images in folder but missing in JSON
missing_images = folder_images - json_images

print(f"Number of images in folder: {len(folder_images)}")
print(f"Number of images in JSON: {len(json_images)}")
print(f"Images missing in JSON: {missing_images}")
