
from pycocotools.coco import COCO
import cv2
import os
import sys

## Paths
data_root = '/home/woody/iwi5/iwi5237h/VinDr-CXR/'
# /home/hpc/iwi5/iwi5237h/Few_Shot_DETR
ann_file = data_root + 'annotations.json'
img_dir = data_root + 'images/train/'  # ✅ Make sure this is correct
# 1. Load annotations
coco = COCO(ann_file)

# 2. Get all image IDs that contain category_id = 14 (for example)
category_id = 8
img_ids = coco.getImgIds(catIds=[category_id])
print(f"Found {len(img_ids)} images with category_id = {category_id}")

# 3. Take the first image ID
img_id = img_ids[0]
img_info = coco.loadImgs([img_id])[0]

# 4. Find the actual filename in img_dir by matching the ID
candidates = [f for f in os.listdir(img_dir) if f.startswith(img_id)]
if not candidates:
    print(f"No file in {img_dir} starts with {img_id}", file=sys.stderr)
    sys.exit(1)

filename = candidates[0]
img_path = os.path.join(img_dir, filename)
print(f"Using image file: {filename}")

# 5. Load the image
img = cv2.imread(img_path)
if img is None:
    print(f"cv2.imread failed on {img_path}", file=sys.stderr)
    sys.exit(1)

# 6. Prepare a cat_id → name map (so we don’t call loadCats() for every box)
cats = coco.loadCats(coco.getCatIds())
cat_id_to_name = {c['id']: c['name'] for c in cats}

# 7. Draw all bboxes for category_id = 14 on this image
ann_ids     = coco.getAnnIds(imgIds=[img_id], catIds=[category_id])
annotations = coco.loadAnns(ann_ids)

for ann in annotations:
    x, y, w, h = map(int, ann['bbox'])

    # Look up the actual category name
    this_cat_id   = ann['category_id']
    this_cat_name = cat_id_to_name[this_cat_id]

    # Draw the box (red) and put the correct label text
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(
        img,
        this_cat_name,                      # use the real name, not "Cardiomegaly" every time
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

# 8. Save the result
output_path = "outshow.jpg"
cv2.imwrite(output_path, img)
print(f"✅ Saved annotated image to: {output_path}")
