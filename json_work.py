import json
import os

# Load annotation file
with open('./annotations.json', 'r') as f:
    data = json.load(f)

# Get all annotated image IDs
annotated_ids = set(ann['image_id'] for ann in data['annotations'])

# List of images in each set (assuming image filenames = image_id + '.jpg')
train_image_ids = set(f.split('.')[0] for f in os.listdir('./images/train'))
test_image_ids = set(f.split('.')[0] for f in os.listdir('./images/test'))
print(f"Train 1: {list(train_image_ids)[:2]}")
print(f"Test 1: {list(test_image_ids)[:2]}")
print(f"Annotated IDs: {list(annotated_ids)[:2]}")



# print(f"Annotated IDs: {list(annotated_ids)[:5]}")
# Count overlap
train_with_ann = train_image_ids & annotated_ids
test_with_ann = test_image_ids & annotated_ids

print(f"Annotated train images: {len(train_with_ann)} / {len(train_image_ids)}")
print(f"Annotated test images: {len(test_with_ann)} / {len(test_image_ids)}")
