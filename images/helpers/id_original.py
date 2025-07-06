import json

# --- CONFIGURATION ---
input_json = "fine_tune_input.json"  # Your fine-tuning JSON file (with 14 classes, remapped)
output_json = "fine_tune_with_restored_class.json"  # Output file
restored_class = {
    "id": 5,  # The original class ID to reintroduce
    "name": "Class_X",  # Replace with actual class name
    "supercategory": "none"  # Optional
}

# --- LOAD JSON ---
with open(input_json, "r") as f:
    coco = json.load(f)

# --- SHIFT BACK CATEGORY IDs ---
# First, shift existing annotations up by +1 if category_id >= 5
for ann in coco["annotations"]:
    if ann["category_id"] >= restored_class["id"]:
        ann["category_id"] += 1

# Same for categories
new_categories = []
for cat in coco["categories"]:
    new_cat = cat.copy()
    if cat["id"] >= restored_class["id"]:
        new_cat["id"] += 1
    new_categories.append(new_cat)

# --- ADD BACK THE REMOVED CATEGORY ---
new_categories.append(restored_class)

# Sort categories by ID for cleanliness
new_categories = sorted(new_categories, key=lambda x: x["id"])

# --- UPDATE JSON ---
coco["categories"] = new_categories

# --- SAVE JSON ---
with open(output_json, "w") as f:
    json.dump(coco, f)

print(f"✅ Fine-tuning JSON updated with restored category ID = {restored_class['id']} and saved to: {output_json}")
