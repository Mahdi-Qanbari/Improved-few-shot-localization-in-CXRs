import json

# --- CONFIGURATION ---
input_json = "../data1/val1.json"  # Your input JSON file (train or val)
output_json = "../data1/val1_remapped.json"  # Output path
removed_cat_id = 1  # ID of the class you removed

# --- LOAD JSON ---
with open(input_json, "r") as f:
    coco = json.load(f)

# --- REMAP ANNOTATIONS ---
new_annotations = []
for ann in coco["annotations"]:
    cat_id = ann["category_id"]
    if cat_id == removed_cat_id:
        continue  # skip annotations for the removed class
    if cat_id > removed_cat_id:
        ann["category_id"] -= 1
    new_annotations.append(ann)

# --- REMAP CATEGORIES ---
new_categories = []
for cat in coco["categories"]:
    if cat["id"] == removed_cat_id:
        continue  # skip the removed class
    new_cat = cat.copy()
    if cat["id"] > removed_cat_id:
        new_cat["id"] -= 1
    new_categories.append(new_cat)

# --- REMAP DONE ---
coco["annotations"] = new_annotations
coco["categories"] = new_categories

# --- SAVE JSON ---
with open(output_json, "w") as f:
    json.dump(coco, f)

print(f"✅ Done. Saved remapped file to: {output_json}")
