import pandas as pd

CSV_PATH = "train.csv"

# Load the CSV
df = pd.read_csv(CSV_PATH)

# Fixed class mapping
fixed_classes = {
    "Aortic enlargement": 0,
    "Atelectasis": 1,
    "Calcification": 2,
    "Cardiomegaly": 3,
    "Consolidation": 4,
    "ILD": 5,
    "Infiltration": 6,
    "Lung Opacity": 7,
    "Nodule/Mass": 8,
    "Other lesion": 9,
    "Pleural effusion": 10,
    "Pleural thickening": 11,
    "Pneumothorax": 12,
    "Pulmonary fibrosis": 13,
    "No finding": 14
}

# Filter to only valid fixed classes
df = df[df['class_name'].isin(fixed_classes)]

# Count total unique images
unique_images_total = df['image_id'].nunique()
print(f"\nTotal unique images across all fixed classes: {unique_images_total}")

# --- Count unique images per class ---
class_image_counts = df.groupby('class_name')['image_id'].nunique().reindex(fixed_classes.keys(), fill_value=0)

# --- Count total annotations per class ---
class_annotation_counts = df['class_name'].value_counts().reindex(fixed_classes.keys(), fill_value=0)

# --- Display results ---
print("\nPer-class statistics:")
print(f"{'Class Name':25} {'# Unique Images':>15} {'# Annotations':>15}")
for class_name in fixed_classes.keys():
    print(f"{class_name:25} {class_image_counts[class_name]:15} {class_annotation_counts[class_name]:15}")
