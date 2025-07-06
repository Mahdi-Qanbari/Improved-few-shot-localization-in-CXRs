import pandas as pd

CSV_PATH = "images/annotations_test.csv"

# Load the CSV
df = pd.read_csv(CSV_PATH)

# # Count how many annotations exist for each class_name
# class_counts = df['class_name'].value_counts().sort_index()

# # Print the results
# print("Annotation counts per class:")
# for class_name, count in class_counts.items():
#     print(f"{class_name}: {count}")





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

# Filter to only fixed classes
df = df[df['class_name'].isin(fixed_classes)]

unique_images_total = df['image_id'].nunique()
print(f"\nTotal unique images across all fixed classes (no duplicates): {unique_images_total}")













