import pandas as pd
import matplotlib.pyplot as plt
# import plotly.express as px              # interactive plotting library

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





# --- Display results ---
print("\nPer-class statistics:")
print(f"{'Class Name':25} {'# Unique Images':>15} {'# Annotations':>15}")
sum = 0
for class_name in fixed_classes.keys():
    print(f"{class_name:25} {class_image_counts[class_name]:15} {class_annotation_counts[class_name]:15}")
    sum += class_image_counts[class_name]

print(f"\nTotal unique images across all fixed classes: {sum}")

# # --- Plot unique image counts per class (ascending) --- using matplotlib ---




# --- Plot unique image counts per class (ascending) ---
sorted_counts = class_image_counts.sort_values(ascending=True)
total_images = sorted_counts.sum()

plt.figure(figsize=(10, 6))
bars = plt.barh(sorted_counts.index, sorted_counts.values, color='skyblue')

# Add labels (count + percentage) at the end of each bar
for bar in bars:
    width = bar.get_width()
    percent = (width / total_images) * 100
    plt.text(width + 1, bar.get_y() + bar.get_height() / 2,
             f'{int(width)} ({percent:.1f}%)',
             va='center', fontsize=9)

plt.title("Unique Images per Class (Ascending Order)")
plt.xlabel("Number of Unique Images")
plt.ylabel("Class Name")
plt.tight_layout()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

# # --- Plot unique image counts per class (ascending) with interactive hover ---
# sorted_counts = class_image_counts.sort_values(ascending=True).reset_index()
# sorted_counts.columns = ['Class Name', 'Unique Images']

# fig = px.bar(
#     sorted_counts,
#     x='Unique Images',
#     y='Class Name',
#     orientation='h',
#     title='Unique Images per Class (Ascending Order)',
#     labels={'Unique Images': 'Number of Unique Images', 'Class Name': 'Class Name'},
#     text='Unique Images'
# )

# fig.update_traces(marker_color='skyblue', textposition='outside')
# fig.update_layout(
#     yaxis=dict(categoryorder='total ascending'),
#     margin=dict(l=100, r=30, t=50, b=30),
#     xaxis=dict(showgrid=True, gridcolor='lightgray')
# )

# fig.show()
