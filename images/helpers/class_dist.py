import json
from collections import Counter, defaultdict

def analyze_class_distribution(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    class_counts = Counter()

    for ann in data['annotations']:
        class_counts[ann['category_id']] += 1

    total = sum(class_counts.values())
    print(f"\n📊 Distribution in {json_path}: (Total Annotations: {total})")
    print("-" * 60)
    for cat_id in sorted(class_counts):
        name = cat_id_to_name[cat_id]
        count = class_counts[cat_id]
        percent = (count / total) * 100
        bar = "█" * int(percent // 2)
        print(f"{name:25} → {count:5} annotations  ({percent:5.2f}%)  {bar}")


# Example usage
analyze_class_distribution("../data1/train1.json")
analyze_class_distribution("../data1/val1.json")
